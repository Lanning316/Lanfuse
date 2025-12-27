import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from einops import rearrange
from timm.models.layers import DropPath

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

class StarReLU(nn.Module):
    def __init__(self, scale_value=1.0, bias_value=0.0,
                 scale_learnable=True, bias_learnable=True,
                 mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1), requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1), requires_grad=bias_learnable)

    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias

class AttentionwithAttInv(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., lf_dy_weight=True,
                 hf_dy_weight=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.lf_dy_weight = lf_dy_weight
        self.hf_dy_weight = hf_dy_weight

        if self.lf_dy_weight:
            self.dy_freq_2 = nn.Linear(dim, self.num_heads, bias=True)
            self.lf_gamma = nn.Parameter(1e-5 * torch.ones((dim)), requires_grad=True)

        if self.hf_dy_weight:
            self.dy_freq = nn.Linear(dim, self.num_heads, bias=True)
            self.hf_gamma = nn.Parameter(1e-5 * torch.ones((dim)), requires_grad=True)

        self.dy_freq_starrelu = StarReLU()
        self.ignore_cls_token = 0

    def forward(self, x, H=None, W=None):
        B, N, C = x.shape
        dy_freq_feat = self.dy_freq_starrelu(x[:, self.ignore_cls_token:])

        dy_freq_lf = None
        if hasattr(self, 'dy_freq_2'):
            dy_freq_lf = self.dy_freq_2(dy_freq_feat).tanh_()
            dy_freq_lf = dy_freq_lf.reshape(B, N - self.ignore_cls_token, self.num_heads, 1).repeat(1, 1, 1,
                                                                                                    C // self.num_heads)
            dy_freq_lf = dy_freq_lf.reshape(B, N - self.ignore_cls_token, C)

        dy_freq = None
        if hasattr(self, 'dy_freq'):
            dy_freq = F.softplus(self.dy_freq(dy_freq_feat))
            dy_freq2 = dy_freq ** 2
            dy_freq = 2 * dy_freq2 / (dy_freq2 + 0.3678)
            dy_freq = dy_freq.reshape(B, N - self.ignore_cls_token, self.num_heads, 1).repeat(1, 1, 1,
                                                                                              C // self.num_heads)
            dy_freq = dy_freq.reshape(B, N - self.ignore_cls_token, C)
            if self.ignore_cls_token > 0:
                dy_freq = torch.cat([torch.zeros([B, self.ignore_cls_token, C], device=dy_freq.device), dy_freq], dim=1)

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_attn = (attn @ v).transpose(1, 2).reshape(B, N, C)

        v_reshape = v.permute(0, 2, 1, 3).reshape(B, N, C)
        v_hf = v_reshape - x_attn

        if hasattr(self, 'dy_freq_2'):
            x_attn = x_attn + x_attn * dy_freq_lf * self.lf_gamma.view(1, 1, -1)

        if hasattr(self, 'dy_freq'):
            x_attn = x_attn + dy_freq * v_hf * self.hf_gamma.view(1, 1, -1)

        x = self.proj(x_attn)
        x = self.proj_drop(x)
        return x
    
class CrossAttentionwithAttInv(nn.Module):
    """
    Cross-Attention version of AttentionwithAttInv (from fdam.py).

    输入:
      q_in:  (B, Nq, C)
      kv_in: (B, Nk, C)  # 通常 Nk == Nq
    输出:
      out:   (B, Nq, C)

    AttInv 核心:
      x_attn = Attn(q,k,v)
      v_hf   = v - x_attn   (当 Nk==Nq 时，逐 token 可减)
      x_attn += x_attn * dy_lf * lf_gamma
      x_attn += v_hf * dy_hf * hf_gamma
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale=None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        lf_dy_weight: bool = True,
        hf_dy_weight: bool = True,
        ignore_cls_token: int = 0,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Cross-Attn: Wq on q_in, Wk/Wv on kv_in
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # 动态频率权重（同 fdam.py）
        self.lf_dy_weight = lf_dy_weight
        self.hf_dy_weight = hf_dy_weight

        if self.lf_dy_weight:
            self.dy_freq_2 = nn.Linear(dim, self.num_heads, bias=True)
            self.lf_gamma = nn.Parameter(1e-5 * torch.ones((dim)), requires_grad=True)

        if self.hf_dy_weight:
            self.dy_freq = nn.Linear(dim, self.num_heads, bias=True)
            self.hf_gamma = nn.Parameter(1e-5 * torch.ones((dim)), requires_grad=True)

        # 用 StarReLU（和 fdam.py 对齐）；如果没导入成功，就 fallback 到普通 ReLU
        if StarReLU is not None:
            self.dy_freq_starrelu = StarReLU()
        else:
            self.dy_freq_starrelu = nn.ReLU(inplace=True)

        self.ignore_cls_token = int(ignore_cls_token)

    def forward(self, q_in: torch.Tensor, kv_in: torch.Tensor, H=None, W=None) -> torch.Tensor:
        """
        H/W 在这里不强依赖（主要给 freq_scale 用），保留接口对齐。
        """
        B, Nq, C = q_in.shape
        B2, Nk, C2 = kv_in.shape
        assert B == B2 and C == C2, "q_in and kv_in batch/channel must match"

        # -------- dy weight from q branch (更合理：另一模态的 query 决定强调什么频段) --------
        q_tokens = q_in[:, self.ignore_cls_token:] if self.ignore_cls_token > 0 else q_in
        dy_feat = self.dy_freq_starrelu(q_tokens)  # (B, Nq-ig, C)

        dy_freq_lf = None
        if hasattr(self, "dy_freq_2"):
            dy_freq_lf = self.dy_freq_2(dy_feat).tanh_()  # (B, Nq-ig, heads)
            dy_freq_lf = dy_freq_lf.reshape(B, Nq - self.ignore_cls_token, self.num_heads, 1)\
                                   .repeat(1, 1, 1, C // self.num_heads)\
                                   .reshape(B, Nq - self.ignore_cls_token, C)
            if self.ignore_cls_token > 0:
                dy_freq_lf = torch.cat(
                    [torch.zeros([B, self.ignore_cls_token, C], device=q_in.device, dtype=q_in.dtype),
                     dy_freq_lf],
                    dim=1
                )

        dy_freq_hf = None
        if hasattr(self, "dy_freq"):
            dy = F.softplus(self.dy_freq(dy_feat))  # (B, Nq-ig, heads)
            dy2 = dy ** 2
            dy = 2 * dy2 / (dy2 + 0.3678)          # fdam.py 同款归一化
            dy = dy.reshape(B, Nq - self.ignore_cls_token, self.num_heads, 1)\
                   .repeat(1, 1, 1, C // self.num_heads)\
                   .reshape(B, Nq - self.ignore_cls_token, C)
            if self.ignore_cls_token > 0:
                dy = torch.cat(
                    [torch.zeros([B, self.ignore_cls_token, C], device=q_in.device, dtype=q_in.dtype),
                     dy],
                    dim=1
                )
            dy_freq_hf = dy

        # -------- Cross-Attention --------
        q = self.q_proj(q_in).reshape(B, Nq, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # (B, h, Nq, d)
        k = self.k_proj(kv_in).reshape(B, Nk, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # (B, h, Nk, d)
        v = self.v_proj(kv_in).reshape(B, Nk, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # (B, h, Nk, d)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (B, h, Nq, Nk)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_attn = (attn @ v).transpose(1, 2).reshape(B, Nq, C)  # (B, Nq, C)

        # -------- Attention Inversion (AttInv) --------
        # 关键：这里要求 Nk == Nq（你当前融合正好是这样）
        if Nk == Nq:
            v_reshape = v.permute(0, 2, 1, 3).reshape(B, Nk, C)  # (B, N, C)
            v_hf = v_reshape - x_attn
        else:
            # 如果未来你搞了不同分辨率/不同 token 数，这里给个退化版本，避免直接炸掉：
            # 用 v 的均值作为“对应 token”的近似。
            v_mean = v.mean(dim=2, keepdim=False).permute(0, 2, 1).reshape(B, 1, C).expand(B, Nq, C)
            v_hf = v_mean - x_attn

        if dy_freq_lf is not None:
            x_attn = x_attn + x_attn * dy_freq_lf * self.lf_gamma.view(1, 1, -1)

        if dy_freq_hf is not None:
            x_attn = x_attn + dy_freq_hf * v_hf * self.hf_gamma.view(1, 1, -1)

        out = self.proj(x_attn)
        out = self.proj_drop(out)
        return out

class Mlp(nn.Module):
    def __init__(self, dim, mlp_ratio=4, out_features=None, act_layer=StarReLU, drop=0.,
                 bias=False, **kwargs):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class GroupDynamicScale(nn.Module):
    def __init__(self, dim, expansion_ratio=1, reweight_expansion_ratio=.125,
                 act1_layer=StarReLU, act2_layer=nn.Identity,
                 bias=False, num_filters=4, size=14, weight_resize=True, group=32, init_scale=1e-5,
                 **kwargs):
        super().__init__()
        self.size = size
        self.filter_size = size // 2 + 1
        self.num_filters = num_filters
        self.dim = dim
        self.group = group
        self.weight_resize = weight_resize

        self.reweight = Mlp(dim, reweight_expansion_ratio, group * num_filters, bias=False)

        self.complex_weights = nn.Parameter(
            torch.randn(num_filters, dim // group, self.size, self.filter_size, dtype=torch.float32) * init_scale
        )
        trunc_normal_(self.complex_weights, std=init_scale)
        self.act2 = act2_layer()

    def forward(self, x):
        # Expects x in [B, C, H, W]
        B, C, H, W = x.shape
        x_rfft = torch.fft.rfft2(x.to(torch.float32), dim=(2, 3), norm='ortho')
        B, C, RH, RW = x_rfft.shape

        x_spatial = x.permute(0, 2, 3, 1) # [B, H, W, C]
        routeing = self.reweight(x_spatial.mean(dim=(1, 2))).view(B, -1,
                                                                  self.num_filters).tanh_()

        weight = self.complex_weights
        if not weight.shape[2:4] == x_rfft.shape[2:4]:
            weight = F.interpolate(weight, size=x_rfft.shape[2:4], mode='bicubic', align_corners=True)

        weight = torch.einsum('bgf,fchw->bgchw', routeing, weight)
        weight = weight.reshape(B, C, RH, RW)

        x_rfft = torch.view_as_complex(torch.stack([x_rfft.real * weight, x_rfft.imag * weight], dim=-1))
        x = torch.fft.irfft2(x_rfft, s=(H, W), dim=(2, 3), norm='ortho')
        return x


class FDAMBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # 1. Replace standard Attention with AttentionwithAttInv
        self.attn = AttentionwithAttInv(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                        attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        # Standard MLP
        self.mlp = Mlp(dim, mlp_ratio=mlp_ratio, act_layer=StarReLU, drop=drop)  # Using StarReLU as per FDAM paper

        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

        # 2. Add GroupDynamicScale (FreqScale)
        # Note: 'size' here is a default, it will resize dynamically in forward if needed
        self.freq_scale_1 = GroupDynamicScale(dim=dim, group=16)
        self.freq_scale_2 = GroupDynamicScale(dim=dim, group=16)

    def forward(self, x, H, W):
        # Attention Path
        x_att = self.attn(self.norm1(x), H, W)

        # Apply FreqScale after Attention
        x_att_4d = to_4d(x_att, H, W)
        x_att_4d = self.freq_scale_1(x_att_4d) + x_att_4d
        x_att = to_3d(x_att_4d)

        x = x + self.drop_path(self.gamma_1 * x_att)

        # MLP Path
        x_mlp = self.mlp(self.norm2(x))

        # Apply FreqScale after MLP
        x_mlp_4d = to_4d(x_mlp, H, W)
        x_mlp_4d = self.freq_scale_2(x_mlp_4d) + x_mlp_4d
        x_mlp = to_3d(x_mlp_4d)

        x = x + self.drop_path(self.gamma_2 * x_mlp)
        return x