import torch
import math
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Block
from util.pos_embed import get_2d_sincos_pos_embed
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert (
            self.head_dim * num_heads == embed_dim
        ), "Embedding dimension needs to be divisible by num_heads"

        self.values = nn.Linear(embed_dim, embed_dim)
        self.keys = nn.Linear(embed_dim, embed_dim)
        self.queries = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # Split input into self.num_heads different parts
        batch_size, seq_length, _ = x.shape
        values = self.values(x)
        keys = self.keys(x)
        queries = self.queries(x)

        values = values.view(batch_size, seq_length, self.num_heads, self.head_dim)
        keys = keys.view(batch_size, seq_length, self.num_heads, self.head_dim)
        queries = queries.view(batch_size, seq_length, self.num_heads, self.head_dim)

        values = values.permute(0, 2, 1, 3)
        keys = keys.permute(0, 2, 1, 3)
        queries = queries.permute(0, 2, 1, 3)

        energy = torch.matmul(queries, keys.permute(0, 1, 3, 2))
        attention = torch.nn.functional.softmax(energy, dim=-1)

        out = torch.matmul(attention, values)
        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_length, self.embed_dim)

        return out

#初版门控机制
# class ModalGate(nn.Module):
#     """A lightweight gating branch that adaptively re-weights modal features."""
#
#     def __init__(self, embed_dim: int, hidden_ratio: float = 0.5):
#         super().__init__()
#         hidden_dim = max(1, int(embed_dim * hidden_ratio))
#         self.mlp = nn.Sequential(
#             nn.Linear(embed_dim * 2, hidden_dim, bias=True),
#             nn.GELU(),
#             nn.Linear(hidden_dim, embed_dim * 2, bias=True),
#         )
#         self.activation = nn.Sigmoid()
#
#     def forward(self, vi_feat: torch.Tensor, ir_feat: torch.Tensor):
#         # vi_feat / ir_feat: (B, N, C)
#         gate_input = torch.cat([vi_feat, ir_feat], dim=-1)
#         modal_weights = self.activation(self.mlp(gate_input))
#         vi_gate, ir_gate = modal_weights.chunk(2, dim=-1)
#         return vi_feat * vi_gate, ir_feat * ir_gate




class ModalGate(nn.Module):
    """A lightweight gating branch that adaptively re-weights modal features."""

    def __init__(self, embed_dim: int, hidden_ratio: float = 0.5):
        super().__init__()
        hidden_dim = max(1, int(embed_dim * hidden_ratio))
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim * 2, bias=True),
        )

        # 关键初始化：
        # 使模块在训练开始时接近恒等映射 (feat * (1 + 0))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

        self.activation = nn.Sigmoid()
        # Depthwise smoothing keeps neighbouring tokens consistent while
        # preserving channel independence.
        self.spatial_smooth = nn.Conv2d(
            embed_dim * 2,
            embed_dim * 2,
            kernel_size=3,
            padding=1,
            groups=embed_dim * 2,
            bias=False,
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.constant_(self.spatial_smooth.weight, 1.0 / 9.0)

    def forward(self, vi_feat: torch.Tensor, ir_feat: torch.Tensor):
        # vi_feat / ir_feat: (B, N, C)
        gate_input = torch.cat([vi_feat, ir_feat], dim=-1)
        #modal_weights = self.activation(self.mlp(gate_input))
        modal_weights = self.mlp(gate_input)

        b, n, c2 = modal_weights.shape
        # Attempt to reshape into a 2D grid for spatial smoothing. If the
        # sequence length cannot form a square (e.g. includes cls tokens), we
        # skip the spatial filtering to avoid shape issues.
        grid_size = int(math.sqrt(n))
        if grid_size * grid_size == n:
            modal_weights = (
                modal_weights.view(b, grid_size, grid_size, c2)
                .permute(0, 3, 1, 2)
            )
            modal_weights = self.spatial_smooth(modal_weights)
            modal_weights = modal_weights.permute(0, 2, 3, 1).reshape(b, n, c2)

        vi_gate, ir_gate = modal_weights.chunk(2, dim=-1)

        #Tanh压缩门控信号到[-1, 1]
        vi_gate = torch.tanh(vi_gate)
        ir_gate = torch.tanh(ir_gate)

        #return vi_feat * vi_gate, ir_feat * ir_gate
        #应用门控，允许缩放范围为 [0, 2]
        return vi_feat * (1 + vi_gate), ir_feat * (1 + ir_gate)


#
#
# class ModalGate(nn.Module):
#     """Token-wise residual gating. Optional confidence injection."""
#     def __init__(self, embed_dim: int, hidden_ratio: float = 0.5, cred_alpha: float = 1.0):
#         super().__init__()
#         hidden_dim = max(1, int(embed_dim * hidden_ratio))
#         self.mlp = nn.Sequential(
#             nn.Linear(embed_dim * 2, hidden_dim, bias=True),
#             nn.GELU(),
#             nn.Linear(hidden_dim, embed_dim * 2, bias=True),
#         )
#         # start near identity
#         nn.init.zeros_(self.mlp[-1].weight)
#         nn.init.zeros_(self.mlp[-1].bias)
#
#         self.cred_alpha = float(cred_alpha)
#
#         self.spatial_smooth = nn.Conv2d(
#             embed_dim * 2, embed_dim * 2,
#             kernel_size=3, padding=1,
#             groups=embed_dim * 2, bias=False,
#         )
#         nn.init.constant_(self.spatial_smooth.weight, 1.0 / 9.0)
#
#     @staticmethod
#     def _bn1(x: torch.Tensor):
#         if x is None:
#             return None
#         if x.dim() == 2:
#             return x.unsqueeze(-1)  # (B,N,1)
#         if x.dim() == 3:
#             return x
#         raise ValueError(f"cred must be (B,N) or (B,N,1), got {x.shape}")
#
#     def forward(self, vi_feat: torch.Tensor, ir_feat: torch.Tensor,
#                 cred_vi: torch.Tensor = None, cred_ir: torch.Tensor = None):
#         # vi/ir: (B,N,C)
#         gate_input = torch.cat([vi_feat, ir_feat], dim=-1)  # (B,N,2C)
#         logits = self.mlp(gate_input)                       # (B,N,2C)
#
#         # optional spatial smoothing when N is square
#         b, n, c2 = logits.shape
#         gs = int(math.sqrt(n))
#         if gs * gs == n:
#             logits_2d = logits.view(b, gs, gs, c2).permute(0, 3, 1, 2)  # (B,2C,H,W)
#             logits_2d = self.spatial_smooth(logits_2d)
#             logits = logits_2d.permute(0, 2, 3, 1).reshape(b, n, c2)
#
#         vi_logit, ir_logit = logits.chunk(2, dim=-1)  # (B,N,C)
#
#         cred_vi = self._bn1(cred_vi)
#         cred_ir = self._bn1(cred_ir)
#         if (cred_vi is not None) and (cred_ir is not None):
#             vi_logit = vi_logit + self.cred_alpha * cred_vi
#             ir_logit = ir_logit + self.cred_alpha * cred_ir
#
#         vi_gate = torch.tanh(vi_logit)  # [-1,1]
#         ir_gate = torch.tanh(ir_logit)
#
#         return vi_feat * (1.0 + vi_gate), ir_feat * (1.0 + ir_gate)
#
#
# class TripleRouter(nn.Module):
#     """Route among [vi, ir, bilin] per token."""
#     def __init__(self, embed_dim: int, hidden_ratio: float = 0.5):
#         super().__init__()
#         hidden_dim = max(1, int(embed_dim * hidden_ratio))
#         self.mlp = nn.Sequential(
#             nn.Linear(embed_dim * 3, hidden_dim, bias=True),
#             nn.GELU(),
#             nn.Linear(hidden_dim, 3, bias=True),
#         )
#         # start near uniform
#         nn.init.zeros_(self.mlp[-1].weight)
#         nn.init.zeros_(self.mlp[-1].bias)
#
#     def forward(self, vi: torch.Tensor, ir: torch.Tensor, bilin: torch.Tensor,
#                 tau: float = 1.0, topk: int = 0):
#         x = torch.cat([vi, ir, bilin], dim=-1)  # (B,N,3C)
#         logits = self.mlp(x)                   # (B,N,3)
#         w = F.softmax(logits / max(tau, 1e-6), dim=-1)
#
#         if topk is not None and topk > 0 and topk < 3:
#             topv, topi = torch.topk(w, k=topk, dim=-1)
#             mask = torch.zeros_like(w).scatter_(-1, topi, 1.0)
#             w = w * mask
#             w = w / (w.sum(dim=-1, keepdim=True) + 1e-6)
#
#         return w, logits
#
#
# def cred_from_cos(vi: torch.Tensor, ir: torch.Tensor, eps: float = 1e-6):
#     """(B,N,1) in [0,1]"""
#     vi_n = F.normalize(vi, dim=-1, eps=eps)
#     ir_n = F.normalize(ir, dim=-1, eps=eps)
#     sim = (vi_n * ir_n).sum(dim=-1, keepdim=True)      # [-1,1]
#     return ((sim + 1.0) * 0.5).clamp(0.0, 1.0)
#
#
# def entropy_from_probs(p: torch.Tensor, eps: float = 1e-6):
#     p = p.clamp_min(eps)
#     return -(p * p.log()).sum(dim=-1, keepdim=True)
#
#
# def cred_from_attn_weights(attn_w: torch.Tensor, eps: float = 1e-6):
#     """
#     attn_w could be:
#       - (B, Nq, Nk)
#       - (Nq, Nk)
#       - (B, heads, Nq, Nk)  [rare in some configs]
#     return: (B, Nq, 1) in [0,1]
#     """
#     if attn_w.dim() == 2:  # (Nq,Nk)
#         p = attn_w.unsqueeze(0)  # (1,Nq,Nk)
#     elif attn_w.dim() == 3:  # (B,Nq,Nk)
#         p = attn_w
#     elif attn_w.dim() == 4:  # (B,H,Nq,Nk)
#         p = attn_w.mean(dim=1)
#     else:
#         raise ValueError(f"Unsupported attn_w shape: {attn_w.shape}")
#
#     # normalize entropy by log(Nk)
#     H = entropy_from_probs(p, eps=eps)                 # (B,Nq,1)
#     H = H / math.log(p.size(-1) + eps)                 # [0,1] approx
#     cred = (1.0 - H).clamp(0.0, 1.0)
#     return cred
#


class SE_Block(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super(SE_Block, self).__init__()
        # 全局平均池化(Fsq操作)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # 两个全连接层(Fex操作)
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),  # 从 c -> c/r
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )

    def forward(self, x):
        # 读取批数据图片数量及通道数
        b, c, h, w = x.size()
        # Fsq操作：经池化后输出b*c的矩阵
        y = self.gap(x).view(b, c)
        # Fex操作：经全连接层输出（b，c，1，1）矩阵
        y = self.fc(y).view(b, c, 1, 1)
        # Fscale操作：将得到的权重乘以原来的特征图x
        return x * y.expand_as(x)

class FPN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FPN, self).__init__()
        
        self.downsample1 = self.downsample_block(in_channels, in_channels)
        self.downsample2 = self.downsample_block(in_channels, in_channels)
        
        self.upsample2 = self.upsample_block(in_channels, out_channels)
        self.upsample1 = self.upsample_block(out_channels, out_channels)
        
    def downsample_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )
    
    def upsample_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        
        x1 = self.downsample1(x)
        x2 = self.downsample2(x1)
        
        x_up2 = self.upsample2(x2)
        x_up1 = self.upsample1(x_up2 + x1)
        
        return x_up1


class Feature_Net(nn.Module):
    def __init__(self, dim):
        super(Feature_Net, self).__init__()
        
        # 第一层卷积层：输入通道数1024，输出通道数2048，卷积核大小3x3，填充1
        self.conv1 = nn.Conv2d(in_channels=dim, out_channels=dim*2, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn1   = nn.BatchNorm2d(num_features=dim*2)
        
        # 第二层卷积层：输入通道数2048，输出通道数2048，卷积核大小3x3，填充1
        self.conv2 = nn.Conv2d(in_channels=dim*2, out_channels=dim*2, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.bn2   = nn.BatchNorm2d(num_features=dim*2)
        
        # 第三层卷积层：输入通道数2048，输出通道数1024，卷积核大小3x3，填充1
        self.conv3 = nn.Conv2d(in_channels=dim*2, out_channels=dim*2, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.bn3   = nn.BatchNorm2d(num_features=dim*2)
        
        # 第四层卷积层：输入通道数1024，输出通道数1024，卷积核大小3x3，填充1
        self.conv4 = nn.Conv2d(in_channels=dim*2, out_channels=dim, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        #self.bn4   = nn.BatchNorm2d(num_features=dim)
        
    def forward(self, x):
        # 前向传播过程
        x = self.bn1(self.relu1(self.conv1(x)))
        x = self.bn2(self.relu2(self.conv2(x)))
        x = self.bn3(self.relu3(self.conv3(x)))
        x = self.relu4(self.conv4(x))
        return x
    

class cross_fusion(nn.Module):  #tradition cross*2
    def __init__(self,embed_dim,img_size=640, patch_size=16, mode='eval'):
        super(cross_fusion, self).__init__()
        self.cross_model = nn.MultiheadAttention(embed_dim = embed_dim, num_heads = 16)
        self.merge_model = nn.MultiheadAttention(embed_dim = embed_dim, num_heads = 16)
        # --- 动态计算参数 ---
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2  # 例如 512/16=32, 32*32=1024
        # ------------------
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=False)
        self.embed_dim = embed_dim
        self.fc1 = nn.Linear(self.embed_dim, self.embed_dim*4, bias=True)
        self.fc2 = nn.Linear(self.embed_dim*4, self.embed_dim, bias=True)
        self.gelu = nn.GELU()
        self.ln = nn.LayerNorm(self.embed_dim)
        self.modal_gate = ModalGate(embed_dim)#门控
        self.initialize_weights()
        self.mode = mode
        self._set_trainable_blocks(self.mode)

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.embed_dim, self.grid_size, cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def _set_trainable_blocks(self, mode):
        #只训练MFM或者MFM和decoder，不训练CFM时，CFM冻结
        if mode == 'train_MFM_mean_CFM_lock' or mode == 'train_MFM_fusion_CFM_lock' or mode == 'train_decoder_MFM':
            for param in self.cross_model.parameters():
                param.requires_grad = False

    def forward(self, vi_latent, ir_latent):
        vi_latent += self.pos_embed
        ir_latent += self.pos_embed
        vi_patten,_  = self.cross_model(ir_latent,vi_latent,vi_latent)  #q,k,v
        ir_patten,_  = self.cross_model(vi_latent,ir_latent,ir_latent)
        patten = vi_patten + ir_patten
        if self.mode == 'train_CFM_mean' or self.mode == 'train_CFM_fusion':
            return patten
        vi_final,_   = self.merge_model(patten,vi_latent,vi_latent)
        ir_final,_   = self.merge_model(patten,ir_latent,ir_latent)
        # fusion       = (vi_final + ir_final)
        vi_gated, ir_gated = self.modal_gate(vi_final, ir_final)
        fusion = (vi_gated + ir_gated)

        ffn = self.fc2(self.gelu(self.fc1(fusion)))
        out = self.ln(ffn+fusion)
        return out

# class cross_fusion_ffn(nn.Module):  # tradition cross*2 + confidence + bilinear + routing
#     def __init__(
#         self,
#         embed_dim,
#         mode='eval',
#         # ---- new knobs ----
#         use_confidence: bool = True,
#         confidence_type: str = "cos",   # "attn_entropy" or "cos"
#         cred_alpha: float = 1.0,                 # how strong confidence affects ModalGate
#         use_bilinear: bool = True,
#         router_tau: float = 1.0,
#         router_topk: int = 0,                    # 0=softmax, 2=top2 routing
#         lambda_router_entropy: float = 0.0,      # >0 to add reg term
#     ):
#         super(cross_fusion_ffn, self).__init__()
#         self.cross_model = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=16)
#         self.merge_model = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=16)
#
#         self.pos_embed = nn.Parameter(torch.zeros(1, 1600, embed_dim), requires_grad=False)
#         self.embed_dim = embed_dim
#
#         self.fc1 = nn.Linear(self.embed_dim, self.embed_dim * 4, bias=True)
#         self.fc2 = nn.Linear(self.embed_dim * 4, self.embed_dim, bias=True)
#         self.gelu = nn.GELU()
#         self.ln = nn.LayerNorm(self.embed_dim)
#
#         # ---- new modules ----
#         self.modal_gate = ModalGate(embed_dim, hidden_ratio=0.5, cred_alpha=cred_alpha)
#
#         self.use_bilinear = bool(use_bilinear)
#         if self.use_bilinear:
#             self.u_proj = nn.Linear(embed_dim, embed_dim, bias=False)
#             self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
#
#         self.triple_router = TripleRouter(embed_dim, hidden_ratio=0.5)
#
#         # ---- knobs ----
#         self.use_confidence = bool(use_confidence)
#         self.confidence_type = confidence_type
#         self.router_tau = float(router_tau)
#         self.router_topk = int(router_topk)
#         self.lambda_router_entropy = float(lambda_router_entropy)
#
#         self.initialize_weights()
#         self.mode = mode
#         self._set_trainable_blocks(self.mode)
#
#     def initialize_weights(self):
#         pos_embed = get_2d_sincos_pos_embed(self.embed_dim, 40, cls_token=False)
#         self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
#
#     def _set_trainable_blocks(self, mode):
#         # 只训练MFM或者MFM和decoder，不训练CFM时，CFM冻结
#         if mode in ['train_MFM_mean_CFM_lock', 'train_MFM_fusion_CFM_lock', 'train_decoder_MFM']:
#             for param in self.cross_model.parameters():
#                 param.requires_grad = False
#
#     def _compute_confidence(self, vi_final: torch.Tensor, ir_final: torch.Tensor,
#                             vi_attn_w: torch.Tensor = None, ir_attn_w: torch.Tensor = None):
#         """
#         return: cred_vi, cred_ir (B,N,1) or (None,None)
#         """
#         if not self.use_confidence:
#             return None, None
#
#         if self.confidence_type == "attn_entropy" and (vi_attn_w is not None) and (ir_attn_w is not None):
#             # vi_attn_w: attention map when q=ir, k/v=vi  -> reflects how ir queries vi
#             # ir_attn_w: attention map when q=vi, k/v=ir
#             # We compute token cred on the query side (Nq == N).
#             cred_q_vi = cred_from_attn_weights(vi_attn_w)  # (B,N,1) (for ir queries)
#             cred_q_ir = cred_from_attn_weights(ir_attn_w)  # (B,N,1) (for vi queries)
#
#             # Use both (simple average) to be symmetric
#             cred = (cred_q_vi + cred_q_ir) * 0.5
#             return cred, cred
#
#         # fallback: cosine consistency
#         cred = cred_from_cos(vi_final, ir_final)
#         return cred, cred
#
#     def _router_entropy_reg(self, weights: torch.Tensor):
#         # weights: (B,N,3)
#         if self.lambda_router_entropy <= 0:
#             return None
#         ent = -(weights.clamp_min(1e-6) * weights.clamp_min(1e-6).log()).sum(dim=-1)  # (B,N)
#         return self.lambda_router_entropy * ent.mean()
#
#     def forward(self, vi_latent, ir_latent, return_gate_reg: bool = False, return_weights: bool = False):
#         """
#         vi_latent/ir_latent: (B,N,C)  你原代码就是这样用的
#         return_gate_reg: True 时返回 (out, gate_reg, weights)
#         """
#         vi_latent = vi_latent + self.pos_embed
#         ir_latent = ir_latent + self.pos_embed
#
#         # ---- CFM ----
#         # 说明：MultiheadAttention 默认输入是 (L,N,E) unless batch_first=True
#         # 你原来代码直接喂 (B,N,C) 也能跑，说明你项目里可能已经 batch_first=True 或者外部转置过
#         # 为了不破坏你现有行为，这里保持调用方式不变，仅显式 need_weights=True
#         vi_patten, vi_attn_w = self.cross_model(ir_latent, vi_latent, vi_latent, need_weights=True)
#         ir_patten, ir_attn_w = self.cross_model(vi_latent, ir_latent, ir_latent, need_weights=True)
#
#         patten = vi_patten + ir_patten
#         if self.mode in ['train_CFM_mean', 'train_CFM_fusion']:
#             return patten
#
#         # ---- MFM ----
#         vi_final, _ = self.merge_model(patten, vi_latent, vi_latent, need_weights=False)
#         ir_final, _ = self.merge_model(patten, ir_latent, ir_latent, need_weights=False)
#
#         # ---- confidence ----
#         cred_vi, cred_ir = self._compute_confidence(vi_final, ir_final, vi_attn_w, ir_attn_w)
#
#         # ---- confidence-aware ModalGate ----
#         vi_gated, ir_gated = self.modal_gate(vi_final, ir_final, cred_vi=cred_vi, cred_ir=cred_ir)
#
#         # ---- bilinear branch ----
#         if self.use_bilinear:
#             bilin = self.u_proj(vi_gated) * self.v_proj(ir_gated)
#         else:
#             bilin = vi_gated * ir_gated  # simple multiplicative
#
#         # ---- 3-way routing ----
#         weights, _ = self.triple_router(
#             vi_gated, ir_gated, bilin,
#             tau=self.router_tau,
#             topk=self.router_topk
#         )  # (B,N,3)
#
#         fusion = (
#             weights[..., 0:1] * vi_gated +
#             weights[..., 1:2] * ir_gated +
#             weights[..., 2:3] * bilin
#         )
#
#         # ---- FFN + LN ----
#         ffn = self.fc2(self.gelu(self.fc1(fusion)))
#         out = self.ln(ffn + fusion)
#
#         gate_reg = self._router_entropy_reg(weights)
#
#         if return_gate_reg and return_weights:
#             return out, gate_reg, weights
#         if return_gate_reg:
#             return out, gate_reg
#         if return_weights:
#             return out, weights
#         return out

    
class Seg_module(nn.Module):
    def __init__(self, embed_dim, img_size):
        super(Seg_module, self).__init__()
        #self.FPN = FPN(embed_dim, embed_dim)

        self.embed_dim = embed_dim
        self.decoder_embed_dim = 512
        self.patch_size = 16
        self.patch_embed = PatchEmbed(img_size, 16, 3, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.decoder_embed = nn.Linear(embed_dim, self.decoder_embed_dim, bias=True)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(self.decoder_embed_dim, 16, 4, qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm)
            for i in range(4)])

        self.decoder_norm = nn.LayerNorm(512)
        self.decoder_pred = nn.Linear(self.decoder_embed_dim, 16**2 * 3, bias=True)
    
    def decoder(self, x):
        x = self.decoder_embed(x)
        x = x + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        return x


    def unpatchify(self,x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, self.embed_dim))#1024
        x = torch.einsum('nhwc->nchw', x)
        
        return x

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        
        h = w = imgs.shape[2]
        x = torch.einsum('nchw->nhwc', imgs)
        x = x.reshape(shape=(imgs.shape[0], h * w, self.embed_dim))#1024
        return x
    
    def trans_mask(self, x):
        """
        x: (N, L, patch_size**2 *1)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0] # type: ignore
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def forward(self, latent):
        # imgs       = self.unpatchify(latent)
        # feature    = self.FPN(imgs)
        # new_latent = self.patchify(feature)
        seg_latent = self.decoder(latent)
        #mask       = self.trans_mask(seg_latent)
        return seg_latent
    

# class cross_fusion_mean(nn.Module):
#     def __init__(self,embed_dim):
#         super(cross_fusion_mean, self).__init__()
#         self.compare_model = nn.MultiheadAttention(embed_dim = embed_dim, num_heads = 16)
#         self.pos_embed = nn.Parameter(torch.zeros(1, 1600, embed_dim), requires_grad=False)
#         self.embed_dim = embed_dim
#         self.fc1 = nn.Linear(self.embed_dim, self.embed_dim*4, bias=True)
#         self.fc2 = nn.Linear(self.embed_dim*4, self.embed_dim, bias=True)
#         self.gelu = nn.GELU()
#         self.initialize_weights()
#
#     def initialize_weights(self):
#         pos_embed = get_2d_sincos_pos_embed(self.embed_dim, 40, cls_token=False)
#         self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
#
#     def forward(self, vi_latent, ir_latent):#, com_latent
#         #num         = vi_latent.shape[1]
#         vi_latent += self.pos_embed
#         ir_latent += self.pos_embed
#         mean_latant = (vi_latent+ir_latent)/2
#         vi_mean,_ = self.compare_model(mean_latant,vi_latent,vi_latent)
#         ir_mean,_ = self.compare_model(mean_latant,ir_latent,ir_latent)
#         latent = self.fc2(self.gelu(self.fc1(vi_mean+ir_mean)))
#
#         return latent
    
# class cross_fusion_FFN(nn.Module):  #tradition cross*2
#     def __init__(self,embed_dim):
#         super(cross_fusion_FFN, self).__init__()
#         self.cross_model = nn.MultiheadAttention(embed_dim = embed_dim, num_heads = 16)
#         self.merge_model = nn.MultiheadAttention(embed_dim = embed_dim, num_heads = 16)
#         self.pos_embed = nn.Parameter(torch.zeros(1, 1600, embed_dim), requires_grad=False)
#         self.embed_dim = embed_dim
#         self.fc1 = nn.Linear(self.embed_dim, self.embed_dim*4, bias=True)
#         self.fc2 = nn.Linear(self.embed_dim*4, self.embed_dim, bias=True)
#         self.gelu = nn.GELU()
#         self.ln = nn.LayerNorm(self.embed_dim)
#
#         self.initialize_weights()
#
#     def initialize_weights(self):
#         pos_embed = get_2d_sincos_pos_embed(self.embed_dim, 40, cls_token=False)
#         self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
#
#     def forward(self, vi_latent, ir_latent):#, com_latent
#         #num         = vi_latent.shape[1]
#         vi_latent += self.pos_embed
#         ir_latent += self.pos_embed
#         vi_patten,_  = self.cross_model(ir_latent,vi_latent,vi_latent)  #q,k,v
#         ir_patten,_  = self.cross_model(vi_latent,ir_latent,ir_latent)
#         patten = vi_patten + ir_patten
#         vi_final,_   = self.merge_model(patten,vi_latent,vi_latent)
#         ir_final,_   = self.merge_model(patten,ir_latent,ir_latent)
#         fusion    = (vi_final + ir_final)
#         ffn = self.fc2(self.gelu(self.fc1(fusion)))
#         out = self.ln(ffn+fusion)
#         return out

