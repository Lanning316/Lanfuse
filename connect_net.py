import torch
import torch.nn as nn


# 提取为独立函数（模块级别）
def unpatchify(x, embed_dim=1024):
    """
    x: (N, L, embed_dim) 融合后的patch特征
    return: (N, embed_dim, h, w) 特征图（h*w = L）
    """
    h = w = int(x.shape[1] ** 0.5)
    assert h * w == x.shape[1], f"patch数量不匹配: {x.shape[1]} (需为h*w)"
    x = x.reshape(shape=(x.shape[0], h, w, embed_dim))
    x = torch.einsum('nhwc->nchw', x)  # 转换为通道优先格式
    return x

def patchify(imgs, embed_dim=1024):
    """
    imgs: (N, embed_dim, h, w) 特征图
    return: (N, L, embed_dim) patch特征（L = h*w）
    """
    h, w = imgs.shape[2], imgs.shape[3]
    x = torch.einsum('nchw->nhwc', imgs)  # 转换为通道最后格式
    x = x.reshape(shape=(imgs.shape[0], h * w, embed_dim))
    return x


class ConnectModel_open(nn.Module):
    def __init__(self, model1, model2, model3):
        super(ConnectModel_open, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        
        # 冻结第二个模型的权重，不进行梯度更新
        for param in self.model1.parameters():
            param.requires_grad = False
        for param in self.model2.parameters():
            param.requires_grad = False
        
    def split_(self,pred):#197可改
        patch_num = 196
        latent_new = pred[:,patch_num+1:,:]
        latent_judge = pred[:,:patch_num+1,:]
        return (latent_new+latent_judge)/2

   
    def unpatchify(self,x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, 1024))
        x = torch.einsum('nhwc->nchw', x)
        #imgs = x.reshape(shape=(x.shape[0], 4, h * p, h * p))
        return x

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        # p = 16
        # assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2]
        # x = imgs.reshape(shape=(imgs.shape[0], 4, h, p, w, p))
        x = torch.einsum('nchw->nhwc', imgs)
        x = x.reshape(shape=(imgs.shape[0], h * w, 1024))
        return x

    def forward(self, x):
        out1 = self.model2(x)
        latent_new  = self.split_(out1)
        img1 = self.unpatchify(latent_new[:,1:,:])
        img2 = self.model3(img1)
        # latent = self.patchify(img2)
        # out3 = self.model1.forward_decoder(latent)
        return img1, img2

class ConnectModel_MAE_Fuion(nn.Module):
    def __init__(self, mae, fusion_layer, mode='eval'):
        super(ConnectModel_MAE_Fuion, self).__init__()
        self.model1 = mae
        self.model2 = fusion_layer
        # for param in self.model1.parameters():
        #     param.requires_grad = False
        self.mode = mode
        self._set_trainable_blocks(self.mode)

    def _set_trainable_blocks(self, mode):
        if mode == 'train_decoder_MFM' or mode == 'train_decoder_CFM_MFM' or 'train_decoder_only':
            print("Open Decoder")
        else:
            print("Close Decoder")
            for param in self.model1.parameters():
                param.requires_grad = False

    def unpatchify(self, x):
        img = self.model1.unpatchify(x)
        return img

    #好像是融合验证集用的，可以先不看
    def val_forward(self, vi, ir):
        vi_latent = self.model1.forward_encoder(vi)
        ir_latent = self.model1.forward_encoder(ir)
        fusion    = self.model2(vi_latent,ir_latent)#,v,i
        out       = self.model1.forward_decoder(fusion)
        return out

    def forward(self, vi, ir):
        with torch.no_grad():
            vi_latent = self.model1.forward_encoder(vi)
            ir_latent = self.model1.forward_encoder(ir)
        fusion = self.model2(vi_latent,ir_latent)#,v,i
        if (self.mode == 'train_decoder_MFM' or self.mode == 'train_decoder_CFM_MFM' or self.mode == 'eval'
            or self.mode == 'train_CFM_mean' or self.mode == 'train_MFM_mean_CFM_lock' or self.mode == 'train_MFM_mean_CFM_open'):
            out = self.model1.forward_decoder(fusion)  # fusion
            return out
        else:
            return fusion



class ConnectModel_MAE(nn.Module):
    def __init__(self, mae):
        super(ConnectModel_MAE, self).__init__()
        self.model1 = mae
    
        
    def forward(self, img):
        with torch.no_grad():
            latent = self.model1.forward_encoder(img)

        out = self.model1.forward_decoder(latent) #fusion
        #out2  = self.model1.forward_decoder(fusion2)
        #out1  = self.model1.forward_decoder(v)
        #out2  = self.model1.forward_decoder(i)

        return out

class ConnectModel_MAE_Fuion_seg(nn.Module):
    def __init__(self, mae, fusion_layer, seg_module):
        super(ConnectModel_MAE_Fuion_seg, self).__init__()
        self.model1 = mae
        self.model2 = fusion_layer
        self.model3 = seg_module
        for param in self.model2.parameters():
            param.requires_grad = False
    def unpatchify(self, x):
        img = self.model1.unpatchify(x)
        return img
    
    def val_forward(self, vi, ir):
        vi_latent = self.model1.forward_encoder(vi)
        ir_latent = self.model1.forward_encoder(ir)
        fusion    = self.model2(vi_latent,ir_latent)[1]
        out       = self.model1.forward_decoder(fusion)
        seg_fusion = self.model3(fusion)

        return out,  seg_fusion#, seg_vi, seg_ir
    def forward(self, vi, ir):
        with torch.no_grad():
            vi_latent = self.model1.forward_encoder(vi)
            ir_latent = self.model1.forward_encoder(ir)
            patten,fusion    = self.model2(vi_latent,ir_latent)

        #out1       = self.model1.forward_decoder(patten)
        out2       = self.model1.forward_decoder(ir_latent)
        out2       = self.model1.unpatchify(out2)
        #seg_fusion1 = self.model3(patten)
        seg_fusion2 = self.model3(fusion)
        # seg_vi     = self.model3(v)
        # seg_ir     = self.model3(i)
        
        return out2,seg_fusion2#, out1, out2#, seg_vi, seg_ir
    
class ConnectModel_MAE_Fuion_2(nn.Module):
    def __init__(self, mae, fusion_layer):
        super(ConnectModel_MAE_Fuion_2, self).__init__()
        self.model1 = mae
        self.model2 = fusion_layer

    def forward(self, vi, ir):
        vi_latent = self.model1.forward_encoder(vi)
        ir_latent = self.model1.forward_encoder(ir)
        fusion,v,i    = self.model2(vi_latent,ir_latent)
        out      = self.model1.forward_decoder(fusion)
        out1     = self.model1.forward_decoder(v)
        out2     = self.model1.forward_decoder(i)

        return out, out1, out2

    
class ED(nn.Module):
    def __init__(self, mae, cnn):
        super(ED, self).__init__()
        self.encoder = mae
        self.decoder = cnn
        for param in self.encoder.parameters():
            param.requires_grad = False
    
        
    def forward(self, imgs):
        latent = self.encoder(imgs)
        out = self.decoder(latent)

        return out
    
    
class EFD(nn.Module):
    def __init__(self, ED, Fusion):
        super(EFD, self).__init__()
        self.encoder = ED.encoder
        self.decoder = ED.decoder
        self.fusion  = Fusion
        for param in self.decoder.parameters():
            param.requires_grad = False
            
    def val_forward(self, imgs_vi,imgs_ir):      
        
        latent_vi = self.encoder(imgs_vi)
        latent_ir = self.encoder(imgs_ir)
        latent = self.fusion(latent_vi,latent_ir)
        out = self.decoder(latent)
        return out
        
    def forward(self, imgs_vi,imgs_ir):      
        with torch.no_grad(): 
            latent_vi = self.encoder(imgs_vi)
            latent_ir = self.encoder(imgs_ir)
        latent = self.fusion(latent_vi,latent_ir)
        out = self.decoder(latent)
        return out


