import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.losses


class SSIM(nn.Module):
    def __init__(self):
        super(SSIM, self).__init__()
        self.window_size  = 11
        self.window_sigma = 1.5
        self.k1           = 0.01
        self.k2           = 0.03

    def gaussian(self, x, y):
        window_size  = self.window_size
        window_sigma = torch.tensor(self.window_sigma)
        return torch.exp(-((x - window_size // 2) ** 2 + (y - window_size // 2) ** 2) / (2 * window_sigma ** 2))

    # def create_window(self):
    #     window_size  = self.window_size
    #     window = torch.tensor([[
    #         [self.gaussian(i, j) for j in range(window_size)]
    #         for i in range(window_size)
    #     ]], dtype=torch.float32)
    #
    #     return window / window.sum()

    def create_window(self):
        window_size = self.window_size
        # 生成二维高斯窗口（window_size x window_size）
        window = torch.tensor([[self.gaussian(i, j) for j in range(window_size)]
                               for i in range(window_size)], dtype=torch.float32)
        # 调整为4维张量：[out_channels=1, in_channels=1, kernel_h, kernel_w]
        window = window.unsqueeze(0).unsqueeze(0)  # 关键修改：增加两个维度
        return window / window.sum()  # 归一化窗口
    
    def rgb_to_y(self,rgb_image):
        
        r = rgb_image[:, 0, :, :]
        g = rgb_image[:, 1, :, :]
        b = rgb_image[:, 2, :, :]     
        y_image = 0.299 * r + 0.587 * g + 0.114 * b  
        
        return y_image

    #新增由于融合任务中使用的是灰度图（单通道），需要调整SSIM类以兼容单通道输入（原代码仅支持 3 通道 RGB 输入）
    def forward(self, y_pred, y_target):

        y_pred   = self.rgb_to_y(y_pred)
        y_target = self.rgb_to_y(y_target)

        window_size  = self.window_size
        k1           = self.k1 
        k2           = self.k2 

        C1 = (k1 * 255) ** 2
        C2 = (k2 * 255) ** 2

        mu_x = F.conv2d(y_pred, self.create_window(), padding=window_size // 2,stride=1)
        mu_y = F.conv2d(y_target, self.create_window(), padding=window_size // 2,stride=1)

        mu_x_mu_y = mu_x * mu_y
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)

        sigma_x_sq = F.conv2d(y_pred * y_pred, self.create_window(), padding=window_size // 2, stride=1) - mu_x_sq
        sigma_y_sq = F.conv2d(y_target * y_target, self.create_window(), padding=window_size // 2, stride=1) - mu_y_sq
        sigma_xy = F.conv2d(y_pred * y_target, self.create_window(), padding=window_size // 2,stride=1) - mu_x_mu_y

        ssim_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
        ssim_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)

        ssim_map = ssim_n / ssim_d
        return ssim_map.mean()

    def forward(self, y_pred, y_target):
        # 新增：判断输入是否为单通道，若为单通道则直接使用，否则转为Y通道
        if y_pred.size(1) == 3:  # 若为RGB三通道
            y_pred = self.rgb_to_y(y_pred)
            y_target = self.rgb_to_y(y_target)
        # 若为单通道（灰度图），直接使用
        else:
            y_pred = y_pred.squeeze(1)  # 移除通道维度，变为[N, H, W]
            y_target = y_target.squeeze(1)

        window_size = self.window_size
        k1 = self.k1
        k2 = self.k2

        C1 = (k1 * 255) ** 2
        C2 = (k2 * 255) ** 2

        # 确保窗口与输入在同一设备
        window = self.create_window().to(y_pred.device)

        mu_x = F.conv2d(y_pred.unsqueeze(1), window, padding=window_size // 2, stride=1)
        mu_y = F.conv2d(y_target.unsqueeze(1), window, padding=window_size // 2, stride=1)

        mu_x_mu_y = mu_x * mu_y
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)

        sigma_x_sq = F.conv2d(y_pred.unsqueeze(1) ** 2, window, padding=window_size // 2, stride=1) - mu_x_sq
        sigma_y_sq = F.conv2d(y_target.unsqueeze(1) ** 2, window, padding=window_size // 2, stride=1) - mu_y_sq
        sigma_xy = F.conv2d(y_pred.unsqueeze(1) * y_target.unsqueeze(1), window, padding=window_size // 2,
                            stride=1) - mu_x_mu_y

        ssim_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
        ssim_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)

        ssim_map = ssim_n / ssim_d
        return ssim_map.mean()

# def rgb_to_gray(images):
#     weight = torch.tensor([[[[0.2989]]],
#                            [[[0.5870]]],
#                            [[[0.1140]]]], dtype=torch.float32)
#
#     # Move the weight tensor to the same device as the images
#     weight = weight.to(images.device)
#
#     # Applying the depthwise convolution
#     gray_images = F.conv2d(images, weight=weight, groups=3)
#
#     # Since the output will have 3 separate channels, sum them up across the channel dimension
#     gray_images = gray_images.sum(dim=1, keepdim=True)
#
#     return gray_images

def rgb_to_gray(images):
    # images: [B,3,H,W], assume already normalized
    r, g, b = images[:, 0:1], images[:, 1:2], images[:, 2:3]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray



class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False)
        self.weighty = nn.Parameter(data=kernely, requires_grad=False)
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)

class Laplacian(nn.Module):
    def __init__(self):
        super(Laplacian, self).__init__()
        kernel = [[1, 1, 1],
                  [1, -8, 1],
                  [1, 1, 1]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, 3, 3]
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x):
        # Apply convolution with the Laplacian kernel
        laplacian = F.conv2d(x, self.weight, padding=1)
        return laplacian


class Fusionloss(nn.Module):
    def __init__(self,alpha,beta,gamma):
        super(Fusionloss, self).__init__()
        self.sobelconv = Sobelxy() 
        self.laplacianconv = Laplacian()
        # self.Loss_ssim = kornia.losses.SSIMLoss(11, reduction='mean')
        self.alpha = alpha  
        self.beta  = beta
        self.gamma = gamma

        # self.ssim = SSIM()  # 初始化SSIM计算模块
        # self.gamma = gamma  # SSIM损失权重

    def forward(self,generate_img,image_vis,image_ir):
        generate_img = generate_img.float()
        image_vis = image_vis.float()
        image_ir = image_ir.float()

        generate_img = generate_img / 255.0
        generate_img = generate_img.clamp(0, 1)
        image_vis = image_vis / 255.0
        image_ir = image_ir / 255.0

        # print(
        #     "vis:", image_vis.min().item(), image_vis.max().item(),
        #     "ir:", image_ir.min().item(), image_ir.max().item(),
        #     "gen:", generate_img.min().item(), generate_img.max().item()
        # )

        generate_img = rgb_to_gray(generate_img)
        image_vis = rgb_to_gray(image_vis)
        image_ir  = rgb_to_gray(image_ir)

        x_in_max = torch.max(image_vis, image_ir)
        loss_in = F.l1_loss(x_in_max, generate_img)

        y_grad            = self.sobelconv(image_vis)
        ir_grad           = self.sobelconv(image_ir)
        generate_img_grad = self.sobelconv(generate_img)
        x_grad_joint      = torch.max(y_grad,ir_grad)
        loss_grad         = F.l1_loss(x_grad_joint,generate_img_grad)
        
        y_laplacian            = torch.abs(self.laplacianconv(image_vis))
        ir_laplacian           = torch.abs(self.laplacianconv(image_ir))
        generate_img_laplacian = torch.abs(self.laplacianconv(generate_img))
        x_laplacian_joint      = torch.max(y_laplacian,ir_laplacian)
        loss_laplacian         = F.l1_loss(x_laplacian_joint,generate_img_laplacian)


        loss_total = self.alpha*loss_grad + self.beta*loss_laplacian + self.gamma*loss_in
        return loss_total,  loss_grad, loss_laplacian,  loss_in

        # loss_total = self.alpha * loss_grad + self.beta * loss_laplacian + self.gamma * loss_ssim
        # return loss_total, loss_grad, loss_laplacian, loss_ssim

