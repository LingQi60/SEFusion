
# from matplotlib import image
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math
# from torchvision.models.vgg import vgg16
# import numpy as np
# from utils.utils_color import RGB_HSV, RGB_YCbCr
# from models.loss_ssim import ssim
# import torchvision.transforms.functional as TF

# class L_color(nn.Module):

#     def __init__(self):
#         super(L_color, self).__init__()

#     def forward(self, x ):

#         b,c,h,w = x.shape

#         mean_rgb = torch.mean(x,[2,3],keepdim=True)
#         mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
#         Drg = torch.pow(mr-mg,2)
#         Drb = torch.pow(mr-mb,2)
#         Dgb = torch.pow(mb-mg,2)
#         k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2),0.5)
#         return k

# class L_Grad(nn.Module):
#     def __init__(self):
#         super(L_Grad, self).__init__()
#         self.sobelconv=Sobelxy()

#     def forward(self, image_A, image_B, image_fused):
#         gradient_A = self.sobelconv(image_A)
#         gradient_B = self.sobelconv(image_B)
#         gradient_fused = self.sobelconv(image_fused)
#         gradient_joint = torch.max(gradient_A, gradient_B)
#         Loss_gradient = F.l1_loss(gradient_fused, gradient_joint)
#         return Loss_gradient
        
# class L_SSIM(nn.Module):
#     def __init__(self):
#         super(L_SSIM, self).__init__()
#         self.sobelconv=Sobelxy()

#     def forward(self, image_A, image_B, image_fused):
#         gradient_A = self.sobelconv(image_A)
#         gradient_B = self.sobelconv(image_B)
#         weight_A = torch.mean(gradient_A) / (torch.mean(gradient_A) + torch.mean(gradient_B))
#         weight_B = torch.mean(gradient_B) / (torch.mean(gradient_A) + torch.mean(gradient_B))
#         Loss_SSIM = weight_A * ssim(image_A, image_fused) + weight_B * ssim(image_B, image_fused)
#         return Loss_SSIM
# class Sobelxy(nn.Module):
#     def __init__(self):
#         super(Sobelxy, self).__init__()
#         kernelx = [[-1, 0, 1],
#                   [-2,0 , 2],
#                   [-1, 0, 1]]
#         kernely = [[1, 2, 1],
#                   [0,0 , 0],
#                   [-1, -2, -1]]
#         kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
#         kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
#         self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
#         self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
#     def forward(self,x):
#         sobelx=F.conv2d(x, self.weightx, padding=1)
#         sobely=F.conv2d(x, self.weighty, padding=1)
#         return torch.abs(sobelx)+torch.abs(sobely)

# class L_Intensity(nn.Module):
#     def __init__(self):
#         super(L_Intensity, self).__init__()

#     def forward(self, image_A, image_B, image_fused):        
#         intensity_joint = torch.max(image_A, image_B)
#         Loss_intensity = F.l1_loss(image_fused, intensity_joint)
#         return Loss_intensity


# class fusion_loss_vif(nn.Module):
#     def __init__(self):
#         super(fusion_loss_vif, self).__init__()
#         self.L_Grad = L_Grad()
#         self.L_Inten = L_Intensity()
#         self.L_SSIM = L_SSIM()

#         # print(1)
#     def forward(self, image_A, image_B, image_fused):
#         loss_l1 = 20 * self.L_Inten(image_A, image_B, image_fused)
#         loss_gradient = 20 * self.L_Grad(image_A, image_B, image_fused)
#         loss_SSIM = 10 * (1 - self.L_SSIM(image_A, image_B, image_fused))
#         fusion_loss = loss_l1 + loss_gradient + loss_SSIM
#         return fusion_loss, loss_gradient, loss_l1, loss_SSIM






from matplotlib import image
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models.vgg import vgg16
import numpy as np
from utils.utils_color import RGB_HSV, RGB_YCbCr
from models.loss_ssim import ssim
import torchvision.transforms.functional as TF

# # class L_color(nn.Module):

# #     def __init__(self):
# #         super(L_color, self).__init__()

# #     def forward(self, x ):

# #         b,c,h,w = x.shape

# #         mean_rgb = torch.mean(x,[2,3],keepdim=True)
# #         mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
# #         Drg = torch.pow(mr-mg,2)
# #         Drb = torch.pow(mr-mb,2)
# #         Dgb = torch.pow(mb-mg,2)
# #         k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2),0.5)

# #         k = torch.mean(k)
# #         return k
    

# class L_color(nn.Module):
#     def __init__(self):
#         super(L_color, self).__init__()

#     def forward(self, x ):
        
#         b,c,h,w = x.shape

#         r,g,b = torch.split(x , 1, dim=1)
#         mean_rgb = torch.mean(x,[2,3],keepdim=True)
#         mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
#         Dr = r-mr
#         Dg = g-mg
#         Db = b-mb
#         k =torch.pow( torch.pow(Dr,2) + torch.pow(Db,2) + torch.pow(Dg,2),0.5)        
 
#         k = torch.mean(k)

#         return k
    
# class L_exp(nn.Module):

#     def __init__(self,patch_size,mean_val):
#         super(L_exp, self).__init__()
#         # print(1)
#         self.pool = nn.AvgPool2d(patch_size)
#         self.mean_val = mean_val
#     def forward(self, x):

#         b,c,h,w = x.shape
#         x = torch.mean(x,1,keepdim=True)
#         mean = self.pool(x)

#         d = torch.mean(torch.pow(mean- torch.FloatTensor([self.mean_val] ).cuda(),2))
#         return d

# class L_Grad(nn.Module):
#     def __init__(self):
#         super(L_Grad, self).__init__()
#         self.sobelconv_3=Sobelxy_3()

#     def forward(self, image_A, image_B, image_fused):
#         gradient_A = self.sobelconv_3(image_A)
#         gradient_B = self.sobelconv_3(image_B)
#         gradient_fused = self.sobelconv_3(image_fused)
#         gradient_joint = torch.max(gradient_A, gradient_B)
#         Loss_gradient = F.l1_loss(gradient_fused, gradient_joint)
#         return Loss_gradient
        
# class L_SSIM(nn.Module):
#     def __init__(self):
#         super(L_SSIM, self).__init__()
#         self.sobelconv_3=Sobelxy_3()

#     def forward(self, image_A, image_B, image_fused):
#         gradient_A = self.sobelconv_3(image_A)
#         gradient_B = self.sobelconv_3(image_B)
#         weight_A = torch.mean(gradient_A) / (torch.mean(gradient_A) + torch.mean(gradient_B))
#         weight_B = torch.mean(gradient_B) / (torch.mean(gradient_A) + torch.mean(gradient_B))
#         Loss_SSIM = weight_A * ssim(image_A, image_fused) + weight_B * ssim(image_B, image_fused)
#         return Loss_SSIM

# # class Sobelxy(nn.Module):
# #     def __init__(self):
# #         super(Sobelxy, self).__init__()
# #         kernelx = [[-1, 0, 1],
# #                   [-2,0 , 2],
# #                   [-1, 0, 1]]
# #         kernely = [[1, 2, 1],
# #                   [0,0 , 0],
# #                   [-1, -2, -1]]
# #         kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
# #         kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
# #         self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
# #         self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
# #     def forward(self,x):
# #         sobelx=F.conv2d(x, self.weightx, padding=1)
# #         sobely=F.conv2d(x, self.weighty, padding=1)
# #         return torch.abs(sobelx)+torch.abs(sobely)

# class Sobelxy_3(nn.Module):
#     def __init__(self):
#         super(Sobelxy_3, self).__init__()
#         kernelx = [[-1, 0, 1],
#                   [-2,0 , 2],
#                   [-1, 0, 1]]
#         kernely = [[1, 2, 1],
#                   [0,0 , 0],
#                   [-1, -2, -1]]
#         kernelx = torch.FloatTensor(kernelx).unsqueeze(0).expand(3,-1,-1,-1)
#         kernely = torch.FloatTensor(kernely).unsqueeze(0).expand(3,-1,-1,-1)
#         self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
#         self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
#     def forward(self,x):
#         sobelx=F.conv2d(x, self.weightx, padding=1,groups=3)
#         sobely=F.conv2d(x, self.weighty, padding=1,groups=3)
#         return torch.abs(sobelx)+torch.abs(sobely)


# class L_Intensity(nn.Module):
#     def __init__(self):
#         super(L_Intensity, self).__init__()

#     def forward(self, image_A, image_B, image_fused):        
#         intensity_joint = torch.max(image_A, image_B)
#         Loss_intensity = F.l1_loss(image_fused, intensity_joint)
#         return Loss_intensity


# class fusion_loss_vif(nn.Module):
#     def __init__(self):
#         super(fusion_loss_vif, self).__init__()
#         # self.L_Grad = L_Grad()
#         self.L_Inten = L_Intensity()
#         # self.L_SSIM = L_SSIM()
#         self.L_color = L_color()
#         # self.L_exp = L_exp(8,0.6) # 4.23凌晨 第一参数是平均池化尺寸，大小应该是8 # 4.27 0.7->0.6 # 4.28 0.6——>0.5 # 4.29 0.5——>0.9 # 5.2 0.9->0.6

#         # print(1)
#     def forward(self, image_A, image_B, image_fused):
#         loss_l1 = 20 * self.L_Inten(image_A, image_B, image_fused) # 5.2 20->10  # 5.2 10->20
#         # # print(1111111111111111111111111111111111111)
#         # loss_gradient = 20 * self.L_Grad(image_A, image_B, image_fused) # 5.2 20->10  # 5.2 10->20
#         # # print(2222222222222222222222222222222222222)
#         # loss_SSIM = 10 * (1 - self.L_SSIM(image_A, image_B, image_fused)) # 4.30 10->5 5.1 5->10
#         # loss_color = 10 * self.L_color(image_fused)
#         # loss_exp = 10 * self.L_exp(image_fused) # 4.27 2->10
#         # print(3333333333333333333333333333333333333)
#         # fusion_loss = loss_l1 + loss_gradient + loss_SSIM + loss_color + loss_exp
#         # fusion_loss = loss_l1 + loss_gradient + loss_SSIM + loss_color
#         fusion_loss = loss_l1
#         return fusion_loss, loss_l1


# 梯度计算
class gradient(nn.Module):
    def __init__(self):
        super(gradient, self).__init__()


    def forward(self, img):
        smooth_kernel_x = torch.tensor([[0, 0], [-1, 1]], dtype=torch.float32).view(1, 1, 2, 2).cuda()
        smooth_kernel_y = smooth_kernel_x.transpose(2, 3)

        smooth_kernel_x = smooth_kernel_x.expand(1, img.size(1), 2, 2)
        smooth_kernel_y = smooth_kernel_y.expand(1, img.size(1), 2, 2)

        padding = img.size(2) // 2, img.size(3) // 2
        gradient_orig_x = torch.abs(F.conv2d(img, smooth_kernel_x, padding=padding))
        gradient_orig_y = torch.abs(F.conv2d(img, smooth_kernel_y, padding=padding))

        grad_min_x = torch.min(gradient_orig_x)
        grad_max_x = torch.max(gradient_orig_x)
        grad_min_y = torch.min(gradient_orig_y)
        grad_max_y = torch.max(gradient_orig_y)
        grad_norm_x = (gradient_orig_x - grad_min_x) / (grad_max_x - grad_min_x + 0.0001)
        grad_norm_y = (gradient_orig_y - grad_min_y) / (grad_max_y - grad_min_y + 0.0001)
        grad_norm = grad_norm_x + grad_norm_y
        return grad_norm


class contrast(nn.Module):
    def __init__(self):
        super(contrast, self).__init__()
    
    def forward(self, x):
        # 计算全局平均值
        mean_x = torch.mean(x, dim=[2, 3], keepdim=True)
        # 计算对比度
        c = torch.sqrt(torch.mean((x - mean_x) ** 2, dim=[2, 3], keepdim=True))
        return c


class angle(nn.Module):
    def __init__(self):
        super(angle, self).__init__()

    def forward(self, a, b):
        vector = torch.mul(a, b)
        # 计算分子
        up = torch.sum(vector)
        # 计算分母
        down = torch.sqrt(torch.sum(a ** 2)) * torch.sqrt(torch.sum(b ** 2))
        # 计算夹角的余弦值
        cos_theta = up / down
        # 得到夹角的弧度值
        theta = torch.acos(cos_theta)
        return theta


# 梯度损失
class L_gradient(nn.Module):
    def __init__(self):
        super(L_gradient, self).__init__()
        self.gradient = gradient()

    def forward(self, img_fusion, img_vi, img_ir):
        gradient_loss = torch.mean((self.gradient(img_fusion) - torch.max(self.gradient(img_ir), self.gradient(img_vi)))**2)
        return gradient_loss
    

class L_contrast(nn.Module):
    def __init__(self):
        super(L_contrast, self).__init__()
        self.contrast = contrast()

    def forward(self, img_fusion, img_vi, img_ir):
        # 匹配损失
        contrast_loss = torch.mean(torch.abs(self.contrast(img_fusion) - torch.max(self.contrast(img_vi), self.contrast(img_ir))))
        return contrast_loss


# 三角距离颜色损失
class L_color_angle(nn.Module):
    def __init__(self):
        super(L_color_angle, self).__init__()
        self.angle = angle()

    def forward(self, output_img, img_vi):
        color_angle_loss = torch.mean(self.angle(output_img[:,0,:,:], img_vi[:,0,:,:]) + self.angle(output_img[:,1,:,:], img_vi[:,1,:,:]) + self.angle(output_img[:,2,:,:], img_vi[:,2,:,:]))
        return color_angle_loss


class L_l1(nn.Module):
    def __init__(self):
        super(L_l1, self).__init__()
        self.angle = angle()

    def forward(self, img_fusion, img_ir):
        l1_loss = torch.mean(self.angle(img_fusion, img_ir))
        return l1_loss
    

# 曝光损失
# exposure_loss = torch.mean(torch.abs(Y_f - E))

# L1损失
# l1 = torch.mean(torch.abs(Y_f - ir))

# color_mutual_loss = torch.mean(torch.abs(1. / (contrast(If) + 0.0001)))

class fusion_loss_vif(nn.Module):
    def __init__(self):
        super(fusion_loss_vif, self).__init__()
        self.L_gradient = L_gradient()
        self.L_contrast = L_contrast()
        self.L_color_angle = L_color_angle()
        self.L_l1 = L_l1()
        # self.L_SSIM = L_SSIM()
        # self.L_color = L_color()
        # self.L_exp = L_exp(8,0.6) # 4.23凌晨 第一参数是平均池化尺寸，大小应该是8 # 4.27 0.7->0.6 # 4.28 0.6——>0.5 # 4.29 0.5——>0.9 # 5.2 0.9->0.6

        # print(1)
    def forward(self, image_vi_Y, image_ir, image_fusion, output_img, image_vi):
        # print(image_vi_Y.shape, image_ir.shape, image_fusion.shape, output_img.shape, image_vi.shape)
        gradient_loss = 200 * self.L_gradient(image_fusion, image_vi_Y, image_ir)
        contrast_loss = 10 * self.L_contrast(image_fusion, image_vi_Y, image_ir) # 1.1->10
        color_angle_loss = 0.5 * self.L_color_angle(output_img, image_vi) # 1.1->0.5
        l1_loss = 1.5 * self.L_l1(image_fusion, image_ir) # 2.5 ->1.5
        # # print(1111111111111111111111111111111111111)
        # loss_gradient = 20 * self.L_Grad(image_A, image_B, image_fused) # 5.2 20->10  # 5.2 10->20
        # # print(2222222222222222222222222222222222222)
        # loss_SSIM = 10 * (1 - self.L_SSIM(image_A, image_B, image_fused)) # 4.30 10->5 5.1 5->10
        # loss_color = 10 * self.L_color(image_fused)
        # loss_exp = 10 * self.L_exp(image_fused) # 4.27 2->10
        # print(3333333333333333333333333333333333333)
        # fusion_loss = loss_l1 + loss_gradient + loss_SSIM + loss_color + loss_exp
        # fusion_loss = loss_l1 + loss_gradient + loss_SSIM + loss_color
        fusion_loss = gradient_loss + contrast_loss + color_angle_loss + l1_loss
        return fusion_loss, gradient_loss, contrast_loss, color_angle_loss, l1_loss
        # fusion_loss = gradient_loss + contrast_loss + l1_loss
        # return fusion_loss, gradient_loss, contrast_loss, l1_loss