from matplotlib import image
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models.vgg import vgg16
import numpy as np
from utils.utils_color import RGB_HSV, RGB_YCbCr
import torchvision.transforms.functional as TF
    
class L_exp(nn.Module):

    def __init__(self,patch_size,mean_val):
        super(L_exp, self).__init__()
        # print(1)
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val
    def forward(self, x):

        b,c,h,w = x.shape
        x = torch.mean(x,1,keepdim=True)
        mean = self.pool(x)

        d = torch.mean(torch.pow(mean- torch.FloatTensor([self.mean_val] ).cuda(),2))
        return d

class gradient(nn.Module):
    def __init__(self):
        super(gradient, self).__init__()


    def forward(self, img):
        smooth_kernel_x = torch.tensor([[0, 0], [-1, 1]], dtype=torch.float32).view(1, 1, 2, 2).cuda()
        smooth_kernel_y = smooth_kernel_x.transpose(2, 3)

        smooth_kernel_x = smooth_kernel_x.expand(1, img.size(1), 2, 2)
        smooth_kernel_y = smooth_kernel_y.expand(1, img.size(1), 2, 2)

        padded_img = F.pad(img, (0, 1, 0, 1), 'constant', 0)
        gradient_orig_x = torch.abs(F.conv2d(padded_img, smooth_kernel_x, padding=0))
        gradient_orig_y = torch.abs(F.conv2d(padded_img, smooth_kernel_y, padding=0))

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
        mean_x = torch.mean(x, dim=[2, 3], keepdim=True)
        c = torch.sqrt(torch.mean((x - mean_x) ** 2, dim=[2, 3], keepdim=True))
        return c


class angle(nn.Module):
    def __init__(self):
        super(angle, self).__init__()

    def forward(self, a, b):
        vector = torch.mul(a, b)
        up = torch.sum(vector)
        down = torch.sqrt(torch.sum(a ** 2)) * torch.sqrt(torch.sum(b ** 2))
        cos_theta = up / down
        theta = torch.acos(cos_theta)
        return theta

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
        contrast_loss = torch.mean(torch.abs(self.contrast(img_fusion) - torch.max(self.contrast(img_vi), self.contrast(img_ir))))
        return contrast_loss

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
    
class fusion_loss_vif(nn.Module):
    def __init__(self):
        super(fusion_loss_vif, self).__init__()
        self.L_gradient = L_gradient()
        self.L_contrast = L_contrast()
        self.L_color_angle = L_color_angle()
        self.L_l1 = L_l1()
        self.L_exp = L_exp(8,0.7)

    def forward(self, image_vi_Y, image_ir, image_fusion, output_img, image_vi):
        gradient_loss = 200 * self.L_gradient(image_fusion, image_vi_Y, image_ir)
        contrast_loss = 15 * self.L_contrast(image_fusion, image_vi_Y, image_ir)
        color_angle_loss = 3 * self.L_color_angle(output_img, image_vi)
        l1_loss = 7.5 * self.L_l1(image_fusion, image_ir)
        loss_exp = 5 * self.L_exp(output_img)
        fusion_loss = gradient_loss + contrast_loss + color_angle_loss + l1_loss + loss_exp
        return fusion_loss, gradient_loss, contrast_loss, color_angle_loss, l1_loss, loss_exp
    
