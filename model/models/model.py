# Author: Ling Qi
# Date: 2024/4/5

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.CGFM import CGFM
from models.Decoder import ESF_decoder

def rgb2ycbcr(img_rgb):
    R = torch.unsqueeze(img_rgb[:, 0, :, :], 1)
    G = torch.unsqueeze(img_rgb[:, 1, :, :], 1)
    B = torch.unsqueeze(img_rgb[:, 2, :, :], 1)
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128/255.0
    Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128/255.0
    return Y, Cb, Cr


def color_recov(Yf, Cb, Cr):
    R = Yf + 1.402*(Cr - 128/255)
    G = Yf - 0.34414*(Cb - 128/255) - 0.71414*(Cr - 128/255)
    B = Yf + 1.772*(Cb - 128/255)
    output = torch.cat([R, G, B], dim=1)
    return output

class GradientFeature(nn.Module):
    def __init__(self):
        super(GradientFeature, self).__init__()

        self.smooth_kernel_x = torch.tensor([[0, 0], [-1, 1]], dtype=torch.float32).view(1, 1, 2, 2).cuda()
        # self.smooth_kernel_x = torch.tensor([[0, 0], [-1, 1]], dtype=torch.float32).view(1, 1, 2, 2)
        self.smooth_kernel_y = self.smooth_kernel_x.transpose(2, 3)

    def forward(self, img):
        self.smooth_kernel_x.requires_grad = False
        self.smooth_kernel_y.requires_grad = False
        self.smooth_kernel_x = self.smooth_kernel_x.expand(img.size(1), img.size(1), 2, 2)
        self.smooth_kernel_y = self.smooth_kernel_y.expand(img.size(1), img.size(1), 2, 2)
        padded_img = F.pad(img, (0, 1, 0, 1), 'constant', 0)
        gradient_orig_x = torch.abs(F.conv2d(padded_img, self.smooth_kernel_x, padding=0))
        gradient_orig_y = torch.abs(F.conv2d(padded_img, self.smooth_kernel_y, padding=0))     
        grad_min_x = torch.min(gradient_orig_x)
        grad_max_x = torch.max(gradient_orig_x)
        grad_min_y = torch.min(gradient_orig_y)
        grad_max_y = torch.max(gradient_orig_y)
        grad_norm_x = (gradient_orig_x - grad_min_x) / (grad_max_x - grad_min_x + 0.0001)
        grad_norm_y = (gradient_orig_y - grad_min_y) / (grad_max_y - grad_min_y + 0.0001)
        grad_norm = grad_norm_x + grad_norm_y

        return grad_norm
    

class Laplacian(nn.Module):
    def __init__(self):
        super(Laplacian, self).__init__()

        self.kernel = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], dtype=torch.float32).view(1, 1, 3, 3).cuda()
        # self.kernel = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], dtype=torch.float32).view(1, 1, 3, 3)

    def forward(self, img):
        self.kernel.requires_grad = False
        self.kernel = self.kernel.expand(img.size(1), img.size(1), 3, 3)
        gradient_orig = torch.abs(F.conv2d(img, self.kernel, padding=1))
        grad_min = torch.min(gradient_orig)
        grad_max = torch.max(gradient_orig)
        grad_norm = (gradient_orig - grad_min) / (grad_max - grad_min + 0.0001)

        return grad_norm
    
class Fusion_con(nn.Module):
    def __init__(self, in_chan=3, img_size=128, window_size=8, embed_dim=60, 
                 mlp_ratio=2, upscale=1, img_range=1, seq_length=2):
        super(Fusion_con, self).__init__()

        self.in_chan = in_chan
        self.img_size = img_size
        self.window_size = window_size
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.upscale = upscale
        self.img_range = img_range
        self.seq_length = seq_length    
        self.SFS = CGFM(img_size=self.img_size, patch_size=1, in_chans_img_a=self.in_chan, in_chans_img_b=self.in_chan, 
                                      out_chans=self.in_chan, embed_dim=self.embed_dim, Ex_depths=[4], Fusion_depths=[2, 2], Re_depths=[4], 
                                      Ex_num_heads=[8], Fusion_num_heads=[8, 8], Re_num_heads=[8], window_size=self.window_size, 
                                      mlp_ratio=self.mlp_ratio, qkv_bias=True, qk_scale=None, drop_rate=0., 
                                      attn_drop_rate=0., drop_path_rate=0.2, norm_layer=nn.LayerNorm, ape=False, 
                                      patch_norm=True, use_checkpoint=False, upscale=self.upscale, img_range=self.img_range, 
                                      upsampler='', resi_connection='1conv', seq_length=self.seq_length)   
        self.conv_after_body_Fusion = nn.Conv2d(2 * self.in_chan, self.in_chan, 3, 1, 1)
        self.conv_after_body_Fusion_channel = nn.Conv2d(2 * self.in_chan, self.in_chan, 3, 1, 1)
        self.conv_after_body_Fusion_final = nn.Conv2d(2 * self.in_chan, self.in_chan, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, lit_img,illu_fea, inf_img):
        lit_img, inf_img = self.SFS(lit_img, illu_fea, inf_img)
        fusion_img = torch.cat([lit_img, inf_img], 1)
        fusion_img = self.lrelu(self.conv_after_body_Fusion(fusion_img))

        return fusion_img
    
class fusion_decoder(nn.Module):
    def __init__(self, in_chan=12, img_size=32, window_size=2, embed_dim=64, 
                 mlp_ratio=2, upscale=1, img_range=1):
        super(fusion_decoder, self).__init__()

        self.in_chan = in_chan
        self.img_size = img_size
        self.window_size = window_size
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.upscale = upscale
        self.img_range = img_range
        self.SFD = ESF_decoder(img_size=self.img_size, patch_size=1, in_chans=self.in_chan, embed_dim=self.embed_dim, 
                                      Ex_depths=[4], Fusion_depths=[2, 2], Re_depths=[4], Ex_num_heads=[8], 
                                      Fusion_num_heads=[8, 8], Re_num_heads=[8], window_size=self.window_size, mlp_ratio=self.mlp_ratio, 
                                      qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., 
                                      drop_path_rate=0.3, norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                                      use_checkpoint=False, upscale=self.upscale, img_range=self.img_range, upsampler='', resi_connection='1conv')

    def forward(self, fusion_img, illu_fea):
        fusion_img = self.SFD(fusion_img, illu_fea)

        return fusion_img
    
class Encoder(nn.Module):
    def __init__(self, en_chan=2):
        super(Encoder, self).__init__()

        self.en_chan = en_chan
        self.conv1 = nn.Conv2d(self.en_chan, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.lrelu = nn.LeakyReLU()

    def forward(self, x):
        x = self.lrelu(self.bn1(self.conv1(x)))
        x = self.lrelu(self.bn2(self.conv2(x)))
        x = torch.sigmoid(self.bn3(self.conv3(x)))
        return x
    
class TextPre(nn.Module):
    def __init__(self):
        super(TextPre, self).__init__()

        self.w1_sobel = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding='same')
        self.w1_layer1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same')
        self.w1_layer2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same')
        self.w1_layer3 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding='same')
        self.bn_sobel = nn.BatchNorm2d(64)
        self.bn_layer1 = nn.BatchNorm2d(128)
        self.bn_layer2 = nn.BatchNorm2d(128)
        self.bn_layer3 = nn.BatchNorm2d(64)
        self.laplacian = Laplacian()
        self.gradient_feature = GradientFeature()

    def forward(self, feature_fusion):
        feature_fusion_laplacian = self.laplacian(feature_fusion)
        feature_new1 = feature_fusion + feature_fusion_laplacian
        feature_fusion_sobel = self.gradient_feature(feature_fusion)
        feature_fusion_sobel_new = self.bn_sobel(self.w1_sobel(feature_fusion_sobel))
        conv1 = F.leaky_relu(self.bn_layer1(self.w1_layer1(feature_new1)))
        conv2 = F.leaky_relu(self.bn_layer2(self.w1_layer2(conv1)))
        conv3 = self.bn_layer3(self.w1_layer3(conv2))
        feature_fusion_gradient = torch.cat((conv3, feature_fusion_sobel_new), 1)
        return feature_fusion_gradient, feature_fusion_sobel_new        
    
class Enhancement(nn.Module):
    def __init__(self):
        super(Enhancement, self).__init__()

        self.conv1 = nn.Conv2d(128, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(128, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(128, 32, kernel_size=7, padding=3)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(128, 32, kernel_size=1, padding=0)
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(128, 64, kernel_size=1, padding=0)
        self.bn5 = nn.BatchNorm2d(64)
        self.lrelu = nn.LeakyReLU()

    def forward(self, fusion_img):
        B, C, H, W = fusion_img.shape
        conv1 = self.lrelu(self.bn1(self.conv1(fusion_img)))
        conv2 = self.lrelu(self.bn2(self.conv2(fusion_img)))
        conv3 = self.lrelu(self.bn3(self.conv3(fusion_img)))
        conv4 = self.lrelu(self.bn4(self.conv4(fusion_img)))
        feature_multiscale = torch.cat([conv1, conv2, conv3, conv4], dim=1)
        feature_shuffle = feature_multiscale
        mean_vector = torch.mean(feature_shuffle, dim=[2, 3], keepdim=True)
        feature_contrast = torch.sqrt(torch.mean((feature_shuffle - mean_vector) ** 2, dim=[2, 3], keepdim=True))
        contrast_vector = torch.mean(feature_contrast, dim=[2, 3], keepdim=True)
        feature_fusion_enhancement = contrast_vector * feature_shuffle
        contrast_vector = self.lrelu(self.bn5(self.conv5(contrast_vector))).expand(-1, -1, H, W)

        return feature_fusion_enhancement, contrast_vector

class Decoder(nn.Module):
    def __init__(self, de_chan=2):
        super(Decoder, self).__init__()

        self.de_chan = de_chan
        self.conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, self.de_chan, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(self.de_chan)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        x = self.lrelu(self.bn1(self.conv1(x)))
        x = self.lrelu(self.bn2(self.conv2(x)))
        x = torch.sigmoid(self.bn3(self.conv3(x)))

        return x
    
class Fusion_block(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, img_size=128, fusionblock_num=3, 
                 window_size=8, embed_dim=64, img_range=1, mlp_ratio=2, 
                 upsampler='', seq_length=2, upscale=1):
        super(Fusion_block, self).__init__()

        self.window_size = window_size
        self.in_channels = in_channels
        self.fusionblock_num = fusionblock_num
        self.img_range = img_range
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.mlp_ratio = mlp_ratio
        self.seq_length = seq_length
        self.upscale = upscale
        self.fusion_layers = nn.ModuleList([])
        for i in range(fusionblock_num):
            self.fusion_layers.append(nn.ModuleList([
                Encoder(en_chan=self.in_channels),
                TextPre(),
                Enhancement(),
                Decoder(de_chan=self.in_channels),
                Fusion_con(in_chan=self.in_channels, img_size=int(self.img_size / (2**(i + 1))), window_size=int(self.window_size / (2**i)),
                           embed_dim=self.embed_dim, mlp_ratio=self.mlp_ratio, upscale=self.upscale, img_range=1, seq_length=self.seq_length),
                nn.Conv2d(self.in_channels, self.in_channels * 2, 4, 2, 1, bias=False),
                nn.Conv2d(self.in_channels, self.in_channels * 2, 4, 2, 1, bias=False)
            ]))
            self.in_channels *= 2
            window_size = int(self.window_size / (2**i))
        self.fusion_decoder_layers = nn.ModuleList([])
        self.in_channels //= 2
        for i in range(fusionblock_num):
            self.fusion_decoder_layers.append(nn.ModuleList([
                fusion_decoder(in_chan=self.in_channels, img_size=int(self.img_size * (2**i)), window_size=int(window_size * (2**i))),
                nn.ConvTranspose2d(self.in_channels, self.in_channels // 2, stride=2, 
                                   kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(self.in_channels, self.in_channels // 2, 1, 1, bias=False),               
            ]))
            self.in_channels //= 2

    def forward(self, lit_img, inf_img_1):
        text_pre_list = []
        fusion_img_list = []
        lit_img_2 = torch.cat([lit_img, lit_img], dim=1)
        inf_img_2 = torch.cat([inf_img_1, inf_img_1], dim=1)
        #Encoder
        for (Encoder, Text_pre, Enhancement, Decoder, Fusion_con, LitImgDownSample, InfImgDownSample) in self.fusion_layers:
            lit_img_2 = Encoder(lit_img_2)
            lit_img_2, text_pre = Text_pre(lit_img_2)
            lit_img_2, contrast_vector = Enhancement(lit_img_2)
            lit_img_2 = Decoder(lit_img_2)
            fusion_img = Fusion_con(lit_img_2, contrast_vector, inf_img_2)
            text_pre_list.append(text_pre)
            fusion_img_list.append(fusion_img)
            lit_img_2 = LitImgDownSample(lit_img_2)
            inf_img_2 = InfImgDownSample(inf_img_2)
        # Decoder
        for i, (FusionDecoder, ImgUpSample, ImgConcat) in enumerate(self.fusion_decoder_layers):
            fusion_img = FusionDecoder(fusion_img, text_pre_list[self.fusionblock_num - 1 - i])
            if i == self.fusionblock_num - 1:
                break
            fusion_img = ImgUpSample(fusion_img)
            fusion_img = torch.cat([fusion_img, fusion_img_list[self.fusionblock_num - 2  - i]], dim=1)
            fusion_img = ImgConcat(fusion_img)

        return fusion_img

class SEFusion(nn.Module):
    def __init__(self, img_size=128, window_size=8, img_range=1., in_channels=2, 
                 out_channels=2, n_feat=60, embed_dim=64, mlp_ratio=2, 
                 upsampler='', seq_length=1, fusionblock_num=3, upscale=2):
        super(SEFusion, self).__init__()
        
        self.window_size = window_size
        self.fusion_block = Fusion_block(in_channels=in_channels, out_channels=out_channels, img_size=img_size, 
                                         embed_dim=embed_dim, fusionblock_num=fusionblock_num, window_size=self.window_size,
                                         img_range=img_range, mlp_ratio=mlp_ratio, upsampler=upsampler, seq_length=seq_length,)
        
    def forward(self, vis_img, inf_img):
        input_vis_img_y, input_vis_img_Cb, input_vis_img_Cr = rgb2ycbcr(vis_img)
        infrared_img_1 = torch.unsqueeze(inf_img[:, 0, :, :], 1)
        fusion_img_Y = self.fusion_block(input_vis_img_y, infrared_img_1)
        fusion_img_Y = torch.unsqueeze(fusion_img_Y[:, 0, :, :], 1)
        output_img = color_recov(fusion_img_Y, input_vis_img_Cb, input_vis_img_Cr)

        return input_vis_img_y, infrared_img_1, fusion_img_Y, output_img

if __name__ == '__main__':
    upscale = 4
    window_size = 8
    height = 128
    width = 128
    model = SEFusion(upscale=2, img_size=128,window_size=window_size, img_range=1.,
                      embed_dim=64, mlp_ratio=2, upsampler='',seq_length=2)
    x = torch.randn((3, 3, height, width))
    y = torch.randn((3, 3, height, width))
    x,y,a,b = model(x,y)
    print(x.shape)
