# Author: Ling Qi
# Date: 2024/4/5

import torch
import torch.nn as nn
# 训练版如下
from models.network_transfusion_channel import transfusion_channel
from models.network_swinfusion_spatial_Y import swinfusion_spatial
from models.network_swinfusion_decoder_Y import swinfusion_decoder
# # 测试版如下
# from network_transfusion_channel import transfusion_channel
# from network_swinfusion_spatial_Y import swinfusion_spatial
# from network_swinfusion_decoder_Y import swinfusion_decoder

def rgb2ycbcr(img_rgb):
    R = torch.unsqueeze(img_rgb[:, 0, :, :], 1)
    G = torch.unsqueeze(img_rgb[:, 1, :, :], 1)
    B = torch.unsqueeze(img_rgb[:, 2, :, :], 1)
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128/255.0
    Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128/255.0
    # img_ycbcr = torch.cat([Y, Cb, Cr], dim=1)
    return Y, Cb, Cr


def color_recov(Yf, Cb, Cr):
    R = Yf + 1.402*(Cr - 128/255)
    G = Yf - 0.34414*(Cb - 128/255) - 0.71414*(Cr - 128/255)
    B = Yf + 1.772*(Cb - 128/255)
    output = torch.cat([R, G, B], dim=1)
    return output


class Illumination_Estimator(nn.Module):
    def __init__(
            self, n_fea_middle, n_fea_in=4, n_fea_out=3):
        super(Illumination_Estimator, self).__init__()

        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)

        self.depth_conv = nn.Conv2d(
            n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_in)#depth-wise conv ,可降低参数数量，但不能减少运算时间

        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

    def forward(self, img):                                                                                                                                                                                                                                 

        #对输入的原始图像的第一维度也即在同一位置的3个channle上求平均值，并扩维回到四维上便于拼接
        mean_c = img.mean(dim=1).unsqueeze(1)

        #将原始input和求均值后的图像在每个channle进行纵向拼接
        input = torch.cat([img,mean_c], dim=1)

        #分别进行卷积输出特征图和特征
        x_1 = self.conv1(input)
        illu_fea = self.depth_conv(x_1)
        illu_map = self.conv2(illu_fea)

        return illu_fea, illu_map
    

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
        # self.TFC = transfusion_channel(img_size=self.img_size, in_chans_img_a=self.in_chan, in_chans_img_b=self.in_chan, out_chans=3,
        #                                Fusion_depths=2, Fusion_num_heads=8, mlp_ratio=4, drop_rate=0., 
        #                                attn_drop_rate=0., drop_path_rate=0.1, downsample_ratio=2, seq_length=2, 
        #                                poem_type='sinusoidal', layer_norm=True, last_layer_norm=True)
        
        self.SFS = swinfusion_spatial(img_size=self.img_size, patch_size=1, in_chans_img_a=self.in_chan, in_chans_img_b=self.in_chan, 
                                      out_chans=self.in_chan, embed_dim=self.embed_dim, Ex_depths=[4], Fusion_depths=[2, 2], Re_depths=[4], 
                                      Ex_num_heads=[6], Fusion_num_heads=[6, 6], Re_num_heads=[6], window_size=self.window_size, 
                                      mlp_ratio=self.mlp_ratio, qkv_bias=True, qk_scale=None, drop_rate=0., 
                                      attn_drop_rate=0., drop_path_rate=0.2, norm_layer=nn.LayerNorm, ape=False, # 4.21中午 drop_path_rate=0.->0.2
                                      patch_norm=True, use_checkpoint=False, upscale=self.upscale, img_range=self.img_range, 
                                      upsampler='', resi_connection='1conv', seq_length=self.seq_length)
        
        self.conv_after_body_Fusion = nn.Conv2d(2 * self.in_chan, self.in_chan, 3, 1, 1)
        self.conv_after_body_Fusion_channel = nn.Conv2d(2 * self.in_chan, self.in_chan, 3, 1, 1)
        self.conv_after_body_Fusion_final = nn.Conv2d(2 * self.in_chan, self.in_chan, 3, 1, 1)
        # 可以尝试别的激活函数
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, lit_img, inf_img):
        # 串联式，不知道哪个效果比较好
        # lit_img, inf_img = self.TFC(lit_img, inf_img)
        # lit_img, inf_img = self.TFC(lit_img, inf_img)
        # print(lit_img.shape, inf_img.shape, "11111")
        lit_img, inf_img = self.SFS(lit_img, inf_img)
        # lit_img, inf_img = self.SFS(lit_img, illu_fea, inf_img)
        # print(lit_img.shape, inf_img.shape, "22222")
        fusion_img = torch.cat([lit_img, inf_img], 1)
        # print(fusion_img.shape, "33333")
        fusion_img = self.lrelu(self.conv_after_body_Fusion(fusion_img))
        # print(fusion_img.shape, "33333")


        # # 并联式，并联的方法可能得特殊设计一下
        # # lit_img_chan, inf_img_chan = transfusion_channle(lit_img, illu_fea, inf_img)
        # # lit_img, inf_img = swinfusion_spatial(lit_img, illu_fea, inf_img)
        # lit_img_channel, inf_img_channel = self.TFC(lit_img, inf_img)
        # print(lit_img_channel.shape, inf_img_channel.shape, "11111")
        # lit_img, inf_img = self.SFS(lit_img, inf_img)
        # print(lit_img.shape, inf_img.shape, "22222")

        # fusion_img_channel = torch.cat([lit_img_channel, inf_img_channel], 1)
        # print(fusion_img_channel.shape, "33333")
        # fusion_img = torch.cat([lit_img, inf_img], 1)
        # print(fusion_img.shape, "44444")

        # fusion_img_channel = self.lrelu(self.conv_after_body_Fusion_channel(fusion_img_channel))
        # print(fusion_img_channel.shape, "55555")
        # fusion_img = self.lrelu(self.conv_after_body_Fusion(fusion_img))
        # print(fusion_img.shape, "66666")
        # fusion_img = torch.cat([fusion_img_channel, fusion_img], 1)
        # print(fusion_img.shape, "77777")

        # fusion_img = self.lrelu(self.conv_after_body_Fusion_final(fusion_img))
        # print(fusion_img.shape, "88888")

        return fusion_img

class fusion_decoder(nn.Module):
    def __init__(self, in_chan=4, img_size=32, window_size=2, embed_dim=60, 
                 mlp_ratio=2, upscale=1, img_range=1):
        super(fusion_decoder, self).__init__()

        self.in_chan = in_chan
        self.img_size = img_size
        self.window_size = window_size
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.upscale = upscale
        self.img_range = img_range
        self.SFD = swinfusion_decoder(img_size=self.img_size, patch_size=1, in_chans=self.in_chan, embed_dim=self.embed_dim, 
                                      Ex_depths=[4], Fusion_depths=[2, 2], Re_depths=[4], Ex_num_heads=[6], 
                                      Fusion_num_heads=[6, 6], Re_num_heads=[6], window_size=self.window_size, mlp_ratio=self.mlp_ratio, 
                                      qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., 
                                      drop_path_rate=0.4, norm_layer=nn.LayerNorm, ape=False, patch_norm=True,# 4.21中午 drop_path_rate=0.->0.2
                                      use_checkpoint=False, upscale=self.upscale, img_range=self.img_range, upsampler='', resi_connection='1conv')

    def forward(self, fusion_img):
        fusion_img = self.SFD(fusion_img)

        return fusion_img
    
# # 结合照度特征解码
# class fusion_decoder(nn.Module):
#     def __init__(self, in_chan=12, img_size=32, window_size=2, embed_dim=60, 
#                  mlp_ratio=2, upscale=1, img_range=1):
#         super(fusion_decoder, self).__init__()

#         self.in_chan = in_chan
#         self.img_size = img_size
#         self.window_size = window_size
#         self.embed_dim = embed_dim
#         self.mlp_ratio = mlp_ratio
#         self.upscale = upscale
#         self.img_range = img_range
#         self.SFD = swinfusion_decoder(img_size=self.img_size, patch_size=1, in_chans=self.in_chan, embed_dim=self.embed_dim, 
#                                       Ex_depths=[4], Fusion_depths=[2, 2], Re_depths=[4], Ex_num_heads=[6], 
#                                       Fusion_num_heads=[6, 6], Re_num_heads=[6], window_size=self.window_size, mlp_ratio=self.mlp_ratio, 
#                                       qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., 
#                                       drop_path_rate=0.3, norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
#                                       use_checkpoint=False, upscale=self.upscale, img_range=self.img_range, upsampler='', resi_connection='1conv')

#     def forward(self, fusion_img, illu_fea):
#         fusion_img = self.SFD(fusion_img, illu_fea)

#         return fusion_img
#结合照度特征进行解码，但解码的方式需要进一步测试
# class fusion_decoder(nn.Module):
#     def __init__(self) -> None:
#         super(fusion_decoder, self).__init__()
    
#     def forward(fusion_img, ill_fea):
#         fusion_img = swinFusion_decoder(fusion_img, ill_fea)

#         return fusion_img
    
class Incoder(nn.Module):
    def __init__(self):
        super(Incoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
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

        self.lrelu = nn.LeakyReLU()

    def forward(self, fusion_img):
        conv1 = self.lrelu(self.bn1(self.conv1(fusion_img)))
        conv2 = self.lrelu(self.bn2(self.conv2(fusion_img)))
        conv3 = self.lrelu(self.bn3(self.conv3(fusion_img)))
        conv4 = self.lrelu(self.bn4(self.conv4(fusion_img)))
        feature_multiscale = torch.cat([conv1, conv2, conv3, conv4], dim=1)
        # feature_shuffle = shuffle_unit(x=feature_multiscale, groups=4) 
        feature_shuffle = feature_multiscale

        # 计算对比度特征
        mean_vector = torch.mean(feature_shuffle, dim=[2, 3], keepdim=True)
        feature_contrast = torch.sqrt(torch.mean((feature_shuffle - mean_vector) ** 2, dim=[2, 3], keepdim=True))
        contrast_vector = torch.mean(feature_contrast, dim=[2, 3], keepdim=True)
        # 通过对比度向量增强特征
        feature_fusion_enhancement = contrast_vector * feature_shuffle

        return feature_fusion_enhancement

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        x = self.lrelu(self.bn1(self.conv1(x)))
        x = self.lrelu(self.bn2(self.conv2(x)))
        x = torch.sigmoid(self.bn3(self.conv3(x)))
        return x
    

class Fusion_block(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, img_size=128, fusionblock_num=3, 
                 window_size=8, embed_dim=60, img_range=1, mlp_ratio=2, 
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
        # RGB图像预处理标准化突出特征
        # if in_channels == 3 or in_channels == 6:
        #     rgb_mean = (0.4488, 0.4371, 0.4040)
        #     rgbrgb_mean = (0.4488, 0.4371, 0.4040, 0.4488, 0.4371, 0.4040)
        #     self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        #     self.mean_in = torch.Tensor(rgbrgb_mean).view(1, 6, 1, 1)
        # else:
        #     self.mean = torch.zeros(1, 1, 1, 1)

        #Encoder
        self.fusion_layers = nn.ModuleList([])
        for i in range(fusionblock_num):
            self.fusion_layers.append(nn.ModuleList([
                Fusion_con(in_chan=self.in_channels, img_size=int(self.img_size / (2**(i + 1))), window_size=int(self.window_size / (2**i)),
                           embed_dim=self.embed_dim, mlp_ratio=self.mlp_ratio, upscale=self.upscale, img_range=1, seq_length=self.seq_length),
                # 特征维度进行升维，同时对半下降维度尺寸
                nn.Conv2d(self.in_channels, self.in_channels * 2, 4, 2, 1, bias=False),# O = (I-K+2P)/S+1
                # nn.Conv2d(self.embed_dim, self.embed_dim, 4, 2, 1, bias=False),
                nn.Conv2d(self.in_channels, self.in_channels * 2, 4, 2, 1, bias=False)
            ]))
            self.in_channels *= 2
            window_size = int(self.window_size / (2**i))
            # print(window_size)

        
        # self.fusion_last_block = Fusion_con(in_chan=self.in_channels)

        # Decoder
        self.fusion_decoder_layers = nn.ModuleList([])
        self.in_channels //= 2
        for i in range(fusionblock_num):
            self.fusion_decoder_layers.append(nn.ModuleList([
                fusion_decoder(in_chan=self.in_channels, img_size=int(self.img_size * (2**i)), window_size=int(window_size * (2**i))),
                nn.ConvTranspose2d(self.in_channels, self.in_channels // 2, stride=2, # 反卷积上采样恢复图片尺寸 out=(in-1)*stride+outputpadding-2*padding+kernelsize
                                   kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(self.in_channels, self.in_channels // 2, 1, 1, bias=False), # 跨层连接融合              
            ]))
            self.in_channels //= 2


    def forward(self, lit_img, inf_img):

        # # 数据特征标准化,不知道照度图是否要这样做
        # self.mean_A = self.mean.type_as(lit_img)
        # self.mean_B = self.mean.type_as(inf_img)
        # self.mean = (self.mean_A + self.mean_B) / 2

        # lit_img = (lit_img - self.mean_A) * self.img_range
        # inf_img = (inf_img - self.mean_B) * self.img_range                
    
        
        # Encoder同时进行融合交互
        # 初始化两个列表，用于储存编码器的输出特征和照明特征
        # illu_fea_list = []
        fusion_img_list = []
        # x = torch.randn((1, 3, 128, 128))
        # fusion_img_list.append(x)
        # x = torch.randn((1, 6, 64, 64))
        # fusion_img_list.append(x)
        # x = torch.randn((1, 12, 32, 32))
        # fusion_img_list.append(x)
        # fusion_img = torch.randn((1, 12, 32, 32))
        for (Fusion_con, LitImgDownSample, InfImgDownSample) in self.fusion_layers:
            # 融合图像
            fusion_img = Fusion_con(lit_img, inf_img)
            # print(fusion_img.shape, illu_fea.shape, lit_img.shape, inf_img.shape, "44444")
            # illu_fea_list.append(illu_fea)
            # lit_img_list.append(lit_img)
            # inf_img_list.append(inf_img)
            fusion_img_list.append(fusion_img)
            # 第一个卷积下采样层为特征下采样
            lit_img = LitImgDownSample(lit_img)
            # print(lit_img.shape, "55555")
            # 第二个卷积下采样层为照明特征下采样
            # illu_fea = IlluFeaDownsample(illu_fea)
            # print(illu_fea.shape, "66666")
            inf_img = InfImgDownSample(inf_img)
            # print(fusion_img)
            # print(fusion_img.shape)
            # print(inf_img.shape, "77777")

        # Decoder
        for i, (FusionDecoder, ImgUpSample, ImgConcat) in enumerate(self.fusion_decoder_layers):
            # swin解码
            # 照度特征指导解码
            # fusion_img = FusionDecoder(fusion_img, illu_fea_list[self.fusionblock_num - 1 - i])
            # print(fusion_img.shape, "11111")
            # 非照度特征指导解码
            fusion_img = FusionDecoder(fusion_img)
            # print(fusion_img)
            # print(fusion_img.shape)
            # print(fusion_img.shape, fusion_img_list[self.fusionblock_num - 2 - i].shape, "33333")
            if i == self.fusionblock_num - 1:
                break
            fusion_img = ImgUpSample(fusion_img)
            # print(fusion_img.shape, "22222")
            # 融合两个相邻的特征维度
            fusion_img = torch.cat([fusion_img, fusion_img_list[self.fusionblock_num - 2  - i]], dim=1)
            # print(fusion_img.shape, "44444") 
            # 上采样返回维度

            # 返回上一层的维度
            fusion_img = ImgConcat(fusion_img)

            # print(fusion_img.shape, "55555")

        # fusion_img = fusion_img / self.img_range + self.mean
        

        return fusion_img


class retifumer(nn.Module):
    def __init__(self, img_size=128, window_size=8, img_range=1., in_channels=2, 
                 out_channels=2, n_feat=60, embed_dim=60, mlp_ratio=2, 
                 upsampler='', seq_length=1, fusionblock_num=4, upscale=2):# 4.19上午 3->4 12GB显存顶不住
        super(retifumer, self).__init__()
        
        self.window_size = window_size

        # Retinex分解块
        # self.illumination_estimator= Illumination_Estimator(n_feat)
        # 融合块
        self.fusion_block = Fusion_block(in_channels=in_channels, out_channels=out_channels, img_size=img_size, 
                                         embed_dim=embed_dim, fusionblock_num=fusionblock_num, window_size=self.window_size,
                                         img_range=img_range, mlp_ratio=mlp_ratio, upsampler=upsampler, seq_length=seq_length,)
        self.enhancement = Enhancement()
        self.decoder = Decoder()
        self.incoder = Incoder()

    def forward(self, img_x, img_y):
        input_vis_img_y, input_vis_img_Cb, input_vis_img_Cr = rgb2ycbcr(img_x)
        input_vis_img_2 = torch.cat([input_vis_img_y, input_vis_img_y], dim=1)
        # illu_fea, illu_map = self.illumination_estimator(img_x)
        # input_img = img_x * illu_map + img_x # 将原始图像和增亮特征进行相加，类似与残差连接，得到强制增量后的图片\
        # infrared_img_x = torch.unsqueeze(img_x[:, 0, :, :], 1)
        infrared_img_1 = torch.unsqueeze(img_y[:, 0, :, :], 1)
        infrared_img_2 = torch.cat([infrared_img_1, infrared_img_1], dim=1)
        # print(infrared_img.shape)   
        # print(input_vis_img_y.shape,infrared_img.shape)
        fusion_img_Y = self.fusion_block(input_vis_img_2, infrared_img_2) # Y通道和红外双通道融合
        fusion_img_Y = torch.unsqueeze(fusion_img_Y[:, 0, :, :], 1)
        fusion_img_Y = self.incoder(fusion_img_Y) 
        fusion_img_Y = self.enhancement(fusion_img_Y)
        fusion_img_Y = self.decoder(fusion_img_Y)
        # print(output_img.shape)
        output_img = color_recov(fusion_img_Y, input_vis_img_Cb, input_vis_img_Cr)

        return input_vis_img_y, infrared_img_1, fusion_img_Y, output_img

if __name__ == '__main__':
    upscale = 4
    window_size = 8
    # height = (1024 // upscale // window_size + 1) * window_size
    height = 128
    # width = (720 // upscale // window_size + 1) * window_size
    width = 128
    model = retifumer(upscale=2, img_size=128,window_size=window_size, img_range=1.,
                      embed_dim=60, mlp_ratio=2, upsampler='',seq_length=2)
    # print(model)
    # print(height, width, model.flops() / 1e9)

    # 测试矩阵
    x = torch.randn((3, 3, height, width))
    y = torch.randn((3, 3, height, width))
    x,y,a,b = model(x,y)
    # print(x)
    print(b.shape)
    # print(x.shape)
