import os

import torch
from einops import rearrange
from torch import nn as nn
from torch.nn import functional as F

import models
from models.help_funcs import TwoLayerConv2d, Transformer, TransformerDecoder
import numpy as np
import matplotlib.pyplot as plt
class ResNet(torch.nn.Module):

    def __init__(self, input_nc, output_nc, backbone='resnet18',
                 resnet_stages_num=4,
                 output_sigmoid=False):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ResNet, self).__init__()
        expand = 1
        if backbone == 'resnet18':
            self.resnet = models.resnet18(pretrained=True,
                                          replace_stride_with_dilation=[False,False,True], input_nc=input_nc)
        elif backbone == 'resnet34':
            self.resnet = models.resnet34(pretrained=True,
                                          replace_stride_with_dilation=[False,False,True], input_nc=input_nc)
        elif backbone == 'resnet50':
            self.resnet = models.resnet50(pretrained=True,
                                          replace_stride_with_dilation=[False,False,True], input_nc=input_nc)
            expand = 4
        else:
            raise NotImplementedError
        self.relu = nn.ReLU()
        self.upsamplex2 = nn.Upsample(scale_factor=2)
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')

        self.resnet_stages_num = resnet_stages_num


        self.output_sigmoid = output_sigmoid
        self.sigmoid = nn.Sigmoid()

        self.pos_stage1 = PA(64 * expand, output_nc)
        self.pos_stage2 = PA(128 * expand, output_nc)
        self.pos_stage3 = PA(256 * expand, output_nc)
        self.pos_stage4 = PA(512 * expand, output_nc)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x_4 = self.resnet.layer1(x) # 1/4, in=64, out=64*expand
        feat1 = x_4

        x_8 = self.resnet.layer2(x_4) # 1/8, in=64*expansion, out=128*expand
        feat2 = x_8

        if self.resnet_stages_num > 2:
            x_16 = self.resnet.layer3(x_8) # 1/16, dilation=1，in=128*expand, out=256*expand
            feat3 = x_16

        if self.resnet_stages_num == 4:
            x_16 = self.resnet.layer4(x_16) # 1/16, dilation=2，in=256*expand, out=512*expand
            feat4 = x_16

        elif self.resnet_stages_num > 4:
            raise NotImplementedError

        out_s4 = self.pos_stage1(feat1)
        out_s8 = self.pos_stage2(feat2)
        if self.resnet_stages_num == 3:
            out_s16 = self.pos_stage3(feat3)
        elif self.resnet_stages_num == 4:
            out_s16 = self.pos_stage4(feat4)
        elif self.resnet_stages_num > 4:
            raise NotImplementedError

        return out_s16, out_s8, out_s4

class PA(nn.Module):
    def __init__(self, inchan = 512, out_chan = 32):
        super().__init__()
        self.conv = nn.Conv2d(inchan, out_chan, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.re = nn.ReLU()
        self.do = nn.Dropout2d(0.2)
        self.pa_conv = nn.Conv2d(out_chan, out_chan, kernel_size=1, padding=0, groups=out_chan)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = self.conv(x)
        x = self.do(self.re(self.bn(x0)))
        return x0 *self.sigmoid(self.pa_conv(x))

class DownsampleConv(nn.Module):
    def __init__(self,in_channels=32):
        super(DownsampleConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False)
        self.batchNorm = nn.BatchNorm2d(32, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.batchNorm(x)
        x = self.relu(x)
        return x

class SpatialAttention(nn.Module):
    def __init__(self, in_channels=32, out_channels=8, kernel_size=3):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        return self.sigmoid(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels,in_channels//ratio,1,bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels//ratio, in_channels,1,bias=False)
        self.sigmod = nn.Sigmoid()
    def forward(self,x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmod(out)



class ConvDownsample(nn.Module):
    def __init__(self,in_channels=32):
        super(ConvDownsample, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.batchNorm = nn.BatchNorm2d(32, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.batchNorm(x)
        x = self.relu(x)
        return x

class Upsample(nn.Module):
    def __init__(self, in_ch, bilinear=False, scale_factor=2, mode='bilinear'):
        super(Upsample, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=scale_factor,
                                  mode=mode,
                                  align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)

    def forward(self, x):
        x = self.up(x)
        return x


class DAT(nn.Module):
    def __init__(self, in_channels, token_len):
        super(DAT, self).__init__()
        self.token_len = token_len
        self.in_channels = in_channels

        # 初始化空间注意力和通道注意力模块
        self.sam = SpatialAttention(in_channels=self.in_channels, out_channels=self.token_len)
        self.cam = ChannelAttention(in_channels=self.in_channels, ratio=2)

    def forward(self, x):
        b, c, h, w = x.shape
        # 1. 计算空间注意力
        spatial_attention = self.sam(x)
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        # 2. 特征图转换为序列，并与空间注意力相乘
        x_temp = x.view([b, c, -1]).contiguous()
        tokens = torch.einsum('bln,bcn->blc', spatial_attention, x_temp)
        # 3. 计算通道注意力并应用到 tokens
        channel_attention = self.cam(x)
        channel_attention = channel_attention.squeeze(-1).squeeze(-1).unsqueeze(1)
        tokens = tokens * channel_attention

        return tokens

class DAT_Transformer(nn.Module):
    def __init__(self, with_pos='learned',
                 in_channels=32, dam_tokenizer=True,
                 token_len=8, with_encoder=True, with_decoder=True,
                 enc_depth=1, dec_depth=1,
                 dim_head=64, decoder_dim_head=64,
                 pool_mode='max', pool_size=2,
                 decoder_softmax=True, with_decoder_pos=None,):

        super(DAT_Transformer, self).__init__()
        self.in_channels = in_channels
        self.token_len = token_len
        self.dam_tokenizer = dam_tokenizer

        if not self.dam_tokenizer:
            self.pooling_size = pool_size
            self.pool_mode = pool_mode
            self.token_len = self.pooling_size * self.pooling_size

        if self.dam_tokenizer:
            self.DAT = DAT(in_channels=self.in_channels, token_len=self.token_len)


        self.with_encoder = with_encoder
        self.with_decoder = with_decoder
        dim = self.in_channels
        mlp_dim = 2*dim
        self.with_pos = with_pos
        if with_pos == 'learned':
            self.pos_embedding = nn.Parameter(torch.randn(1, self.token_len*2, in_channels))
        decoder_pos_size = 256//4
        self.with_decoder_pos = with_decoder_pos
        if self.with_decoder_pos == 'learned':
            self.pos_embedding_decoder =nn.Parameter(torch.randn(1, self.in_channels,
                                                                 decoder_pos_size,
                                                                 decoder_pos_size))
        self.enc_depth = enc_depth
        self.dec_depth = dec_depth
        self.dim_head = dim_head
        self.decoder_dim_head = decoder_dim_head
        self.transformer = Transformer(dim=dim, depth=self.enc_depth, heads=8,
                                       dim_head=self.dim_head,
                                       mlp_dim=mlp_dim, dropout=0)
        self.transformer_decoder = TransformerDecoder(dim=dim, depth=self.dec_depth,
                            heads=8, dim_head=self.decoder_dim_head, mlp_dim=mlp_dim, dropout=0,
                                                      softmax=decoder_softmax)

    def _forward_dam_tokens(self, x):
        tokens = self.DAT(x)
        return tokens

    def _forward_reshape_tokens(self, x):
        # b,c,h,w = x.shape
        if self.pool_mode == 'max':
            x = F.adaptive_max_pool2d(x, [self.pooling_size, self.pooling_size])
        elif self.pool_mode == 'ave':
            x = F.adaptive_avg_pool2d(x, [self.pooling_size, self.pooling_size])
        else:
            x = x
        tokens = rearrange(x, 'b c h w -> b (h w) c')
        return tokens

    def _forward_transformer_encoder(self, x):
        if self.with_pos:
            x += self.pos_embedding
        x = self.transformer(x)
        return x


    def _forward_transformer_decoder(self, x, m):
        b, c, h, w = x.shape
        if self.with_decoder_pos == 'fix':
            x = x + self.pos_embedding_decoder
        elif self.with_decoder_pos == 'learned':
            x = x + self.pos_embedding_decoder
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.transformer_decoder(x, m)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x

    def _forward_simple_decoder(self, x, m):
        b, c, h, w = x.shape
        b, l, c = m.shape
        m = m.expand([h,w,b,l,c])
        m = rearrange(m, 'h w b l c -> l b c h w')
        m = m.sum(0)
        x = x + m
        return x

    def forward(self, x1, x2):
        #  forward tokenzier
        if self.dam_tokenizer:
            token1 = self._forward_dam_tokens(x1)
            token2 = self._forward_dam_tokens(x2)
        else:
            token1 = self._forward_reshape_tokens(x1)
            token2 = self._forward_reshape_tokens(x2)
        if self.with_encoder:
            self.tokens_ = torch.cat([token1, token2], dim=1)
            self.tokens = self._forward_transformer_encoder(self.tokens_)
            token1, token2 = self.tokens.chunk(2, dim=1)
        # forward transformer decoder
        if self.with_decoder:
            x1 = self._forward_transformer_decoder(x1, token1)
            x2 = self._forward_transformer_decoder(x2, token2)
        else:
            x1 = self._forward_simple_decoder(x1, token1)
            x2 = self._forward_simple_decoder(x2, token2)

        return x1, x2


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
    def forward(self, x):
        return self.conv(x)

class ResidualECAM(nn.Module):
    def __init__(self, embedding_dim=32, ratio=16):
        super(ResidualECAM, self).__init__()
        #组间关系
        self.cam_inter = ChannelAttention(embedding_dim * 3, ratio)
        #组内关系
        self.cam_intra = ChannelAttention(embedding_dim, ratio // 4)

    def forward(self, x1, x2, x3):
        concat = torch.cat([x1, x2, x3], 1)
        #组间关系
        inter = self.cam_inter(concat)
        inter = inter * concat
        #组内关系
        sum = torch.sum(torch.stack((x1, x2, x3)), dim=0)
        intra = self.cam_intra(sum)
        intra = intra.repeat(1, 3, 1, 1)
        intra = intra * concat

        return concat + inter + intra

class SinglescaleFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SinglescaleFusion, self).__init__()
        self.conv_s4 = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x1_s4, x2_s4):
        x_s4 = torch.cat((x1_s4, x2_s4), dim=1)  # s4尺度特征串联
        fused_features = self.conv_s4(x_s4)  # 保持s4尺寸不变
        return fused_features


class MFAFM(nn.Module):
    def __init__(self, embedding_dim):
        super(MFAFM, self).__init__()
        # 定义每个尺度的卷积操作
        self.cat_s16 = ConvBlock(in_channels=2 * embedding_dim, out_channels=embedding_dim)
        self.Ups16_bilinear = Upsample(in_ch=embedding_dim, bilinear=True)  # 两倍上采样

        self.cat_s8 = ConvBlock(in_channels=3 * embedding_dim, out_channels=embedding_dim)
        self.Ups8_bilinear = Upsample(in_ch=embedding_dim, bilinear=True)  # 两倍上采样

        self.cat_s4 = ConvBlock(in_channels=3 * embedding_dim, out_channels=embedding_dim)

        # 集成通道注意力模块
        self.resecam = ResidualECAM(embedding_dim=embedding_dim, ratio=16)


    def forward(self, x1_s16, x2_s16, x1_s8, x2_s8, x1_s4, x2_s4):
        # s16 尺度特征串联和卷积
        x_s16 = self.cat_s16(torch.cat((x1_s16, x2_s16), dim=1))
        x_s16_up = F.interpolate(x_s16, size=x1_s4.shape[2:], mode='bicubic', align_corners=True)  # 上采样到 s4 尺度

        # s8 尺度特征串联和卷积，包含 s16 的两倍上采样特征
        x_s8 = self.cat_s8(torch.cat((x1_s8, x2_s8, self.Ups16_bilinear(x_s16)), dim=1))
        x_s8_up = F.interpolate(x_s8, size=x1_s4.shape[2:], mode='bicubic', align_corners=True)  # 上采样到 s4 尺度

        # s4 尺度特征串联和卷积，包含 s8 的两倍上采样特征
        x_s4 = self.cat_s4(torch.cat((x1_s4, x2_s4, self.Ups8_bilinear(x_s8)), dim=1))

        # 使用ECAM模块融合多尺度特征
        out = self.resecam(x_s16_up, x_s8_up, x_s4)

        return out




class ConcatenationFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConcatenationFusion, self).__init__()

        # 为每个尺度定义卷积和批量归一化层
        self.conv_s16 = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.conv_s8 = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.conv_s4 = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        # 上采样层，将s16和s8的输出调整为s4的尺寸
        self.upsample_s16 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)  # 将s16上采样到s4
        self.upsample_s8 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 将s8上采样到s4

    def forward(self, x1_s16, x2_s16, x1_s8, x2_s8, x1_s4, x2_s4):
        # 在每个尺度上进行特征串联和卷积
        x_s16 = torch.cat((x1_s16, x2_s16), dim=1)  # s16尺度特征串联
        x_s16 = self.conv_s16(x_s16)
        x_s16 = self.upsample_s16(x_s16)  # 上采样到s4尺寸

        x_s8 = torch.cat((x1_s8, x2_s8), dim=1)  # s8尺度特征串联
        x_s8 = self.conv_s8(x_s8)
        x_s8 = self.upsample_s8(x_s8)  # 上采样到s4尺寸

        x_s4 = torch.cat((x1_s4, x2_s4), dim=1)  # s4尺度特征串联
        x_s4 = self.conv_s4(x_s4)  # 保持s4尺寸不变
        # 将所有尺度的结果在通道维度上串联
        fused_features = torch.cat((x_s16, x_s8, x_s4), dim=1)

        return fused_features

class SummationFusion(nn.Module):
    def __init__(self):
        super(SummationFusion, self).__init__()
        # 上采样层，将 s16 和 s8 的输出调整为 s4 的尺寸
        self.upsample_s16 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)  # 将 s16 上采样到 s4
        self.upsample_s8 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 将 s8 上采样到 s4

    def forward(self, x1_s16, x2_s16, x1_s8, x2_s8, x1_s4, x2_s4):
        # 在每个尺度上直接相加
        x_s16 = x1_s16 + x2_s16  # s16 尺度特征相加
        x_s16 = self.upsample_s16(x_s16)  # 上采样到 s4 尺寸

        x_s8 = x1_s8 + x2_s8  # s8 尺度特征相加
        x_s8 = self.upsample_s8(x_s8)  # 上采样到 s4 尺寸

        x_s4 = x1_s4 + x2_s4  # s4 尺度特征相加，保持 s4 尺寸不变
        # 将所有尺度的结果在通道维度上串联
        fused_features = torch.cat((x_s16, x_s8, x_s4), dim=1)

        return fused_features

class SubtractionFusion(nn.Module):
    def __init__(self):
        super(SubtractionFusion, self).__init__()
        # 上采样层，将 s16 和 s8 的输出调整为 s4 的尺寸
        self.upsample_s16 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)  # 将 s16 上采样到 s4
        self.upsample_s8 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 将 s8 上采样到 s4

    def forward(self, x1_s16, x2_s16, x1_s8, x2_s8, x1_s4, x2_s4):
        # 在每个尺度上进行特征相减（差分）并取绝对值
        x_s16 = torch.abs(x1_s16 - x2_s16)  # s16 尺度特征差分取绝对值
        x_s16 = self.upsample_s16(x_s16)  # 上采样到 s4 尺度

        x_s8 = torch.abs(x1_s8 - x2_s8)  # s8 尺度特征差分取绝对值
        x_s8 = self.upsample_s8(x_s8)  # 上采样到 s4 尺度

        x_s4 = torch.abs(x1_s4 - x2_s4)  # s4 尺度特征差分取绝对值，保持 s4 尺度不变
        # 将所有尺度的结果在通道维度上串联
        fused_features = torch.cat((x_s16, x_s8, x_s4), dim=1)

        return fused_features

class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out) * 0.1
        out = torch.add(out, residual)
        return out

#最终计算变化图
class ClassifyLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ClassifyLayer, self).__init__()
        padding = kernel_size // 2
        # 定义第一个卷积层，没有偏置项（bias=False）
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                               padding=padding, stride=1, bias=False)
        # 添加批量归一化层
        self.bn = nn.BatchNorm2d(in_channels)
        # 添加ReLU激活函数
        self.relu = nn.ReLU(inplace=True)
        # 定义第二个卷积层，输出通道数为 out_channels
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                               padding=padding, stride=1)

    def forward(self, x):
        # 通过第一个卷积层
        x = self.conv1(x)
        # 应用批量归一化
        x = self.bn(x)
        # 应用ReLU激活函数
        x = self.relu(x)
        # 通过第二个卷积层
        x = self.conv2(x)
        return x



class MutiscaleTransformer(nn.Module):

    def __init__(self, backbone='resnet18', resnet_stages_num=3, input_nc=3, embedding_dim=32, pre_fusion=True, token_len=8, use_dat=True, use_transfromer=True, enc_depth=2, dec_depth=8, edm=64, ddm=8, post_fusion=1, n_class=2, configname='MT_res18_stages4_prefus_damtoken_trans_fuse1_1119'):
        super(MutiscaleTransformer, self).__init__()
        self.pre_fusion = pre_fusion
        self.use_dat = use_dat
        self.use_trans = use_transfromer
        self.post_fusion = post_fusion
        self.Res = ResNet(input_nc=input_nc, output_nc=embedding_dim, backbone=backbone,  resnet_stages_num=resnet_stages_num)

        self.configname = configname

        if self.pre_fusion:
            self.conv_stride_1_1 = ConvDownsample(in_channels=embedding_dim)
            self.conv_stride_1_2 = ConvDownsample(in_channels=embedding_dim)
            self.conv_stride_1_3 = ConvDownsample(in_channels=embedding_dim)
            self.conv_stride_2_1 = ConvDownsample(in_channels=embedding_dim)
            self.conv_stride_2_2 = ConvDownsample(in_channels=embedding_dim)
            self.conv_stride_2_3 = ConvDownsample(in_channels=embedding_dim)

            self.upsample_1_1 = Upsample(in_ch=embedding_dim, bilinear=True)
            self.upsample_1_2 = Upsample(in_ch=embedding_dim, bilinear=True)
            self.upsample_1_3 = Upsample(in_ch=embedding_dim, bilinear=True, scale_factor=4)
            self.upsample_2_1 = Upsample(in_ch=embedding_dim, bilinear=True)
            self.upsample_2_2 = Upsample(in_ch=embedding_dim, bilinear=True)
            self.upsample_2_3 = Upsample(in_ch=embedding_dim, bilinear=True, scale_factor=4)



        if self.post_fusion != 0:
            self.Trans_s16 = DAT_Transformer(in_channels=embedding_dim, dam_tokenizer=self.use_dat, token_len=token_len, with_encoder=self.use_trans, with_decoder=self.use_trans, enc_depth=enc_depth, dec_depth=dec_depth, dim_head=edm, decoder_dim_head=ddm)
            self.Trans_s8 = DAT_Transformer(in_channels=embedding_dim, dam_tokenizer=self.use_dat, token_len=token_len, with_encoder=self.use_trans, with_decoder=self.use_trans, enc_depth=enc_depth, dec_depth=dec_depth, dim_head=edm, decoder_dim_head=ddm)
        self.Trans_s4 = DAT_Transformer(in_channels=embedding_dim, dam_tokenizer=self.use_dat, token_len=token_len, with_encoder=self.use_trans, with_decoder=self.use_trans, enc_depth=enc_depth, dec_depth=dec_depth, dim_head=edm, decoder_dim_head=ddm)


        #如果选择使用单尺度融合
        if self.post_fusion == 0:
            self.singlescale_fusion = SinglescaleFusion(in_channels=embedding_dim, out_channels=embedding_dim)
        #如果选择使用MFAFM
        if self.post_fusion == 1:
            self.MFFM = MFAFM(embedding_dim=embedding_dim)
        #如果使用串联融合
        if self.post_fusion == 2:
            self.concatenation_fusion = ConcatenationFusion(in_channels=embedding_dim, out_channels=embedding_dim)
        # 如果使用求和融合
        if self.post_fusion == 3:
            self.summation_fusion = SummationFusion()
        # 如果使用差分融合
        if self.post_fusion == 4:
            self.subtraction_fusion = SubtractionFusion()


        #Final linear fusion layer
        self.final_fuse = nn.Sequential(
            nn.Conv2d(in_channels=embedding_dim * 3, out_channels=embedding_dim,
                                        kernel_size=1),
            nn.BatchNorm2d(embedding_dim)
        )


        #Final predction head
        self.convd2x    = Upsample(in_ch=embedding_dim, bilinear=False)
        self.resid_2x   = nn.Sequential(ResidualBlock(embedding_dim))
        self.convd1x    = Upsample(in_ch=embedding_dim, bilinear=False)
        self.resid_1x   = nn.Sequential(ResidualBlock(embedding_dim))
        self.classifier = ClassifyLayer(embedding_dim, n_class, kernel_size=3)

    def generate_heatmap(self, feature_map, name, dir ):
        feature_map = feature_map.mean(dim=1, keepdim=True)
        heatmap = feature_map.squeeze().cpu().detach().numpy()
        # 最大最小归一化到 [0, 1] 之间
        heatmap_min = np.min(heatmap)
        heatmap_max = np.max(heatmap)
        heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)  # 归一化到 [0, 1] 之间
        # 使用相对路径保存热图
        save_dir = os.path.join("heatmaps", "heatmaps_LEVIR", self.configname, dir)
        os.makedirs(save_dir, exist_ok=True)
        # 生成并保存热力图
        for i in range(heatmap.shape[0]):
            image_name = name[i] if isinstance(name, list) else name
            save_path = os.path.join(save_dir, f"{image_name}")
            fig = plt.figure()
            plt.imshow(heatmap[i], cmap='jet_r')
            plt.axis('off')
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig)



    def forward(self, img1, img2, name):

        out1_s16, out1_s8, out1_s4 = self.Res(img1)
        out2_s16, out2_s8, out2_s4 = self.Res(img2)

        if self.pre_fusion:
            out1_s4_down_1 = self.conv_stride_1_1(out1_s4)
            out1_s4_down_2 = self.conv_stride_1_2(out1_s4_down_1)
            out1_s8_down = self.conv_stride_1_3(out1_s8)
            out1_s8_up = self.upsample_1_1(out1_s8)
            out1_s16_up_1 = self.upsample_1_2(out1_s16)
            out1_s16_up_2 = self.upsample_1_3(out1_s16)
            #时期一各尺度相加融合
            out1_s16 = out1_s16 + out1_s4_down_2 + out1_s8_down
            out1_s8 = out1_s8 + out1_s4_down_1 + out1_s16_up_1
            out1_s4 = out1_s4 + out1_s16_up_2 + out1_s8_up

            #对时期二的各尺寸特征图做上采样和下采样
            out2_s4_down_1 = self.conv_stride_2_1(out2_s4)
            out2_s4_down_2 = self.conv_stride_2_2(out2_s4_down_1)
            out2_s8_down = self.conv_stride_2_3(out2_s8)
            out2_s8_up = self.upsample_2_1(out2_s8)
            out2_s16_up_1 = self.upsample_2_2(out2_s16)
            out2_s16_up_2 = self.upsample_2_3(out2_s16)
            #时期二各尺度相加融合
            out2_s16 = out2_s16 + out2_s4_down_2 + out2_s8_down
            out2_s8 = out2_s8 + out2_s4_down_1 + out2_s16_up_1
            out2_s4 = out2_s4 + out2_s16_up_2 + out2_s8_up

        if self.post_fusion != 0:
            out1_s16, out2_s16 = self.Trans_s16(out1_s16, out2_s16) #1/16 32
            out1_s8, out2_s8 = self.Trans_s8(out1_s8, out2_s8) #1/8 32
        out1_s4, out2_s4 = self.Trans_s4(out1_s4, out2_s4) #1/4 32


        if self.post_fusion == 0:
            out = self.singlescale_fusion(out1_s4, out2_s4)
        if self.post_fusion == 1:
            out = self.MFFM(out1_s16, out2_s16, out1_s8, out2_s8, out1_s4, out2_s4)
            out = self.final_fuse(out)
        elif self.post_fusion == 2:
            out = self.concatenation_fusion(out1_s16, out2_s16, out1_s8, out2_s8, out1_s4, out2_s4)
            out = self.final_fuse(out)
        elif self.post_fusion == 3:
            out = self.summation_fusion(out1_s16, out2_s16, out1_s8, out2_s8, out1_s4, out2_s4)
            out = self.final_fuse(out)
        elif self.post_fusion == 4:
            out = self.subtraction_fusion(out1_s16, out2_s16, out1_s8, out2_s8, out1_s4, out2_s4)
            out = self.final_fuse(out)

        #添加热力图
        if not self.training:
            self.generate_heatmap(out, name, "out")

        out = self.convd2x(out)
        out = self.resid_2x(out)
        out = self.convd1x(out)
        out = self.resid_1x(out)

        #最终分类
        out = self.classifier(out)
        output = []
        output.append(out)
        return output
