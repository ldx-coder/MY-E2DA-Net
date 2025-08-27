import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import math
import numpy as np
from functools import partial
from typing import Optional, Callable, Union
from einops import rearrange
from ..modules.conv import Conv, DWConv, RepConv, GhostConv, autopad
from ..modules.block import *


__all__ = ['SDFM']


class SDFM(nn.Module):
    '''
    superficial detail fusion module
    '''

    def __init__(self, channels=64, r=4):
        super(SDFM, self).__init__()
        inter_channels = int(channels // r)

        self.Recalibrate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            # 移除 Conv 层中的 BatchNorm，或者使用其他归一化
            nn.Sequential(
                nn.Conv2d(2 * channels, 2 * inter_channels, 1),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(2 * inter_channels, 2 * channels, 1),
                nn.Sigmoid()
            ),
        )

        self.channel_agg = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 1),
            nn.ReLU(inplace=True)
        )

        self.local_att = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(channels, inter_channels, 1),
                nn.ReLU(inplace=True)
            ),
            nn.Conv2d(inter_channels, channels, 1),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Sequential(
                nn.Conv2d(channels, inter_channels, 1),
                nn.ReLU(inplace=True)
            ),
            nn.Conv2d(inter_channels, channels, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        x1, x2 = data
        _, c, _, _ = x1.shape

        # 处理batch size为1的情况
        if self.training and x1.size(0) == 1:
            with torch.no_grad():
                return self._forward_impl(x1, x2, c)
        else:
            return self._forward_impl(x1, x2, c)

    def _forward_impl(self, x1, x2, c):
        input = torch.cat([x1, x2], dim=1)
        recal_w = self.Recalibrate(input)
        recal_input = recal_w * input  ## 先对特征进行一步自校正
        recal_input = recal_input + input
        x1, x2 = torch.split(recal_input, c, dim=1)
        agg_input = self.channel_agg(recal_input)  ## 进行特征压缩 因为只计算一个特征的权重

        # 处理全局注意力中的BatchNorm问题
        if self.training and agg_input.size(0) == 1:
            with torch.no_grad():
                local_w = self.local_att(agg_input)  ## 局部注意力 即spatial attention
                global_w = self.global_att(agg_input)  ## 全局注意力 即channel attention
        else:
            local_w = self.local_att(agg_input)
            global_w = self.global_att(agg_input)

        w = self.sigmoid(local_w * global_w)  ## 计算特征x1的权重
        xo = w * x1 + (1 - w) * x2  ## fusion results ## 特征聚合
        return xo


# class SDFM(nn.Module):
#     '''
#     superficial detail fusion module
#     '''
#
#     def __init__(self, channels=64, r=4):
#         super(SDFM, self).__init__()
#         inter_channels = int(channels // r)
#
#         self.Recalibrate = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             Conv(2 * channels, 2 * inter_channels),
#             Conv(2 * inter_channels, 2 * channels, act=nn.Sigmoid()),
#         )
#
#         self.channel_agg = Conv(2 * channels, channels)
#
#         self.local_att = nn.Sequential(
#             Conv(channels, inter_channels, 1),
#             Conv(inter_channels, channels, 1, act=False),
#         )
#
#         self.global_att = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             Conv(channels, inter_channels, 1),
#             Conv(inter_channels, channels, 1),
#         )
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, data):
#         x1, x2 = data
#         _, c, _, _ = x1.shape
#         input = torch.cat([x1, x2], dim=1)
#         recal_w = self.Recalibrate(input)
#         recal_input = recal_w * input  ## 先对特征进行一步自校正
#         recal_input = recal_input + input
#         x1, x2 = torch.split(recal_input, c, dim=1)
#         agg_input = self.channel_agg(recal_input)  ## 进行特征压缩 因为只计算一个特征的权重
#         local_w = self.local_att(agg_input)  ## 局部注意力 即spatial attention
#         global_w = self.global_att(agg_input)  ## 全局注意力 即channel attention
#         w = self.sigmoid(local_w * global_w)  ## 计算特征x1的权重
#         xo = w * x1 + (1 - w) * x2  ## fusion results ## 特征聚合
#         return xo


#
# class SDFM(nn.Module):
#     def __init__(self, channels=64, r=4):
#         super(SDFM, self).__init__()
#         inter_channels = int(channels // r)
#
#         # 1. 三路联合校准
#         self.global_calib = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(3 * channels, inter_channels, 1),
#             nn.ReLU(),
#             nn.Conv2d(inter_channels, 3 * channels, 1),
#             nn.Sigmoid()
#         )
#
#         # 2. 两分支注意力（保持与原SDFM一致）
#         self.local_att = nn.Sequential(
#             nn.Conv2d(3 * channels, inter_channels, 1),
#             nn.ReLU(),
#             nn.Conv2d(inter_channels, 3 * channels, 1)
#         )
#         self.global_att = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(3 * channels, inter_channels, 1),
#             nn.ReLU(),
#             nn.Conv2d(inter_channels, 3 * channels, 1)
#         )
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x1, x2, x3):
#         _, c, _, _ = x1.shape
#
#         # 1. 三路拼接与联合校准
#         x_cat = torch.cat([x1, x2, x3], dim=1)  # [B, 3C, H, W]
#         calib_weights = self.global_calib(x_cat)  # [B, 3C, 1, 1]
#         a, b, c = torch.split(calib_weights * x_cat, c, dim=1)  # 各[B,C,H,W]
#
#         # 2. 交叉相加 (你的核心设计)
#         A = x1 + b + c  # x1与b,c交互
#         B = x2 + a + c  # x2与a,c交互
#         C = x3 + a + b  # x3与a,b交互
#         ABC = torch.cat([A, B, C], dim=1)  # [B, 3C, H, W]
#
#         # 3. 注意力融合（与原SDFM相同逻辑）
#         local_w = self.local_att(ABC)  # [B, 3C, H, W]
#         global_w = self.global_att(ABC)  # [B, 3C, 1, 1]
#         fused_weights = self.sigmoid(local_w * global_w)  # [B, 3C, H, W]
#
#         # 4. 加权输出（保持输出通道为C）
#         A_weight, B_weight, C_weight = torch.split(fused_weights, c, dim=1)
#         out = (A * A_weight + B * B_weight + C * C_weight) / 3  # [B, C, H, W]
#         return out

#
# class SDFM(nn.Module):
#     def __init__(self, c1, c2=None, r=4):  # 必须用c1作为主参数
#         super().__init__()
#         c2 = c2 or c1
#
#         # 1. 动态通道对齐
#         self.channel_align = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv2d(c1, c2, 1),
#                 nn.BatchNorm2d(c2),
#                 nn.SiLU()
#             ) for _ in range(3)
#         ])
#
#         # 2. 三路校准
#         self.global_calib = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(3 * c2, 3 * c2 // r, 1),
#             nn.ReLU(),
#             nn.Conv2d(3 * c2 // r, 3 * c2, 1),
#             nn.Sigmoid()
#         )
#
#         # 3. 注意力分支
#         self.local_att = nn.Sequential(
#             nn.Conv2d(3 * c2, 3 * c2 // r, 1),
#             nn.ReLU(),
#             nn.Conv2d(3 * c2 // r, 3 * c2, 1)
#         )
#         self.global_att = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(3 * c2, 3 * c2 // r, 1),
#             nn.ReLU(),
#             nn.Conv2d(3 * c2 // r, 3 * c2, 1)
#         )
#         self.sigmoid = nn.Sigmoid()
#
#         # 4. 输出控制
#         self.conv_out = nn.Conv2d(3 * c2, c2, 1)
#
#     def forward(self, x):
#         x1, x2, x3 = x
#         x1, x2, x3 = [align(t) for align, t in zip(self.channel_align, [x1, x2, x3])]
#
#         # 后续流程保持不变...
#         x_cat = torch.cat([x1, x2, x3], dim=1)
#         weights = self.global_calib(x_cat)
#         a, b, c = torch.split(weights * x_cat, x1.shape[1], dim=1)
#
#         A = x1 + b + c
#         B = x2 + a + c
#         C = x3 + a + b
#         ABC = torch.cat([A, B, C], dim=1)
#
#         local_w = self.local_att(ABC)
#         global_w = self.global_att(ABC)
#         fused_weights = self.sigmoid(local_w * global_w)
#
#         A_w, B_w, C_w = torch.split(fused_weights, x1.shape[1], dim=1)
#         out = (A * A_w + B * B_w + C * C_w) / 3
#         return self.conv_out(out)
#