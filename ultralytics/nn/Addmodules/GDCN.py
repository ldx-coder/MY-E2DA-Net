import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import math
import numpy as np

from ..modules.conv import Conv, autopad
from ..modules.block import *


__all__ = ['C2f_GDCN']



class GCCA(nn.Module):
    def __init__(self, inp, reduction=32):
        super(GCCA, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # [n, c, h, 1]
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # [n, c, 1, w]
        self.gap = nn.AdaptiveAvgPool2d(1)  # [n, c, 1, 1]

        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1)
        self.conv_g = nn.Conv2d(mip, inp, kernel_size=1)  # 新增GAP分支的卷积

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        # 1. 计算各分支特征
        x_h = self.pool_h(x)  # [n, c, h, 1]
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # [n, c, w, 1]
        x_gap = self.gap(x)  # [n, c, 1, 1]

        # 2. 调整GAP维度以匹配拼接
        x_gap = x_gap.expand(-1, -1, h, 1)  # [n, c, h, 1]（与x_h同维）

        # 3. 直接拼接三个分支
        y = torch.cat([x_h, x_w, x_gap], dim=2)  # [n, c, h+w+h, 1]

        # 4. 统一处理特征
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # 5. 分割时注意总长度是 h+w+h
        split_size = [h, w, h]  # 对应x_h, x_w, x_gap的维度
        x_h, x_w, x_gap = torch.split(y, split_size, dim=2)

        # 6. 调整维度
        x_w = x_w.permute(0, 1, 3, 2)  # [n, mip, 1, w]
        x_gap = x_gap.permute(0, 1, 3, 2)  # [n, mip, 1, h]

        # 7. 计算注意力权重
        a_h = self.conv_h(x_h).sigmoid()  # [n, c, h, 1]
        a_w = self.conv_w(x_w).sigmoid()  # [n, c, 1, w]
        a_g = self.conv_g(x_gap).sigmoid()  # [n, c, 1, h]

        # 8. 合并注意力（广播到完整尺寸）
        out = identity * a_w * a_h * a_g.permute(0, 1, 3, 2)  # a_g转置为 [n, c, h, 1]
        return out





class DCNv2_Offset_Attention(nn.Module):
    def __init__(self, in_channels, kernel_size, stride, deformable_groups=1) -> None:
        super().__init__()

        padding = autopad(kernel_size, None, 1)
        self.out_channel = (deformable_groups * 3 * kernel_size * kernel_size)
        self.conv_offset_mask = nn.Conv2d(in_channels, self.out_channel, kernel_size, stride, padding, bias=True)
        # self.attention = MPCA(self.out_channel)
        self.attention = GCCA(self.out_channel)

    def forward(self, x):
        conv_offset_mask = self.conv_offset_mask(x)
        conv_offset_mask = self.attention(conv_offset_mask)
        return conv_offset_mask


class GDCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=None, groups=1, dilation=1, act=True, deformable_groups=1):
        super(GDCN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        padding = autopad(kernel_size, padding, dilation)
        self.padding = (padding, padding)
        self.dilation = (dilation, dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, *self.kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels))

        self.conv_offset_mask = DCNv2_Offset_Attention(in_channels, kernel_size, stride, deformable_groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = Conv.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.has_deform_conv = hasattr(torch.ops.torchvision, 'deform_conv2d')
        self.reset_parameters()

    def forward(self, x):
        offset_mask = self.conv_offset_mask(x)
        o1, o2, mask = torch.chunk(offset_mask, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        # 修改这里：添加兼容性检查
        if self.has_deform_conv:
            # 使用 deform_conv2d
            x = torch.ops.torchvision.deform_conv2d(
                x,
                self.weight,
                offset,
                mask,
                self.bias,
                self.stride[0], self.stride[1],
                self.padding[0], self.padding[1],
                self.dilation[0], self.dilation[1],
                self.groups,
                self.deformable_groups,
                True
            )
        else:
            # 回退到普通卷积（作为替代方案）
            # 这里可以使用其他可变形卷积的实现，或者简单的普通卷积
            x = F.conv2d(
                x, self.weight, self.bias, self.stride,
                self.padding, self.dilation, self.groups
            )
        x = self.bn(x)
        x = self.act(x)
        return x

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        std = 1. / math.sqrt(n)
        self.weight.data.uniform_(-std, std)
        self.bias.data.zero_()
        self.conv_offset_mask.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.conv_offset_mask.bias.data.zero_()


class Bottleneck_GDCN(Bottleneck):
    """Standard bottleneck with DCNV2."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv2 = GDCN(c_, c2, k[1], 1)





class C2f_GDCN(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Bottleneck_GDCN(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))




class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))




class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6