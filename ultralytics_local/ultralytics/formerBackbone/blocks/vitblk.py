import torch
from torch import nn
from torch.nn import init

from ultralytics.nn.modules import CBAM


class ViTBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=None):
        super().__init__()
        ch_scale = out_channel / in_channel
        if not stride:
            stride = int(ch_scale) if ch_scale >= 1 else 1

        self.dwconv = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, groups=in_channel)
        self.cbam = CBAM(in_channel)
        self.bn0 = nn.BatchNorm2d(in_channel)
        self.conv = nn.Conv2d(in_channel, int(in_channel * ch_scale), kernel_size=1, stride=stride)
        self.act = nn.GELU()
        self.ch_scale = ch_scale
        self.stride = stride
        self.bn1 = nn.BatchNorm2d(int(in_channel * ch_scale))

    def forward(self, x: torch.Tensor):
        raw_x = x
        x = self.dwconv(x)
        x = self.cbam(x)
        x = x + raw_x
        x = self.bn0(x)
        x = self.act(x)

        y = self.conv(x)
        y = self.bn1(y)
        return y


class ViTBlock2(nn.Module):
    def __init__(self, in_channel, out_channel, stride=None):
        super().__init__()
        ch_scale = out_channel / in_channel
        if not stride:
            stride = int(ch_scale) if ch_scale >= 1 else 1

        self.dwconv = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, groups=in_channel)
        self.cbam = CBAM(in_channel)
        self.bn0 = nn.BatchNorm2d(in_channel)
        self.conv = nn.Conv2d(in_channel, int(in_channel * ch_scale), kernel_size=1, stride=stride)
        self.act = nn.GELU()
        self.ch_scale = ch_scale
        self.stride = stride
        self.bn1 = nn.BatchNorm2d(int(in_channel * ch_scale))

    def forward(self, x: torch.Tensor):
        raw_x = x
        x = self.dwconv(x)
        x = self.cbam(x)
        x = x + raw_x
        raw_x = x
        x = self.bn0(x)
        x = self.act(x)
        x = x + raw_x

        y = self.conv(x)
        y = self.bn1(y)
        return y


class ViTBlock3(nn.Module):
    def __init__(self, in_channel, out_channel, stride=None):
        super().__init__()
        ch_scale = out_channel / in_channel
        if not stride:
            stride = int(ch_scale) if ch_scale >= 1 else 1

        self.dwconv = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, groups=in_channel)
        self.cbam = CBAM(in_channel)
        self.bn0 = nn.BatchNorm2d(in_channel)
        self.conv = nn.Conv2d(in_channel, int(in_channel * ch_scale), kernel_size=1, stride=stride)
        self.act = nn.GELU()
        self.ch_scale = ch_scale
        self.stride = stride
        self.bn1 = nn.BatchNorm2d(int(in_channel * ch_scale))
        self.pconv = nn.Conv2d(in_channel, int(in_channel * ch_scale), kernel_size=1, stride=stride,
                               groups=int(in_channel * ch_scale) if ch_scale <= 1 else in_channel)
        init.constant_(self.pconv.weight, 1)

    def forward(self, x: torch.Tensor):
        raw_x = x
        x = self.dwconv(x)
        x = self.cbam(x)
        x = self.bn0(x)
        x = self.act(x)
        x = x + raw_x

        y = self.conv(x)
        y = self.bn1(y)
        y = self.pconv(raw_x) + y
        return y


class ViTBlock4(nn.Module):
    def __init__(self, in_channel, out_channel, stride=None):
        super().__init__()
        ch_scale = out_channel / in_channel
        if not stride:
            stride = int(ch_scale) if ch_scale >= 1 else 1

        self.dwconv = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=stride, padding=1, groups=in_channel)
        self.cbam = CBAM(in_channel)
        self.bn0 = nn.BatchNorm2d(in_channel)
        self.conv = nn.Conv2d(in_channel, int(in_channel * ch_scale), kernel_size=1, stride=1)
        self.act = nn.GELU()
        self.ch_scale = ch_scale
        self.stride = stride
        self.bn1 = nn.BatchNorm2d(int(in_channel * ch_scale))
        self.pconv = nn.Conv2d(in_channel, int(in_channel * ch_scale), kernel_size=1,
                               groups=int(in_channel * ch_scale) if ch_scale <= 1 else in_channel)
        self.scale = nn.Sequential()
        if stride > 1:
            self.scale.append(nn.MaxPool2d(stride + 1, stride, padding=int((stride + 1) / 2)))
        init.constant_(self.pconv.weight, 1)

    def forward(self, x: torch.Tensor):
        raw_x = x
        x = self.dwconv(x)
        x = self.cbam(x)
        x = self.bn0(x)
        x = self.act(x)
        raw_x = self.scale(raw_x)
        x = x + raw_x

        y = self.conv(x)
        y = self.bn1(y)
        y = self.pconv(raw_x) + y
        return y


class ViTBlock5(nn.Module):
    def __init__(self, in_channel, out_channel, stride=None):
        super().__init__()
        ch_scale = out_channel / in_channel
        if not stride:
            stride = int(ch_scale) if ch_scale >= 1 else 1

        self.dwconv = nn.Conv2d(in_channel, in_channel, kernel_size=5, stride=stride, padding=2, groups=in_channel)
        self.cbam = CBAM(in_channel)
        self.bn0 = nn.BatchNorm2d(in_channel)
        self.conv = nn.Conv2d(in_channel, int(in_channel * ch_scale), kernel_size=1, stride=1)
        self.act = nn.GELU()
        self.ch_scale = ch_scale
        self.stride = stride
        self.bn1 = nn.BatchNorm2d(int(in_channel * ch_scale))
        p_gp = int(in_channel * ch_scale) if ch_scale <= 1 else in_channel
        if ((out_channel > in_channel and out_channel % in_channel != 0) or
                (out_channel < in_channel and in_channel % out_channel != 0)):
            p_gp = 1
        self.pconv = nn.Conv2d(in_channel, int(in_channel * ch_scale), kernel_size=1, groups=p_gp, bias=False)
        self.scale = nn.Sequential()
        if stride > 1:
            self.scale.append(nn.MaxPool2d(5, stride, padding=2))

        init.constant_(self.pconv.weight, 1)

    def forward(self, x: torch.Tensor):
        raw_x = x
        x = self.dwconv(x)
        x = self.cbam(x)
        x = self.bn0(x)
        x = self.act(x)
        raw_x = self.scale(raw_x)
        x = x + raw_x

        y = self.conv(x)
        y = self.bn1(y)
        y = self.pconv(raw_x) + y
        return y
