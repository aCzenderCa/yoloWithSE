import math

import torch
from torch import nn
from torch.nn import init

from ultralytics.nn.modules import CBAM, Bottleneck


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

        # 5x5*inc*w*h + inc*w*h*3 + inc*outc*inc*w*h + outc*w*h*3
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
        init.uniform_(self.dwconv.bias, -0.1, 0.1)
        init.uniform_(self.cbam.channel_attention.fc.bias, -0.1, 0.1)

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


class ViTBlock6P(nn.Module):
    def __init__(self, in_channel, out_channel, stride=None, rep=1):
        super().__init__()
        ch_scale = out_channel / in_channel
        if not stride:
            stride = int(ch_scale) if ch_scale >= 1 else 1

        self.dwconv = nn.Conv2d(in_channel, in_channel, kernel_size=5, stride=stride, padding=2, groups=in_channel)
        self.cbam = CBAM(in_channel)
        self.bn0 = nn.BatchNorm2d(in_channel)
        self.act = nn.GELU()
        self.ch_scale = ch_scale
        self.stride = stride

        self.post = []
        for i in range(rep):
            if i == 0:
                self.post.append(nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1))
            else:
                self.post.append(
                    nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, groups=out_channel))
            self.post.append(nn.BatchNorm2d(out_channel))
        self.post = nn.Sequential(*self.post)

        p_gp = int(in_channel * ch_scale) if ch_scale <= 1 else in_channel
        if ((out_channel > in_channel and out_channel % in_channel != 0) or
                (out_channel < in_channel and in_channel % out_channel != 0)):
            p_gp = 1
        self.pconv = nn.Conv2d(in_channel, int(in_channel * ch_scale), kernel_size=1, groups=p_gp, bias=False)
        self.scale = nn.Sequential()
        if stride > 1:
            self.scale.append(nn.MaxPool2d(5, stride, padding=2))

        init.constant_(self.pconv.weight, 1)
        init.uniform_(self.dwconv.bias, -0.1, 0.1)
        init.uniform_(self.cbam.channel_attention.fc.bias, -0.1, 0.1)

    def forward(self, x: torch.Tensor):
        raw_x = x
        x = self.dwconv(x)
        x = self.cbam(x)
        x = self.bn0(x)
        x = self.act(x)
        raw_x = self.scale(raw_x)
        x = x + raw_x

        y = self.post(x)
        y = self.pconv(raw_x) + y
        return y


class ViTBlockS1P(nn.Module):
    def __init__(self, in_channel, out_channel, stride=None, rep=1):
        super().__init__()
        assert in_channel % 2 == 0
        self.extra_ch = out_channel - in_channel // 2
        self.in_seq = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // 2, kernel_size=1, stride=stride),
        )
        if self.extra_ch <= 0:
            self.in_seq.append(nn.Conv2d(in_channel // 2, out_channel - 1, kernel_size=1))

        self.act = nn.GELU()
        self.bnY = nn.BatchNorm2d(out_channel)

        if self.extra_ch > 0:
            gcd_for_out = math.gcd(self.extra_ch, in_channel)
            self.out_seq = nn.Sequential(
                nn.Conv2d(in_channel, self.extra_ch, kernel_size=5, stride=stride, groups=gcd_for_out),
                nn.BatchNorm2d(self.extra_ch),
                CBAM(self.extra_ch),
            )

            for _ in range(rep - 1):
                self.in_seq.append(nn.BatchNorm2d(in_channel // 2))
                self.in_seq.append(nn.Conv2d(in_channel // 2, in_channel // 2, kernel_size=1, stride=1))

                self.out_seq.append(
                    nn.Conv2d(self.extra_ch, self.extra_ch, kernel_size=5, stride=stride, groups=self.extra_ch))
                self.out_seq.append(nn.BatchNorm2d(self.extra_ch))
        else:
            self.out_seq = nn.Conv2d(in_channel, 1, kernel_size=5, stride=stride, groups=in_channel)


    def forward(self, x: torch.Tensor):
        y1 = self.in_seq(x)
        y2 = self.out_seq(x)
        y = torch.cat([y1, y2], 1) # b c x y
        y = self.bnY(y)
        y = self.act(y)

        return y

