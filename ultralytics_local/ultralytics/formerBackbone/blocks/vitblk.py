import math

import einops
import einops.layers.torch as einn
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F

from ultralytics.nn.modules import CBAM, Bottleneck, ChannelAttention, SpatialAttention, RepVGGDW, RepConv, DWConv, \
    Conv, LightConv


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


class SAWithBn(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = SpatialAttention()

    def forward(self, x):
        raw_x = x
        x = self.sa(x)
        return torch.cat([raw_x, x], dim=1)


class ViTBlock6P(nn.Module):
    def __init__(self, in_channel, out_channel, stride=None):
        super().__init__()
        ch_scale = out_channel / in_channel
        if not stride:
            stride = int(ch_scale) if ch_scale >= 1 else 1

        self.small_blk = nn.Sequential(
            DWConv(in_channel, in_channel, k=5, s=stride),
            MHSpatialAttentionP(in_channel, math.gcd(in_channel, 4)),
        )
        self.scale = nn.Sequential()
        if stride > 1:
            self.scale.append(nn.MaxPool2d(5, stride, padding=2))

        self.post = nn.Sequential(
            LightConv(in_channel, out_channel, k=3, act=False),
        )

        self.pconv = nn.Sequential(
        )
        if out_channel != in_channel:
            self.pconv.append(DWConv(in_channel, out_channel, k=1, act=False))

    def forward(self, x: torch.Tensor):
        raw_x = x
        x = self.small_blk(x)
        raw_x1 = self.scale(raw_x)
        x = x + raw_x1

        y = self.post(x)
        y = self.pconv(raw_x1) + y
        return y


class DWConvK(nn.Module):
    def __init__(self, in_channel, out_channel, k=3, s=1, act=True):
        super().__init__()
        self.dwconv = DWConv(in_channel, out_channel, k, s, act=act)
        self.add = in_channel == out_channel and s == 1

    def forward(self, x):
        if self.add:
            return x + self.dwconv(x)
        return self.dwconv(x)


class ViTBlock6PRep(nn.Module):
    def __init__(self, in_channel, out_channel, rep=1):
        super().__init__()

        self.small_blk = nn.Sequential(
            DWConv(in_channel, out_channel, k=7),
            CBAM(out_channel),
        )

        self.net = nn.Sequential(
        )
        for i in range(rep):
            self.net.append(SpatialAttention())
            self.net.append(DWConvK(out_channel, out_channel, k=7, act=False))

        self.pconv = nn.Sequential(
        )
        if out_channel != in_channel:
            pc = DWConv(in_channel, out_channel, k=1)
            self.pconv.append(pc)
            init.constant_(pc.conv.weight, 1)

    def forward(self, x: torch.Tensor):
        raw_x = self.pconv(x)
        x = self.small_blk(x)
        y = self.net(x + raw_x)

        return y + raw_x + x


class MHSpatialAttentionP(nn.Module):
    def __init__(self, ch, head):
        super().__init__()
        self.att_in = nn.Sequential(
            einn.Reduce(f"b (c {ch // head}) h w -> b c h w", "mean")
        )
        self.att_conv = nn.Sequential(
            LightConv(head, head, k=7, act=False),
        )
        self.att_out = nn.Sequential(
            einn.Reduce(f"b c h w -> b (c {ch // head}) h w", "repeat")
        )

        self.ch_att = ChannelAttention(ch)

    def forward(self, x: torch.Tensor):
        att_x = self.att_in(x)
        att_x = self.att_conv(att_x)
        att_x = self.att_out(att_x)
        att_x = att_x * self.ch_att(x)
        return att_x * x


class MHSpatialAttentionWithBn(nn.Module):
    def __init__(self, ch, head):
        super().__init__()
        self.att = MHSpatialAttentionP(ch, head)

    def forward(self, x: torch.Tensor):
        att_x = self.att(x)
        return torch.cat([x, att_x], dim=1)


class ViTBlock1PPRep(nn.Module):
    def __init__(self, in_channel, out_channel, rep=1, emb_head=None):
        super().__init__()

        self.small_blk = nn.Sequential(
            DWConv(in_channel, out_channel, k=7),
            ChanSpatialAttention(out_channel),
        )

        if emb_head is not None:
            self.emb = nn.Parameter(torch.ones((emb_head, in_channel)))

        self.net = nn.Sequential(
        )
        for i in range(rep):
            self.net.append(ChanSpatialAttention(out_channel))
            self.net.append(DWConvK(out_channel, out_channel, k=7, act=False))

        self.pconv = nn.Sequential(
        )
        if out_channel != in_channel:
            pc = DWConv(in_channel, out_channel, k=1, act=False)
            self.pconv.append(pc)
            init.constant_(pc.conv.weight, 1)

    def forward(self, x: torch.Tensor):
        raw_x = self.pconv(x)
        x = self.small_blk(x)
        y = self.net(x + raw_x)

        return y + raw_x + x


class ViTBlock1PPEmb(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, emb_head=1):
        super().__init__()

        self.emb = nn.Parameter(torch.rand((emb_head, in_channel)) * 0.9 + 0.1)
        self.emb_head = emb_head
        self.in_c = in_channel

        self.small_blk = nn.Sequential(
            Conv(in_channel, out_channel // emb_head, k=3, s=stride),
            SpatialAttention(),
        )

        self.net = nn.Sequential(
            DWConv(out_channel, out_channel, k=7, act=False),
        )

    def forward(self, x: torch.Tensor):
        xs = []
        for i in range(self.emb_head):
            x_i = self.emb[i].reshape((1, self.in_c, 1, 1)) * x
            x_i = self.small_blk(x_i)
            xs.append(x_i)
        x = torch.cat(xs, dim=1)
        y = self.net(x)

        return y + x


class ViTBlock1PP(nn.Module):
    def __init__(self, in_channel, out_channel, stride=None):
        super().__init__()

        self.small_blk = nn.Sequential(
            DWConv(in_channel, out_channel, k=7, s=stride),
            ChanSpatialAttention(out_channel),
        )
        self.scale = nn.Sequential()
        if stride > 1:
            self.scale.append(nn.MaxPool2d(5, stride, padding=2))

        self.pconv = nn.Sequential(
        )
        if out_channel != in_channel:
            pc = DWConv(in_channel, out_channel, k=1, act=False)
            self.pconv.append(pc)
            init.constant_(pc.conv.weight, 1)

    def forward(self, x: torch.Tensor):
        raw_x = x
        y = self.small_blk(x)
        y = y + self.pconv(self.scale(raw_x))
        return y


class ChanSpatialAttention(nn.Module):
    def __init__(self, ch, kernel_size=7):
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(ch, ch, 1, 1, 0, bias=True)

    def forward(self, x):
        x_l = self.fc(self.pool(x))

        f = self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)) * x_l)
        return x * f
