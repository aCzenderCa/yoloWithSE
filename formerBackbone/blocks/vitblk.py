import einops
import torch
import torch.nn.functional as F
from torch import nn

from ultralytics_local.ultralytics.nn.modules import CBAM
from ultralytics_local.ultralytics.nn.modules.activation import AGLU


class ViTBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        ch_scale = out_channel / in_channel
        stride = int(ch_scale) if ch_scale >= 1 else 1

        self.dwconv = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, groups=in_channel)
        self.cbam = CBAM(in_channel)
        self.bn0 = nn.BatchNorm2d(in_channel)
        self.conv = nn.Conv2d(in_channel, int(in_channel * ch_scale), kernel_size=1, stride=stride)
        self.act = AGLU()
        self.ch_scale = ch_scale
        self.stride = stride
        self.bn1 = nn.BatchNorm2d(int(in_channel * ch_scale))
        if self.ch_scale != 1 or self.stride != 1:
            self.avgpool = nn.AdaptiveAvgPool2d(self.stride)

    def forward(self, x: torch.Tensor):
        x = self.dwconv(x)
        x = self.cbam(x)
        x = self.bn0(x)
        x = self.act(x)

        y = self.conv(x)
        if self.ch_scale == 1 and self.stride == 1:
            y = y + x
        else:
            x = self.avgpool(x)
            x = einops.reduce(x, "n c s1 s2 -> n (c s1) 1 1", reduction="mean")
            if self.ch_scale < 1:
                x = einops.reduce(x, f"n (c {int(1 / self.ch_scale)}) 1 1 -> n c 1 1", reduction="mean")
            y = x * y
        y = self.bn1(y)
        return y
