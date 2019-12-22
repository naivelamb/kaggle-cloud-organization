# @Author: Xuan Cao <xuan>
# @Date:   2019-12-22, 12:19:37
# @Last modified by:   xuan
# @Last modified time: 2019-12-22, 1:33:55



import torch
import torch.nn as nn
import torch.nn.functional as F

def resize_like(x, reference, mode='bilinear'):
    if x.shape[2:] !=  reference.shape[2:]:
        if mode=='bilinear':
            x = F.interpolate(x, size=reference.shape[2:],mode='bilinear',align_corners=False)
        if mode=='nearest':
            x = F.interpolate(x, size=reference.shape[2:],mode='nearest')
    return x

class SeparableConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv = nn.Conv2d(in_channel, in_channel, kernel_size, stride, padding, dilation, groups=in_channel, bias=bias)
        self.bn   = nn.BatchNorm2d(in_channel)
        self.pointwise = nn.Conv2d(in_channel, out_channel, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x

class JPU(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(JPU, self).__init__()

        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channel[0], out_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel[1], out_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channel[2], out_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

        self.dilation0 = nn.Sequential(
            SeparableConv2d(3*out_channel, out_channel, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.dilation1 = nn.Sequential(
            SeparableConv2d(3*out_channel, out_channel, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.dilation2 = nn.Sequential(
            SeparableConv2d(3*out_channel, out_channel, kernel_size=3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.dilation3 = nn.Sequential(
            SeparableConv2d(3*out_channel, out_channel, kernel_size=3, padding=8, dilation=8, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        x0 = self.conv0(x[0])
        x1 = self.conv1(x[1])
        x2 = self.conv2(x[2])

        x0 = resize_like(x0, x2, mode='nearest')
        x1 = resize_like(x1, x2, mode='nearest')
        x = torch.cat([x0,x1,x2], dim=1)

        d0 = self.dilation0(x)
        d1 = self.dilation1(x)
        d2 = self.dilation2(x)
        d3 = self.dilation3(x)
        x = torch.cat([d0,d1,d2,d3], dim=1)
        return x
