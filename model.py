"""
CODNet — Camouflaged Object Detection Network
Architecture: convnext_small backbone + CBAM + FAM + SAM + deep supervision
Notebook 13 Upgrade
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class CBAM(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_c, in_c // 8, 1),
            nn.ReLU(),
            nn.Conv2d(in_c // 8, in_c, 1),
            nn.Sigmoid()
        )
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        x   = x * self.ca(x)
        avg = torch.mean(x, dim=1, keepdim=True)
        mx  = torch.max(x,  dim=1, keepdim=True)[0]
        x   = x * self.sa(torch.cat([avg, mx], dim=1))
        return x


class FAM(nn.Module):
    """Feature Aggregation Module"""
    def __init__(self, high_c, low_c, out_c):
        super().__init__()
        self.conv_high = nn.Conv2d(high_c, out_c, 1)
        self.conv_low  = nn.Conv2d(low_c,  out_c, 1)
        self.conv = nn.Sequential(
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

    def forward(self, x_high, x_low):
        x_high = F.interpolate(x_high, size=x_low.shape[2:],
                               mode='bilinear', align_corners=False)
        return self.conv(self.conv_high(x_high) + self.conv_low(x_low))


class SAM(nn.Module):
    """Scale-Aware Module — parallel dilated convolutions"""
    def __init__(self, in_c, out_c):
        super().__init__()
        mid = out_c // 4
        self.b1 = nn.Sequential(nn.Conv2d(in_c, mid, 1), nn.BatchNorm2d(mid), nn.ReLU())
        self.b2 = nn.Sequential(nn.Conv2d(in_c, mid, 3, padding=2, dilation=2), nn.BatchNorm2d(mid), nn.ReLU())
        self.b3 = nn.Sequential(nn.Conv2d(in_c, mid, 3, padding=4, dilation=4), nn.BatchNorm2d(mid), nn.ReLU())
        self.b4 = nn.Sequential(nn.Conv2d(in_c, mid, 3, padding=8, dilation=8), nn.BatchNorm2d(mid), nn.ReLU())
        self.fuse = nn.Sequential(
            nn.Conv2d(out_c, out_c, 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

    def forward(self, x):
        return self.fuse(torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1))


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )
        self.attn = CBAM(out_c)

    def forward(self, x):
        return self.attn(self.conv(x))


class CODNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            'convnext_small', pretrained=False, features_only=True
        )
        ch = self.backbone.feature_info.channels()  # [96, 192, 384, 768]

        self.sam4 = SAM(ch[3], 256)
        self.fam4 = FAM(256,  ch[2], 256)
        self.fam3 = FAM(128,  ch[1], 128)
        self.fam2 = FAM(64,   ch[0], 64)

        self.up4 = ConvBlock(256, 256)
        self.up3 = ConvBlock(256, 128)
        self.up2 = ConvBlock(128, 64)
        self.up1 = ConvBlock(64,  32)

        self.final     = nn.Conv2d(32, 1, 1)
        self.edge_head = nn.Conv2d(32, 1, 1)
        self.ds3       = nn.Conv2d(128, 1, 1)
        self.ds4       = nn.Conv2d(256, 1, 1)

    def forward(self, x):
        f1, f2, f3, f4 = self.backbone(x)

        d4 = self.up4(self.sam4(f4))
        d3 = self.up3(self.fam4(d4, f3))
        d2 = self.up2(self.fam3(d3, f2))
        d1 = self.up1(self.fam2(d2, f1))

        out  = self.final(d1)
        edge = self.edge_head(d1)
        out3 = self.ds4(d4)
        out4 = self.ds3(d3)

        return out, edge, out3, out4
