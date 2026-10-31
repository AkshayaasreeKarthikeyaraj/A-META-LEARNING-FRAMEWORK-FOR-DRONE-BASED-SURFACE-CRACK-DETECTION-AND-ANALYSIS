
import torch
import torch.nn as nn
import torch.nn.functional as F


class DFEB(nn.Module):
    def __init__(self, in_channels=256, out_channels=128):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, padding=1
        )

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, padding=2, dilation=2
        )

        self.conv3 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, padding=4, dilation=4
        )

        self.conv4 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, padding=8, dilation=8
        )

        self.fuse = nn.Conv2d(
            out_channels * 4,
            out_channels,
            kernel_size=1
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x1))
        x4 = self.relu(self.conv4(x1))

        features = torch.cat([x1, x2, x3, x4], dim=1)

        out = self.fuse(features)
        out = self.bn(out)
        out = self.relu(out)

        return out
