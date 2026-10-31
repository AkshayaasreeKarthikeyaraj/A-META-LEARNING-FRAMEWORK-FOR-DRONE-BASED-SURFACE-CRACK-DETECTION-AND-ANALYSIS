
import torch
import torch.nn as nn
import torch.nn.functional as F


class ADCB(nn.Module):
    def __init__(self, channels=128):
        super().__init__()

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            dilation=1
        )

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=2,
            dilation=2
        )

        self.conv3 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=4,
            dilation=4
        )

        self.fusion = nn.Conv2d(
            channels * 3,
            channels,
            kernel_size=1
        )

        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x))
        x3 = self.relu(self.conv3(x))

        x = torch.cat([x1, x2, x3], dim=1)

        x = self.fusion(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class DALFM(nn.Module):
    def __init__(self, channels=128):
        super().__init__()

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(
                2,
                1,
                kernel_size=7,
                padding=3
            ),
            nn.Sigmoid()
        )

    def forward(self, x):

        channel_weight = self.channel_attention(x)

        x_channel = x * channel_weight

        avg_map = torch.mean(
            x_channel,
            dim=1,
            keepdim=True
        )

        max_map, _ = torch.max(
            x_channel,
            dim=1,
            keepdim=True
        )

        spatial_input = torch.cat(
            [avg_map, max_map],
            dim=1
        )

        spatial_weight = self.spatial_attention(
            spatial_input
        )

        output = x_channel * spatial_weight

        return output
