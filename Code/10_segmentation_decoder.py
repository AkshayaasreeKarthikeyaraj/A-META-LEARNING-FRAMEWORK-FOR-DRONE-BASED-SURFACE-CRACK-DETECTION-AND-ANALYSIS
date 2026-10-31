
import torch
import torch.nn as nn
import torch.nn.functional as F

class SegmentationDecoder(nn.Module):
    def __init__(self, in_channels=128):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=3,
            padding=1
        )

        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(
            64,
            32,
            kernel_size=3,
            padding=1
        )

        self.bn2 = nn.BatchNorm2d(32)

        self.output = nn.Conv2d(
            32,
            1,
            kernel_size=1
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, output_size):

        x = self.relu(
            self.bn1(
                self.conv1(x)
            )
        )

        x = F.interpolate(
            x,
            size=output_size,
            mode="bilinear",
            align_corners=False
        )

        x = self.relu(
            self.bn2(
                self.conv2(x)
            )
        )

        x = self.output(x)

        return x
