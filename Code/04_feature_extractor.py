
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        self.block1 = ConvBlock(3, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.block2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.block3 = ConvBlock(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        self.block4 = ConvBlock(128, 256)

    def forward(self, x):
        f1 = self.block1(x)
        x = self.pool1(f1)

        f2 = self.block2(x)
        x = self.pool2(f2)

        f3 = self.block3(x)
        x = self.pool3(f3)

        f4 = self.block4(x)

        return {
            "f1": f1,
            "f2": f2,
            "f3": f3,
            "f4": f4
        }
