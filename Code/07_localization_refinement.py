
import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalizationRefinement(nn.Module):
    def __init__(self, feature_channels=128):
        super().__init__()

        self.conv1 = nn.Conv2d(
            feature_channels + 1,
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

        self.conv3 = nn.Conv2d(
            32,
            1,
            kernel_size=1
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, features, coarse_map, output_size):

        coarse_small = F.interpolate(
            coarse_map,
            size=features.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        x = torch.cat(
            [features, coarse_small],
            dim=1
        )

        x = self.relu(
            self.bn1(
                self.conv1(x)
            )
        )

        x = self.relu(
            self.bn2(
                self.conv2(x)
            )
        )

        refined_map = self.conv3(x)

        refined_map = F.interpolate(
            refined_map,
            size=output_size,
            mode="bilinear",
            align_corners=False
        )

        return refined_map
