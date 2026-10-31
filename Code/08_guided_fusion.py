
import torch
import torch.nn as nn
import torch.nn.functional as F


class GuidedFusion(nn.Module):
    def __init__(self, feature_channels=128):
        super().__init__()

        self.feature_projection = nn.Conv2d(
            feature_channels,
            feature_channels,
            kernel_size=1
        )

        self.guidance_projection = nn.Conv2d(
            1,
            feature_channels,
            kernel_size=1
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(
                feature_channels * 2,
                feature_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, segmentation_features, refined_map):

        refined_small = F.interpolate(
            refined_map,
            size=segmentation_features.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        feature_part = self.feature_projection(
            segmentation_features
        )

        guidance_part = self.guidance_projection(
            refined_small
        )

        fused = torch.cat(
            [feature_part, guidance_part],
            dim=1
        )

        fused = self.fusion(fused)

        return fused
