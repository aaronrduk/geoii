"""
Utility infrastructure detection heads.

Two heads cover the two Utility shapefile types in the dataset:
  - UtilityLineHead  → Utility.shp (pipelines, overhead wires, poles)
  - UtilityPolyHead  → Utility_Poly_.shp (transformers, overhead tanks,
                        pump houses, substations)
"""

import torch
import torch.nn as nn


class UtilityLineHead(nn.Module):
    """
    Head for linear utility infrastructure (pipelines, wires, poles).

    Linear utilities share properties with roads (thin, elongated),
    so we use dilated convolutions for wide receptive field.
    """

    def __init__(self, in_channels: int = 256):
        super(UtilityLineHead, self).__init__()

        self.decoder = nn.Sequential(
            # Dilated conv for linear feature context
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            # 1×1 bottleneck for fine detail
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.seg_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),  # Binary: utility line vs background
        )

    def forward(self, x: torch.Tensor) -> dict:
        """
        Returns:
            dict: {'utility_line_mask': (N, 1, H, W)}
        """
        features = self.decoder(x)
        return {"utility_line_mask": self.seg_head(features)}


class UtilityPolyHead(nn.Module):
    """
    Head for polygon utility infrastructure (transformers, tanks, substations).

    Polygon utilities are small but compact objects. A channel-attention
    squeeze-and-excitation block helps focus on discriminative channels.
    """

    def __init__(self, in_channels: int = 256):
        super(UtilityPolyHead, self).__init__()

        # Squeeze-and-excitation channel attention
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.seg_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),  # Binary: utility polygon vs background
        )

    def forward(self, x: torch.Tensor) -> dict:
        """
        Returns:
            dict: {'utility_poly_mask': (N, 1, H, W)}
        """
        # Apply channel attention before decoding
        x = x * self.se(x)
        features = self.decoder(x)
        return {"utility_poly_mask": self.seg_head(features)}
