"""
Railway detection head.

Detects railway lines as elongated linear features distinct from roads.
"""

import torch
import torch.nn as nn


class RailwayHead(nn.Module):
    """
    Neural network head for railway line segmentation.

    Railway lines are thin linear features. They differ from roads
    in texture (ballast, parallel rails) and are extracted using
    multi-scale dilated convolutions to capture long-range connectivity.
    """

    def __init__(self, in_channels: int = 256):
        """
        Args:
            in_channels (int): Number of input channels from FPN.
        """
        super(RailwayHead, self).__init__()

        self.decoder = nn.Sequential(
            # Dilated conv: wider receptive field for linear connectivity
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=3, dilation=3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.seg_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),  # Binary: railway vs background
        )

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x (torch.Tensor): FPN features (N, C, H, W)
        Returns:
            dict: {'railway_mask': (N, 1, H, W)}
        """
        features = self.decoder(x)
        return {"railway_mask": self.seg_head(features)}
