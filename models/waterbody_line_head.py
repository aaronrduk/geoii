"""
Waterbody Line detection head.

Detects linear water features such as drains, channels, and streams.
"""

import torch
import torch.nn as nn


class WaterbodyLineHead(nn.Module):
    """
    Neural network head for linear waterbody feature extraction.

    Linear water features (drains, canals) appear as narrow, elongated
    regions. Dilated convolutions are used to capture their extent
    without sacrificing spatial resolution.
    """

    def __init__(self, in_channels: int = 256):
        """
        Args:
            in_channels (int): Number of input channels from FPN.
        """
        super(WaterbodyLineHead, self).__init__()

        self.decoder = nn.Sequential(
            # Dilated conv to capture river/drain extent
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            # Standard refinement
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
            nn.Conv2d(32, 1, kernel_size=1),  # Binary segmentation
        )

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x (torch.Tensor): FPN features (N, C, H, W)
        Returns:
            dict: {'waterbody_line_mask': (N, 1, H, W)}
        """
        features = self.decoder(x)
        return {"waterbody_line_mask": self.seg_head(features)}
