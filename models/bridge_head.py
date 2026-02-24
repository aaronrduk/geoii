"""
Bridge detection head.

Detects bridges, which appear as polygons or linear structures
crossing over water bodies or roads.
"""

import torch
import torch.nn as nn


class BridgeHead(nn.Module):
    """
    Neural network head for bridge segmentation.

    Bridges have mixed polygon/line geometry and often overlap
    with road and waterbody features. A standard decoder is used
    with sufficient depth to disentangle bridges from context.
    """

    def __init__(self, in_channels: int = 256):
        """
        Args:
            in_channels (int): Number of input channels from FPN.
        """
        super(BridgeHead, self).__init__()

        self.decoder = nn.Sequential(
            # Wide receptive field to capture bridge geometry
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=2, dilation=2),
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
            nn.Conv2d(32, 1, kernel_size=1),  # Binary: bridge vs background
        )

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x (torch.Tensor): FPN features (N, C, H, W)
        Returns:
            dict: {'bridge_mask': (N, 1, H, W)}
        """
        features = self.decoder(x)
        return {"bridge_mask": self.seg_head(features)}
