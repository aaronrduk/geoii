"""
Road Centre Line detection head.

Detects the centreline of roads as thin linear features.
Uses multi-scale dilated convolutions to capture long-range connectivity.
"""

import torch
import torch.nn as nn


class RoadCenterlineHead(nn.Module):
    """
    Neural network head for road centre-line extraction.

    Road centrelines are thin (1-2 px wide) linear features.
    We use dilated (atrous) convolutions at multiple rates to expand
    the receptive field without losing spatial resolution, which is
    critical for preserving thin-structure connectivity.
    """

    def __init__(self, in_channels: int = 256):
        """
        Args:
            in_channels (int): Number of input channels from FPN.
        """
        super(RoadCenterlineHead, self).__init__()

        # ASPP-lite: three parallel dilated convolutions to capture
        # road context at different spatial scales.
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=8, dilation=8),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Fuse branches and progressively upsample
        self.fuse = nn.Sequential(
            nn.Conv2d(128 * 3, 256, kernel_size=1),
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

        # Final binary segmentation output
        self.seg_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),  # Binary: centreline vs background
        )

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x (torch.Tensor): FPN features (N, C, H, W)
        Returns:
            dict: {'road_centerline_mask': (N, 1, H, W)}
        """
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        fused = self.fuse(torch.cat([b1, b2, b3], dim=1))
        return {"road_centerline_mask": self.seg_head(fused)}
