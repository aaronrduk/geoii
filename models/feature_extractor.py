"""
Main feature extractor model combining backbone and all task heads.

This is the complete model architecture for SVAMITVA feature extraction,
supporting all 10 shapefile categories in the dataset.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, Optional

from .building_head import BuildingHead
from .road_head import RoadHead
from .road_centerline_head import RoadCenterlineHead
from .waterbody_head import WaterbodyHead
from .waterbody_line_head import WaterbodyLineHead
from .waterbody_point_head import WaterbodyPointHead
from .utility_head import UtilityLineHead, UtilityPolyHead
from .bridge_head import BridgeHead
from .railway_head import RailwayHead


class FeatureExtractorModel(nn.Module):
    """
    Multi-task feature extraction model for SVAMITVA drone imagery.

    Architecture:
        1. Shared Backbone (ResNet50) — Extracts features from input image
        2. Feature Pyramid Network (FPN) — Multi-scale feature fusion
        3. Task-specific Heads (10 shapefile types):
           - BuildingHead          → building_mask + roof_type
           - RoadHead              → road_mask          (Road polygon)
           - RoadCenterlineHead    → road_centerline_mask
           - WaterbodyHead         → waterbody_mask     (Water_Body polygon)
           - WaterbodyLineHead     → waterbody_line_mask
           - WaterbodyPointHead    → waterbody_point_mask
           - UtilityLineHead       → utility_line_mask  (pipelines, wires)
           - UtilityPolyHead       → utility_poly_mask  (transformers, tanks)
           - BridgeHead            → bridge_mask
           - RailwayHead           → railway_mask
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        num_roof_classes: int = 5,
    ):
        """
        Args:
            backbone (str): Backbone architecture ('resnet50', 'resnet34')
            pretrained (bool): Whether to use ImageNet pretrained weights
            num_roof_classes (int): Number of roof type classes (incl. background)
        """
        super(FeatureExtractorModel, self).__init__()

        self.backbone_name = backbone
        self.backbone, self.feature_channels = self._create_backbone(
            backbone, pretrained
        )

        # Feature Pyramid Network
        self.fpn = self._create_fpn(self.feature_channels)

        fpn_channels = 256  # Standard FPN output channels

        # ── Task heads ────────────────────────────────────────────────────────
        self.building_head = BuildingHead(
            in_channels=fpn_channels, num_roof_classes=num_roof_classes
        )
        self.road_head = RoadHead(in_channels=fpn_channels)
        self.road_centerline_head = RoadCenterlineHead(in_channels=fpn_channels)
        self.waterbody_head = WaterbodyHead(in_channels=fpn_channels)
        self.waterbody_line_head = WaterbodyLineHead(in_channels=fpn_channels)
        self.waterbody_point_head = WaterbodyPointHead(in_channels=fpn_channels)
        self.utility_line_head = UtilityLineHead(in_channels=fpn_channels)
        self.utility_poly_head = UtilityPolyHead(in_channels=fpn_channels)
        self.bridge_head = BridgeHead(in_channels=fpn_channels)
        self.railway_head = RailwayHead(in_channels=fpn_channels)

        self._initialize_weights()

    # ── Backbone ──────────────────────────────────────────────────────────────

    def _create_backbone(self, backbone: str, pretrained: bool):
        if backbone == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            resnet = models.resnet50(weights=weights)
            backbone_model = nn.ModuleDict(
                {
                    "conv1": resnet.conv1,
                    "bn1": resnet.bn1,
                    "relu": resnet.relu,
                    "maxpool": resnet.maxpool,
                    "layer1": resnet.layer1,  # C2: 256 ch
                    "layer2": resnet.layer2,  # C3: 512 ch
                    "layer3": resnet.layer3,  # C4: 1024 ch
                    "layer4": resnet.layer4,  # C5: 2048 ch
                }
            )
            feature_channels = {"C2": 256, "C3": 512, "C4": 1024, "C5": 2048}

        elif backbone == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            resnet = models.resnet34(weights=weights)
            backbone_model = nn.ModuleDict(
                {
                    "conv1": resnet.conv1,
                    "bn1": resnet.bn1,
                    "relu": resnet.relu,
                    "maxpool": resnet.maxpool,
                    "layer1": resnet.layer1,
                    "layer2": resnet.layer2,
                    "layer3": resnet.layer3,
                    "layer4": resnet.layer4,
                }
            )
            feature_channels = {"C2": 64, "C3": 128, "C4": 256, "C5": 512}
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        return backbone_model, feature_channels

    # ── FPN ───────────────────────────────────────────────────────────────────

    def _create_fpn(self, feature_channels: Dict[str, int]) -> nn.ModuleDict:
        fpn_channels = 256
        fpn = nn.ModuleDict(
            {
                "lateral_c5": nn.Conv2d(feature_channels["C5"], fpn_channels, 1),
                "lateral_c4": nn.Conv2d(feature_channels["C4"], fpn_channels, 1),
                "lateral_c3": nn.Conv2d(feature_channels["C3"], fpn_channels, 1),
                "lateral_c2": nn.Conv2d(feature_channels["C2"], fpn_channels, 1),
                "refine_c5": nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1),
                "refine_c4": nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1),
                "refine_c3": nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1),
                "refine_c2": nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1),
            }
        )
        return fpn

    # ── Weight initialisation ─────────────────────────────────────────────────

    def _initialize_weights(self):
        modules_to_init = [
            self.fpn,
            self.building_head,
            self.road_head,
            self.road_centerline_head,
            self.waterbody_head,
            self.waterbody_line_head,
            self.waterbody_point_head,
            self.utility_line_head,
            self.utility_poly_head,
            self.bridge_head,
            self.railway_head,
        ]
        for module in modules_to_init:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    # ── Forward passes ────────────────────────────────────────────────────────

    def forward_backbone(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.backbone["conv1"](x)
        x = self.backbone["bn1"](x)
        x = self.backbone["relu"](x)
        x = self.backbone["maxpool"](x)

        c2 = self.backbone["layer1"](x)  # 1/4 resolution
        c3 = self.backbone["layer2"](c2)  # 1/8
        c4 = self.backbone["layer3"](c3)  # 1/16
        c5 = self.backbone["layer4"](c4)  # 1/32

        return {"C2": c2, "C3": c3, "C4": c4, "C5": c5}

    def forward_fpn(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        p5 = self.fpn["lateral_c5"](features["C5"])

        p4 = self.fpn["lateral_c4"](features["C4"])
        p4 = p4 + F.interpolate(p5, size=p4.shape[-2:], mode="nearest")

        p3 = self.fpn["lateral_c3"](features["C3"])
        p3 = p3 + F.interpolate(p4, size=p3.shape[-2:], mode="nearest")

        p2 = self.fpn["lateral_c2"](features["C2"])
        p2 = p2 + F.interpolate(p3, size=p2.shape[-2:], mode="nearest")

        p5 = self.fpn["refine_c5"](p5)
        p4 = self.fpn["refine_c4"](p4)
        p3 = self.fpn["refine_c3"](p3)
        p2 = self.fpn["refine_c2"](p2)

        # P3 gives a good balance: 1/8 resolution with rich semantics
        return p3

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.

        Args:
            x (torch.Tensor): Input image batch (N, 3, H, W)

        Returns:
            dict: Predictions for all 10 shapefile categories:
                building_mask       (N, 1, H, W)  — binary
                roof_type           (N, C, H, W)  — multi-class (C = num_roof_classes)
                road_mask           (N, 1, H, W)
                road_centerline_mask(N, 1, H, W)
                waterbody_mask      (N, 1, H, W)
                waterbody_line_mask (N, 1, H, W)
                waterbody_point_mask(N, 1, H, W)
                utility_line_mask   (N, 1, H, W)
                utility_poly_mask   (N, 1, H, W)
                bridge_mask         (N, 1, H, W)
                railway_mask        (N, 1, H, W)
        """
        backbone_features = self.forward_backbone(x)
        fpn_features = self.forward_fpn(backbone_features)

        outputs = {
            **self.building_head(fpn_features),  # building_mask, roof_type
            **self.road_head(fpn_features),  # road_mask
            **self.road_centerline_head(fpn_features),  # road_centerline_mask
            **self.waterbody_head(fpn_features),  # waterbody_mask
            **self.waterbody_line_head(fpn_features),  # waterbody_line_mask
            **self.waterbody_point_head(fpn_features),  # waterbody_point_mask
            **self.utility_line_head(fpn_features),  # utility_line_mask
            **self.utility_poly_head(fpn_features),  # utility_poly_mask
            **self.bridge_head(fpn_features),  # bridge_mask
            **self.railway_head(fpn_features),  # railway_mask
        }

        # Upsample all outputs to input spatial resolution
        input_h, input_w = x.shape[2], x.shape[3]
        for key in outputs:
            if outputs[key].shape[-2:] != (input_h, input_w):
                outputs[key] = F.interpolate(
                    outputs[key],
                    size=(input_h, input_w),
                    mode="bilinear",
                    align_corners=False,
                )

        return outputs

    # ── Utilities ─────────────────────────────────────────────────────────────

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True


# ── Quick self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = FeatureExtractorModel(backbone="resnet50", pretrained=False)
    print(f"Backbone: {model.backbone_name}")
    print(f"Total parameters: {model.get_num_parameters():,}")

    dummy = torch.randn(2, 3, 512, 512)
    print(f"\nInput: {dummy.shape}")

    with torch.no_grad():
        out = model(dummy)

    print("\nOutputs:")
    for k, v in out.items():
        print(f"  {k}: {v.shape}")
