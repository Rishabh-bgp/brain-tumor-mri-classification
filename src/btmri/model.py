"""EfficientNet-B0 classifier and an export wrapper that also returns class activation maps."""
from __future__ import annotations

import timm
import torch
import torch.nn as nn

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
ARCH = "efficientnet_b0.ra_in1k"
IMG_SIZE = 224


def build_model(pretrained: bool = True, num_classes: int = 4) -> nn.Module:
    return timm.create_model(ARCH, pretrained=pretrained, num_classes=num_classes, drop_rate=0.2)


def to_input(gray_uint8: torch.Tensor) -> torch.Tensor:
    """(N,H,W) or (N,1,H,W) uint8 greyscale -> ImageNet-normalised (N,3,H,W) float."""
    x = gray_uint8.float().div(255.0)
    if x.dim() == 3:
        x = x.unsqueeze(1)
    x = x.expand(-1, 3, -1, -1)
    mean = torch.tensor(MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


class ExportWrapper(nn.Module):
    """Input: (N,1,224,224) greyscale in [0,1]. Outputs: logits (N,4) and CAM maps (N,4,7,7).

    For a global-average-pooled linear head, Grad-CAM weights equal the classifier weights divided by the number of
    spatial positions, so these maps are Grad-CAM maps up to a positive scale (before ReLU)."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor):
        x = (x.expand(-1, 3, -1, -1) - self.mean) / self.std
        feats = self.model.forward_features(x)  # (N,1280,7,7)
        logits = self.model.forward_head(feats)
        w = self.model.classifier.weight  # (4,1280)
        cam = torch.einsum("kc,nchw->nkhw", w, feats)
        return logits, cam
