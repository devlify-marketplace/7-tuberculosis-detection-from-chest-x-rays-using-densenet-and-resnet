"""Model builders for the DenseNet121 / ResNet50 comparison."""
from __future__ import annotations

import torch.nn as nn
from torchvision import models

SUPPORTED_ARCHITECTURES = ("densenet121", "resnet50")


def build_model(architecture: str, pretrained: bool = True, freeze_backbone: bool = True) -> nn.Module:
    architecture = architecture.lower()
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(f"architecture must be one of {SUPPORTED_ARCHITECTURES}")
    if architecture == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, 2)
        head = model.classifier
    else:
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 2)
        head = model.fc
    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in head.parameters():
            parameter.requires_grad = True
    return model
