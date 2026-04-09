"""
Model architectures for the occlusion detection classifier.

ResNet18 and ConvNeXt-Small with a binary classification head (clean vs
occluded). These are the two architectures the published
``smegmarip/face-recognition/occlusion_classifier.onnx`` was trained with.

Used by ``export_onnx.py`` to instantiate the correct torchvision backbone
before loading a ``.pth`` checkpoint and exporting to ONNX. If you retrain
the classifier with a different backbone, add a new factory function here
and update the ``training.model`` value in ``export_config.yaml``.
"""
from __future__ import annotations

import torch.nn as nn
from torchvision.models import (
    ResNet18_Weights,
    ConvNeXt_Small_Weights,
    convnext_small,
    resnet18,
)


def get_resnet18_occlusion(pretrained: bool = True) -> nn.Module:
    """Return a ResNet18 with a 2-class binary classification head.

    Args:
        pretrained: When True, loads ImageNet-pretrained torchvision weights
            as a starting point for transfer learning. Set to False when
            loading a fully-trained checkpoint in export_onnx.py (the weights
            get overwritten anyway).
    """
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)  # 2 classes: [clean, occluded]
    return model


def get_convnext_small_occlusion(pretrained: bool = True) -> nn.Module:
    """Return a ConvNeXt-Small with a 2-class binary classification head."""
    weights = ConvNeXt_Small_Weights.DEFAULT if pretrained else None
    model = convnext_small(weights=weights)
    num_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(num_features, 2)  # 2 classes: [clean, occluded]
    return model


def get_model(model_name: str, pretrained: bool = True) -> nn.Module:
    """Factory: return a classification model by name.

    Args:
        model_name: 'resnet18' or 'convnext_small'.
        pretrained: ImageNet pretrained weights. See get_resnet18_occlusion.

    Raises:
        ValueError: if ``model_name`` is unknown.
    """
    if model_name == "resnet18":
        return get_resnet18_occlusion(pretrained)
    if model_name == "convnext_small":
        return get_convnext_small_occlusion(pretrained)
    raise ValueError(f"Unknown model: {model_name}. Use 'resnet18' or 'convnext_small'.")
