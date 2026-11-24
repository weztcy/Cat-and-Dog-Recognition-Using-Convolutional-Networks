"""
Neural Network Architectures for Cat & Dog Image Classification.
Includes Custom Baseline CNN and Pretrained MobileNetV2 with Transfer Learning and Fine-Tuning support.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class CustomBaselineCNN(nn.Module):
    """
    Lightweight Custom Convolutional Neural Network baseline.
    4 Convolutional blocks with Batch Normalization, ReLU, and MaxPooling,
    followed by Global Average Pooling, Dropout, and Dense projection.
    """
    def __init__(self, dropout_rate: float = 0.4):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 224x224 -> 112x112
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 112x112 -> 56x56
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 56x56 -> 28x28
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4: 28x28 -> 14x14
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(64, 1)  # Single logit output for binary classification
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        pooled = self.gap(feat)
        out = self.classifier(pooled)
        return out


def build_mobilenet_v2(pretrained: bool = True, freeze_features: bool = True, dropout_rate: float = 0.3) -> nn.Module:
    """
    Constructs MobileNetV2 for binary classification.
    Optionally freezes feature extraction backbone.
    """
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)

    if freeze_features:
        for param in model.features.parameters():
            param.requires_grad = False

    # Replace classifier head for binary classification
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(in_features, 1)
    )

    return model


def unfreeze_mobilenet_top_layers(model: nn.Module, start_layer_index: int = 14):
    """
    Unfreezes top layers of MobileNetV2 features for fine-tuning.
    Default starts from block 14 to 18 (final layers).
    """
    for i, block in enumerate(model.features):
        if i >= start_layer_index:
            for param in block.parameters():
                param.requires_grad = True
        else:
            for param in block.parameters():
                param.requires_grad = False

    # Ensure classifier is trainable
    for param in model.classifier.parameters():
        param.requires_grad = True
