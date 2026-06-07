"""Model definitions for UQ-HyQNet."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision import models

from .quantum_layers import PennyLaneQuantumLayer


def _configure_torch_cache() -> None:
    """Use a project-local model cache when TORCH_HOME is not already set."""
    if os.environ.get("TORCH_HOME"):
        return
    cache_root = Path(__file__).resolve().parents[1] / ".cache" / "torch"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(cache_root)
    torch.hub.set_dir(str(cache_root / "hub"))


def _load_mobilenet(pretrained: bool = True) -> models.MobileNetV2:
    """Load MobileNetV2, falling back to random weights if pretrained download fails."""
    _configure_torch_cache()
    if not pretrained:
        return models.mobilenet_v2(weights=None)
    try:
        weights = models.MobileNet_V2_Weights.DEFAULT
        return models.mobilenet_v2(weights=weights)
    except Exception as exc:
        print(
            "Warning: Could not load pretrained MobileNetV2 weights. "
            f"Using random weights instead. Reason: {exc}"
        )
        return models.mobilenet_v2(weights=None)


def _configure_backbone_training(
    features: nn.Sequential,
    freeze_backbone: bool,
    unfreeze_last_n_blocks: int = 0,
) -> None:
    """Freeze MobileNet features, optionally unfreezing the last few blocks.

    This is useful for MRI transfer learning: the early layers stay general, while
    the final blocks can adapt to the medical-image domain.
    """
    if not freeze_backbone:
        for param in features.parameters():
            param.requires_grad = True
        return

    for param in features.parameters():
        param.requires_grad = False

    if unfreeze_last_n_blocks <= 0:
        return

    blocks_to_unfreeze = list(features.children())[-unfreeze_last_n_blocks:]
    for block in blocks_to_unfreeze:
        for param in block.parameters():
            param.requires_grad = True


class SmallClassicalCNN(nn.Module):
    """A small CNN baseline trained from scratch."""

    def __init__(self, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)

    def get_cam_target_layer(self) -> nn.Module:
        """Last convolutional layer for Grad-CAM."""
        return self.features[12]


class MobileNetMLP(nn.Module):
    """MobileNetV2 feature extractor plus classical MLP classifier."""

    def __init__(
        self,
        num_classes: int,
        dropout: float = 0.3,
        freeze_backbone: bool = True,
        pretrained: bool = True,
        unfreeze_last_n_blocks: int = 0,
    ):
        super().__init__()
        base = _load_mobilenet(pretrained=pretrained)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(base.last_channel, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        _configure_backbone_training(
            self.features,
            freeze_backbone=freeze_backbone,
            unfreeze_last_n_blocks=unfreeze_last_n_blocks,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def get_cam_target_layer(self) -> nn.Module:
        """Last feature block for Grad-CAM."""
        return self.features[-1]


class HybridQuantumCNN(nn.Module):
    """MobileNetV2 features plus a simulated PennyLane variational quantum circuit."""

    def __init__(
        self,
        num_classes: int,
        n_qubits: int = 4,
        n_quantum_layers: int = 2,
        dropout: float = 0.3,
        freeze_backbone: bool = True,
        pretrained: bool = True,
        unfreeze_last_n_blocks: int = 0,
    ):
        super().__init__()
        base = _load_mobilenet(pretrained=pretrained)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.reducer = nn.Linear(base.last_channel, n_qubits)
        self.quantum = PennyLaneQuantumLayer(
            n_qubits=n_qubits,
            n_quantum_layers=n_quantum_layers,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(n_qubits, num_classes)
        self.n_qubits = n_qubits

        _configure_backbone_training(
            self.features,
            freeze_backbone=freeze_backbone,
            unfreeze_last_n_blocks=unfreeze_last_n_blocks,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        # Scale to a useful angle range for AngleEmbedding.
        x = torch.tanh(self.reducer(x)) * math.pi
        x = self.quantum(x)
        x = self.dropout(x)
        return self.classifier(x)

    def get_cam_target_layer(self) -> nn.Module:
        """Last MobileNet feature block for Grad-CAM."""
        return self.features[-1]


def build_model(model_name: str, num_classes: int, config: dict[str, Any]) -> nn.Module:
    """Factory function used by training and evaluation scripts."""
    model_name = str(model_name).lower()
    dropout = float(config.get("dropout", 0.3))
    freeze_backbone = bool(config.get("freeze_backbone", True))
    pretrained = bool(config.get("pretrained_backbone", True))
    unfreeze_last_n_blocks = int(config.get("unfreeze_last_n_blocks", 0))

    if model_name == "classical_cnn":
        return SmallClassicalCNN(num_classes=num_classes, dropout=dropout)
    if model_name == "mobilenet_mlp":
        return MobileNetMLP(
            num_classes=num_classes,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
            pretrained=pretrained,
            unfreeze_last_n_blocks=unfreeze_last_n_blocks,
        )
    if model_name == "hybrid_quantum":
        return HybridQuantumCNN(
            num_classes=num_classes,
            n_qubits=int(config.get("n_qubits", 4)),
            n_quantum_layers=int(config.get("n_quantum_layers", 2)),
            dropout=dropout,
            freeze_backbone=freeze_backbone,
            pretrained=pretrained,
            unfreeze_last_n_blocks=unfreeze_last_n_blocks,
        )

    raise ValueError(
        f"Unknown model_name '{model_name}'. "
        "Choose from: classical_cnn, mobilenet_mlp, hybrid_quantum."
    )
