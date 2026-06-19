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


def _load_torchvision_model(builder_name: str, weights_name: str, pretrained: bool = True) -> nn.Module:
    """Load a torchvision model, falling back to random weights if needed."""
    _configure_torch_cache()
    builder = getattr(models, builder_name)
    if not pretrained:
        try:
            return builder(weights=None)
        except TypeError:
            return builder(pretrained=False)

    weights_enum = getattr(models, weights_name, None)
    weights = weights_enum.DEFAULT if weights_enum is not None else None
    try:
        if weights is not None:
            return builder(weights=weights)
        return builder(pretrained=True)
    except Exception as exc:
        print(
            f"Warning: Could not load pretrained {builder_name} weights. "
            f"Using random weights instead. Reason: {exc}"
        )
        try:
            return builder(weights=None)
        except TypeError:
            return builder(pretrained=False)


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


def _configure_transfer_learning(
    model: nn.Module,
    classifier_head: nn.Module,
    trainable_tail_modules: list[nn.Module],
    freeze_backbone: bool,
    unfreeze_last_n_blocks: int = 0,
) -> None:
    """Freeze pretrained baselines while keeping the new head trainable."""
    if not freeze_backbone:
        for param in model.parameters():
            param.requires_grad = True
        return

    for param in model.parameters():
        param.requires_grad = False

    if unfreeze_last_n_blocks > 0:
        for module in trainable_tail_modules[-unfreeze_last_n_blocks:]:
            for param in module.parameters():
                param.requires_grad = True

    for param in classifier_head.parameters():
        param.requires_grad = True


class TorchvisionBaseline(nn.Module):
    """Thin wrapper that exposes a Grad-CAM target layer for pretrained baselines."""

    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None):
        super().__init__()
        self.model = model
        self.target_layer = target_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def get_cam_target_layer(self) -> nn.Module:
        """Return the last convolutional block when the architecture has one."""
        if self.target_layer is None:
            raise ValueError("This architecture does not expose a 2D convolutional Grad-CAM layer.")
        return self.target_layer


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


def _last_conv_layer(module: nn.Module) -> nn.Module | None:
    """Find the last Conv2d module for Grad-CAM."""
    conv_layers = [child for child in module.modules() if isinstance(child, nn.Conv2d)]
    return conv_layers[-1] if conv_layers else None


def _build_resnet_baseline(
    variant: str,
    num_classes: int,
    dropout: float,
    freeze_backbone: bool,
    pretrained: bool,
    unfreeze_last_n_blocks: int,
) -> nn.Module:
    weights_name = "ResNet18_Weights" if variant == "resnet18" else "ResNet50_Weights"
    base = _load_torchvision_model(variant, weights_name, pretrained=pretrained)
    in_features = base.fc.in_features
    base.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    _configure_transfer_learning(
        base,
        classifier_head=base.fc,
        trainable_tail_modules=[base.layer1, base.layer2, base.layer3, base.layer4],
        freeze_backbone=freeze_backbone,
        unfreeze_last_n_blocks=unfreeze_last_n_blocks,
    )
    return TorchvisionBaseline(base, target_layer=base.layer4[-1])


def _build_vgg16_baseline(
    num_classes: int,
    dropout: float,
    freeze_backbone: bool,
    pretrained: bool,
    unfreeze_last_n_blocks: int,
) -> nn.Module:
    base = _load_torchvision_model("vgg16", "VGG16_Weights", pretrained=pretrained)
    for module in base.classifier.modules():
        if isinstance(module, nn.Dropout):
            module.p = dropout
    in_features = base.classifier[-1].in_features
    base.classifier[-1] = nn.Linear(in_features, num_classes)
    _configure_transfer_learning(
        base,
        classifier_head=base.classifier[-1],
        trainable_tail_modules=list(base.features.children()),
        freeze_backbone=freeze_backbone,
        unfreeze_last_n_blocks=unfreeze_last_n_blocks,
    )
    return TorchvisionBaseline(base, target_layer=_last_conv_layer(base.features))


def _build_densenet121_baseline(
    num_classes: int,
    dropout: float,
    freeze_backbone: bool,
    pretrained: bool,
    unfreeze_last_n_blocks: int,
) -> nn.Module:
    base = _load_torchvision_model("densenet121", "DenseNet121_Weights", pretrained=pretrained)
    in_features = base.classifier.in_features
    base.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    _configure_transfer_learning(
        base,
        classifier_head=base.classifier,
        trainable_tail_modules=[
            base.features.denseblock1,
            base.features.transition1,
            base.features.denseblock2,
            base.features.transition2,
            base.features.denseblock3,
            base.features.transition3,
            base.features.denseblock4,
        ],
        freeze_backbone=freeze_backbone,
        unfreeze_last_n_blocks=unfreeze_last_n_blocks,
    )
    return TorchvisionBaseline(base, target_layer=base.features.denseblock4)


def _build_efficientnet_b0_baseline(
    num_classes: int,
    dropout: float,
    freeze_backbone: bool,
    pretrained: bool,
    unfreeze_last_n_blocks: int,
) -> nn.Module:
    base = _load_torchvision_model("efficientnet_b0", "EfficientNet_B0_Weights", pretrained=pretrained)
    in_features = base.classifier[-1].in_features
    base.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    _configure_transfer_learning(
        base,
        classifier_head=base.classifier,
        trainable_tail_modules=list(base.features.children()),
        freeze_backbone=freeze_backbone,
        unfreeze_last_n_blocks=unfreeze_last_n_blocks,
    )
    return TorchvisionBaseline(base, target_layer=base.features[-1])


def _build_vit_b_16_baseline(
    num_classes: int,
    freeze_backbone: bool,
    pretrained: bool,
    unfreeze_last_n_blocks: int,
) -> nn.Module:
    base = _load_torchvision_model("vit_b_16", "ViT_B_16_Weights", pretrained=pretrained)
    in_features = base.heads.head.in_features
    base.heads.head = nn.Linear(in_features, num_classes)
    tail_modules = list(base.encoder.layers.children())
    _configure_transfer_learning(
        base,
        classifier_head=base.heads,
        trainable_tail_modules=tail_modules,
        freeze_backbone=freeze_backbone,
        unfreeze_last_n_blocks=unfreeze_last_n_blocks,
    )
    return TorchvisionBaseline(base, target_layer=None)


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
    if model_name in {"resnet18", "resnet50"}:
        return _build_resnet_baseline(
            variant=model_name,
            num_classes=num_classes,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
            pretrained=pretrained,
            unfreeze_last_n_blocks=unfreeze_last_n_blocks,
        )
    if model_name == "vgg16":
        return _build_vgg16_baseline(
            num_classes=num_classes,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
            pretrained=pretrained,
            unfreeze_last_n_blocks=unfreeze_last_n_blocks,
        )
    if model_name == "densenet121":
        return _build_densenet121_baseline(
            num_classes=num_classes,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
            pretrained=pretrained,
            unfreeze_last_n_blocks=unfreeze_last_n_blocks,
        )
    if model_name == "efficientnet_b0":
        return _build_efficientnet_b0_baseline(
            num_classes=num_classes,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
            pretrained=pretrained,
            unfreeze_last_n_blocks=unfreeze_last_n_blocks,
        )
    if model_name == "vit_b_16":
        return _build_vit_b_16_baseline(
            num_classes=num_classes,
            freeze_backbone=freeze_backbone,
            pretrained=pretrained,
            unfreeze_last_n_blocks=unfreeze_last_n_blocks,
        )

    raise ValueError(
        f"Unknown model_name '{model_name}'. "
        "Choose from: classical_cnn, mobilenet_mlp, hybrid_quantum, vgg16, "
        "resnet18, resnet50, densenet121, efficientnet_b0, vit_b_16."
    )
