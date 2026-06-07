"""Calibration metrics and temperature scaling."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "uq_hyqnet_matplotlib"),
)

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Compute Expected Calibration Error (ECE)."""
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    accuracies = predictions == labels
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (confidences > lower) & (confidences <= upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            confidence_in_bin = confidences[in_bin].mean()
            ece += abs(confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return float(ece)


def brier_score_multiclass(
    probabilities: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
) -> float:
    """Compute multiclass Brier score."""
    one_hot = np.eye(num_classes)[labels]
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def reliability_curve_data(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> dict[str, list[float]]:
    """Return bin accuracy/confidence values for a reliability diagram."""
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    accuracies = predictions == labels
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    bin_centers: list[float] = []
    bin_accuracies: list[float] = []
    bin_confidences: list[float] = []
    bin_counts: list[int] = []

    for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (confidences > lower) & (confidences <= upper)
        bin_centers.append(float((lower + upper) / 2.0))
        bin_counts.append(int(in_bin.sum()))
        if in_bin.any():
            bin_accuracies.append(float(accuracies[in_bin].mean()))
            bin_confidences.append(float(confidences[in_bin].mean()))
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append(0.0)

    return {
        "bin_centers": bin_centers,
        "bin_accuracies": bin_accuracies,
        "bin_confidences": bin_confidences,
        "bin_counts": bin_counts,
    }


def save_reliability_diagram(
    probabilities: np.ndarray,
    labels: np.ndarray,
    output_path: str | Path,
    n_bins: int = 15,
) -> None:
    """Save a reliability diagram plot."""
    data = reliability_curve_data(probabilities, labels, n_bins=n_bins)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    plt.bar(
        data["bin_centers"],
        data["bin_accuracies"],
        width=1.0 / n_bins,
        alpha=0.7,
        edgecolor="black",
        label="Observed accuracy",
    )
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


class TemperatureScaler(nn.Module):
    """Learn one positive temperature value for validation-set calibration."""

    def __init__(self):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(1))

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temperature).clamp(min=1e-4, max=100.0)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature


def fit_temperature_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    max_iter: int = 50,
) -> TemperatureScaler:
    """Fit temperature scaling on validation logits."""
    scaler = TemperatureScaler().to(logits.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS([scaler.log_temperature], lr=0.01, max_iter=max_iter)

    logits = logits.detach()
    labels = labels.detach()

    def closure():
        optimizer.zero_grad()
        loss = criterion(scaler(logits), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return scaler


@torch.no_grad()
def collect_logits(model: nn.Module, dataloader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect logits and labels from a dataloader."""
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    for batch in dataloader:
        images, labels = batch[0].to(device), batch[1].to(device)
        logits = model(images)
        all_logits.append(logits.detach())
        all_labels.append(labels.detach())
    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)


def fit_temperature(
    model: nn.Module,
    val_loader,
    device: torch.device,
) -> TemperatureScaler:
    """Fit temperature scaling using a model and validation loader."""
    logits, labels = collect_logits(model, val_loader, device)
    return fit_temperature_from_logits(logits, labels)
