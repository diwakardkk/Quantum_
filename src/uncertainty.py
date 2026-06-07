"""Uncertainty estimation helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn


def softmax_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Convert logits to probabilities."""
    return torch.softmax(logits, dim=1)


def max_softmax_probability(probabilities: np.ndarray) -> np.ndarray:
    """Maximum softmax probability, used as confidence."""
    return probabilities.max(axis=1)


def prediction_entropy(probabilities: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Compute predictive entropy for each sample."""
    return -np.sum(probabilities * np.log(probabilities + eps), axis=1)


def enable_dropout(model: nn.Module) -> None:
    """Turn on dropout layers while leaving other layers in eval mode."""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


@torch.no_grad()
def mc_dropout_inference(
    model: nn.Module,
    dataloader,
    device: torch.device,
    samples: int = 20,
) -> dict[str, Any]:
    """Run Monte Carlo dropout inference.

    This repeats prediction with dropout active and averages the probabilities.
    """
    model.eval()
    all_sample_probs: list[np.ndarray] = []
    labels_np: np.ndarray | None = None
    paths: list[str] = []

    for sample_idx in range(samples):
        model.eval()
        enable_dropout(model)
        probs_this_sample: list[np.ndarray] = []
        labels_this_sample: list[np.ndarray] = []
        paths_this_sample: list[str] = []

        for batch in dataloader:
            images = batch[0].to(device)
            labels = batch[1].cpu().numpy()
            batch_paths = list(batch[2]) if len(batch) > 2 else [""] * len(labels)
            logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            probs_this_sample.append(probs)
            labels_this_sample.append(labels)
            paths_this_sample.extend(batch_paths)

        all_sample_probs.append(np.concatenate(probs_this_sample, axis=0))
        if sample_idx == 0:
            labels_np = np.concatenate(labels_this_sample, axis=0)
            paths = paths_this_sample

    stacked = np.stack(all_sample_probs, axis=0)
    mean_probs = stacked.mean(axis=0)
    variance = stacked.var(axis=0).mean(axis=1)
    entropy = prediction_entropy(mean_probs)

    return {
        "probabilities": mean_probs,
        "variance": variance,
        "entropy": entropy,
        "labels": labels_np,
        "paths": paths,
    }


def uncertainty_histogram_values(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> dict[str, np.ndarray]:
    """Split entropy values into correct and wrong prediction groups."""
    preds = probabilities.argmax(axis=1)
    entropy = prediction_entropy(probabilities)
    correct = preds == labels
    return {
        "correct_entropy": entropy[correct],
        "wrong_entropy": entropy[~correct],
    }


def rejection_curve(
    probabilities: np.ndarray,
    labels: np.ndarray,
    thresholds: list[float],
) -> list[dict[str, float]]:
    """Evaluate accuracy after rejecting low-confidence predictions."""
    confidences = max_softmax_probability(probabilities)
    preds = probabilities.argmax(axis=1)
    rows: list[dict[str, float]] = []
    total = len(labels)

    for threshold in thresholds:
        keep = confidences >= threshold
        kept = int(keep.sum())
        rejected = total - kept
        accuracy = float((preds[keep] == labels[keep]).mean()) if kept > 0 else float("nan")
        rows.append(
            {
                "threshold": float(threshold),
                "coverage": float(kept / total) if total else 0.0,
                "rejection_rate": float(rejected / total) if total else 0.0,
                "accuracy_on_retained": accuracy,
                "retained_samples": kept,
                "rejected_samples": rejected,
            }
        )
    return rows
