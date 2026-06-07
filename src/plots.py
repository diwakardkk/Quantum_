"""Plotting utilities."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

# Make plotting work cleanly on servers where the home directory is not writable.
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "uq_hyqnet_matplotlib"),
)

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize

from .calibration import reliability_curve_data
from .uncertainty import uncertainty_histogram_values


def plot_training_curves(history: list[dict], output_path: str | Path) -> None:
    """Save training/validation loss and accuracy curves."""
    if not history:
        return
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, [row["train_loss"] for row in history], label="train")
    plt.plot(epochs, [row["val_loss"] for row in history], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, [row["train_acc"] for row in history], label="train")
    plt.plot(epochs, [row["val_acc"] for row in history], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: list[str],
    output_path: str | Path,
) -> None:
    """Save a confusion matrix heatmap."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 6))
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)

    threshold = matrix.max() / 2.0 if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > threshold else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_roc_curve(
    labels: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
    output_path: str | Path,
) -> bool:
    """Save ROC curve. Returns False if ROC cannot be computed."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        plt.figure(figsize=(7, 6))
        if len(class_names) == 2:
            fpr, tpr, _ = roc_curve(labels, probabilities[:, 1])
            plt.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.3f}")
        else:
            y_bin = label_binarize(labels, classes=list(range(len(class_names))))
            for idx, class_name in enumerate(class_names):
                fpr, tpr, _ = roc_curve(y_bin[:, idx], probabilities[:, idx])
                plt.plot(fpr, tpr, label=f"{class_name} AUC = {auc(fpr, tpr):.3f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
        return True
    except Exception as exc:
        print(f"Warning: ROC curve could not be plotted: {exc}")
        plt.close()
        return False


def plot_reliability_diagram(
    probabilities: np.ndarray,
    labels: np.ndarray,
    output_path: str | Path,
    n_bins: int = 15,
) -> None:
    """Save a reliability diagram."""
    data = reliability_curve_data(probabilities, labels, n_bins=n_bins)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect")
    plt.bar(
        data["bin_centers"],
        data["bin_accuracies"],
        width=1.0 / n_bins,
        alpha=0.75,
        edgecolor="black",
        label="Accuracy",
    )
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.title("Reliability Diagram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_uncertainty_histogram(
    probabilities: np.ndarray,
    labels: np.ndarray,
    output_path: str | Path,
) -> None:
    """Save entropy histogram for correct vs wrong predictions."""
    values = uncertainty_histogram_values(probabilities, labels)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    correct_entropy = np.asarray(values["correct_entropy"], dtype=float)
    wrong_entropy = np.asarray(values["wrong_entropy"], dtype=float)
    correct_entropy = correct_entropy[np.isfinite(correct_entropy)]
    wrong_entropy = wrong_entropy[np.isfinite(wrong_entropy)]
    combined = np.concatenate([correct_entropy, wrong_entropy])
    upper = float(combined.max() * 1.05) if combined.size else 1.0
    upper = max(upper, 1.0)
    bins = np.linspace(0.0, upper, 21)

    plt.figure(figsize=(7, 5))
    if correct_entropy.size:
        plt.hist(correct_entropy, bins=bins, alpha=0.7, label="Correct")
    if wrong_entropy.size:
        plt.hist(wrong_entropy, bins=bins, alpha=0.7, label="Wrong")
    if not correct_entropy.size and not wrong_entropy.size:
        plt.text(0.5, 0.5, "No finite entropy values", ha="center", va="center")
    plt.xlabel("Prediction Entropy")
    plt.ylabel("Count")
    plt.title("Uncertainty Histogram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_rejection_curve(rows: list[dict], output_path: str | Path) -> None:
    """Save rejection curve based on confidence thresholds."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    thresholds = [row["threshold"] for row in rows]
    accuracies = [row["accuracy_on_retained"] for row in rows]
    coverages = [row["coverage"] for row in rows]

    plt.figure(figsize=(7, 5))
    plt.plot(thresholds, accuracies, marker="o", label="Accuracy on retained")
    plt.plot(thresholds, coverages, marker="s", label="Coverage")
    plt.xlabel("Confidence Threshold")
    plt.ylabel("Value")
    plt.ylim(0, 1.05)
    plt.title("Rejection Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
