"""Classification metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    """Compute common classification metrics."""
    labels = list(range(len(class_names)))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
    }

    try:
        if len(class_names) == 2:
            metrics["roc_auc_macro"] = float(roc_auc_score(y_true, y_prob[:, 1]))
        else:
            y_true_bin = label_binarize(y_true, classes=labels)
            metrics["roc_auc_macro"] = float(
                roc_auc_score(y_true_bin, y_prob, average="macro", multi_class="ovr")
            )
    except Exception as exc:
        metrics["roc_auc_macro"] = None
        metrics["roc_auc_warning"] = str(exc)

    return metrics


def flatten_metrics_for_csv(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten top-level scalar metrics for a simple metrics.csv file."""
    rows: list[dict[str, Any]] = []
    for key, value in metrics.items():
        if isinstance(value, (int, float, str)) or value is None:
            rows.append({"metric": key, "value": value})
    return rows

