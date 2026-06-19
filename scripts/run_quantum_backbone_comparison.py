"""Train stronger proposed quantum-head models for paper comparison.

This script runs four proposed variants:

1. DenseNet-121 + quantum head, 8 qubits, 3 quantum layers
2. DenseNet-121 + quantum head, 8 qubits, 4 quantum layers
3. ResNet-50 + quantum head, 8 qubits, 3 quantum layers
4. ResNet-50 + quantum head, 8 qubits, 4 quantum layers

It reuses the existing training/evaluation pipeline, so the generated metrics are
directly comparable with the earlier UQ-HyQNet and pretrained-baseline runs.

Example:
    python3 scripts/run_quantum_backbone_comparison.py --config config_200epoch_gpu.yaml
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate import evaluate_checkpoint
from src.train import train_model
from src.utils import load_config, resolve_path, save_json


REFERENCE_ROWS = [
    {
        "model_name": "classical_cnn",
        "model": "Classical CNN",
        "accuracy": 88.26,
        "precision_macro": 88.17,
        "recall_macro": 88.99,
        "f1_macro": 88.51,
        "roc_auc_macro": 97.44,
        "brier_score": 18.90,
        "expected_calibration_error": 6.37,
    },
    {
        "model_name": "mobilenet_mlp",
        "model": "MobileNet MLP",
        "accuracy": 94.35,
        "precision_macro": 94.21,
        "recall_macro": 95.11,
        "f1_macro": 94.57,
        "roc_auc_macro": 99.40,
        "brier_score": 10.47,
        "expected_calibration_error": 4.84,
    },
    {
        "model_name": "hybrid_quantum",
        "model": "UQ-HyQNet",
        "accuracy": 95.05,
        "precision_macro": 94.93,
        "recall_macro": 95.51,
        "f1_macro": 95.17,
        "roc_auc_macro": 98.73,
        "brier_score": 8.45,
        "expected_calibration_error": 2.97,
    },
    {
        "model_name": "vgg16",
        "model": "VGG-16",
        "accuracy": 96.23,
        "precision_macro": 96.20,
        "recall_macro": 96.57,
        "f1_macro": 96.35,
        "roc_auc_macro": 99.50,
        "brier_score": 7.00,
        "expected_calibration_error": 3.25,
    },
    {
        "model_name": "resnet18",
        "model": "ResNet-18",
        "accuracy": 97.74,
        "precision_macro": 97.69,
        "recall_macro": 97.92,
        "f1_macro": 97.80,
        "roc_auc_macro": 99.77,
        "brier_score": 3.98,
        "expected_calibration_error": 1.93,
    },
    {
        "model_name": "resnet50",
        "model": "ResNet-50",
        "accuracy": 97.90,
        "precision_macro": 97.85,
        "recall_macro": 98.14,
        "f1_macro": 97.99,
        "roc_auc_macro": 99.55,
        "brier_score": 4.02,
        "expected_calibration_error": 1.91,
    },
    {
        "model_name": "densenet121",
        "model": "DenseNet-121",
        "accuracy": 98.17,
        "precision_macro": 98.04,
        "recall_macro": 98.41,
        "f1_macro": 98.21,
        "roc_auc_macro": 99.72,
        "brier_score": 3.49,
        "expected_calibration_error": 1.67,
    },
    {
        "model_name": "efficientnet_b0",
        "model": "EfficientNet-B0",
        "accuracy": 97.85,
        "precision_macro": 97.79,
        "recall_macro": 98.05,
        "f1_macro": 97.91,
        "roc_auc_macro": 99.77,
        "brier_score": 3.70,
        "expected_calibration_error": 1.78,
    },
]

METRIC_COLUMNS = [
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "roc_auc_macro",
    "brier_score",
    "expected_calibration_error",
]

VARIANTS = [
    {
        "model_name": "densenet121_quantum",
        "model": "DenseNet-121 + Quantum Head",
        "backbone": "densenet121",
        "n_qubits": 8,
        "n_quantum_layers": 3,
    },
    {
        "model_name": "densenet121_quantum",
        "model": "DenseNet-121 + Quantum Head",
        "backbone": "densenet121",
        "n_qubits": 8,
        "n_quantum_layers": 4,
    },
    {
        "model_name": "resnet50_quantum",
        "model": "ResNet-50 + Quantum Head",
        "backbone": "resnet50",
        "n_qubits": 8,
        "n_quantum_layers": 3,
    },
    {
        "model_name": "resnet50_quantum",
        "model": "ResNet-50 + Quantum Head",
        "backbone": "resnet50",
        "n_qubits": 8,
        "n_quantum_layers": 4,
    },
]


def _percent(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return value * 100.0


def _read_rejection_at(run_dir: Path, threshold: float) -> dict[str, float | None]:
    path = run_dir / "logs" / "rejection_curve.csv"
    if not path.exists():
        return {
            "coverage_at_threshold": None,
            "retained_accuracy_at_threshold": None,
            "rejection_rate_at_threshold": None,
        }
    rows = list(csv.DictReader(path.open()))
    if not rows:
        return {
            "coverage_at_threshold": None,
            "retained_accuracy_at_threshold": None,
            "rejection_rate_at_threshold": None,
        }
    selected = min(rows, key=lambda row: abs(float(row["threshold"]) - threshold))
    return {
        "coverage_at_threshold": _percent(selected["coverage"]),
        "retained_accuracy_at_threshold": _percent(selected["accuracy_on_retained"]),
        "rejection_rate_at_threshold": _percent(selected["rejection_rate"]),
    }


def _row_from_metrics(
    variant: dict[str, Any],
    run_dir: Path,
    metrics: dict[str, Any],
    rejection_threshold: float,
) -> dict[str, Any]:
    mc_metrics = metrics.get("mc_dropout", {}) or {}
    row = {
        "model_name": variant["model_name"],
        "model": variant["model"],
        "variant": f"{variant['backbone']}_q{variant['n_qubits']}_l{variant['n_quantum_layers']}",
        "backbone": variant["backbone"],
        "n_qubits": variant["n_qubits"],
        "n_quantum_layers": variant["n_quantum_layers"],
        "run_dir": str(run_dir),
        "accuracy": _percent(metrics.get("accuracy")),
        "precision_macro": _percent(metrics.get("precision_macro")),
        "recall_macro": _percent(metrics.get("recall_macro")),
        "f1_macro": _percent(metrics.get("f1_macro")),
        "roc_auc_macro": _percent(metrics.get("roc_auc_macro")),
        "brier_score": _percent(metrics.get("brier_score")),
        "expected_calibration_error": _percent(metrics.get("expected_calibration_error")),
        "temperature": metrics.get("temperature"),
        "mc_accuracy": _percent(mc_metrics.get("accuracy")),
        "mc_f1_macro": _percent(mc_metrics.get("f1_macro")),
        "mc_brier_score": _percent(mc_metrics.get("brier_score")),
        "mc_expected_calibration_error": _percent(mc_metrics.get("expected_calibration_error")),
        "parameter_count_total": metrics.get("parameter_count_total"),
        "parameter_count_trainable": metrics.get("parameter_count_trainable"),
        "model_size_mb": metrics.get("model_size_mb"),
        "inference_time_per_image_sec": metrics.get("inference_time_per_image_sec"),
    }
    row.update(_read_rejection_at(run_dir, rejection_threshold))
    return row


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _best_flags(rows: list[dict[str, Any]], columns: list[str], lower_is_better: set[str]) -> dict[str, set[int]]:
    flags: dict[str, set[int]] = {}
    for column in columns:
        values = [
            (idx, float(row[column]))
            for idx, row in enumerate(rows)
            if row.get(column) is not None and math.isfinite(float(row[column]))
        ]
        if not values:
            flags[column] = set()
            continue
        target = min(value for _, value in values) if column in lower_is_better else max(value for _, value in values)
        flags[column] = {idx for idx, value in values if abs(value - target) < 1e-9}
    return flags


def _format_cell(value: Any, bold: bool = False) -> str:
    if value is None:
        return "--"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(value):
        return "--"
    text = f"{value:.2f}"
    return rf"\textbf{{{text}}}" if bold else text


def _latex_escape(text: str) -> str:
    return text.replace("_", r"\_")


def write_classification_table(rows: list[dict[str, Any]], path: Path, caption: str, label: str) -> None:
    flags = _best_flags(
        rows,
        columns=METRIC_COLUMNS,
        lower_is_better={"brier_score", "expected_calibration_error"},
    )
    lines = [
        r"\begin{table*}[width=\textwidth,cols=8,pos=t]",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\scriptsize",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular*}{\tblwidth}{@{\extracolsep{\fill}} l c c c c c c c @{}}",
        r"\toprule",
        r"\textbf{Model} &",
        r"\shortstack{\textbf{Accuracy}\\\textbf{(\%)}} &",
        r"\shortstack{\textbf{Precision}\\\textbf{Macro (\%)}} &",
        r"\shortstack{\textbf{Recall}\\\textbf{Macro (\%)}} &",
        r"\shortstack{\textbf{F1}\\\textbf{Macro (\%)}} &",
        r"\shortstack{\textbf{ROC-AUC}\\\textbf{Macro (\%)}} &",
        r"\shortstack{\textbf{Brier}\\\textbf{Score (\%)}} &",
        r"\shortstack{\textbf{ECE}\\\textbf{(\%)}} \\",
        r"\midrule",
    ]
    for idx, row in enumerate(rows):
        cells = [_format_cell(row.get(column), idx in flags[column]) for column in METRIC_COLUMNS]
        model = rf"\textbf{{{_latex_escape(str(row['model']))}}}"
        if row.get("n_qubits") and row.get("n_quantum_layers"):
            model += rf" ({int(row['n_qubits'])}q, {int(row['n_quantum_layers'])}L)"
        lines.append(f"{model} & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular*}", r"\end{table*}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_uncertainty_table(rows: list[dict[str, Any]], path: Path, caption: str, label: str) -> None:
    columns = [
        "brier_score",
        "expected_calibration_error",
        "mc_brier_score",
        "mc_expected_calibration_error",
        "coverage_at_threshold",
        "retained_accuracy_at_threshold",
        "rejection_rate_at_threshold",
    ]
    flags = _best_flags(
        rows,
        columns=columns,
        lower_is_better={
            "brier_score",
            "expected_calibration_error",
            "mc_brier_score",
            "mc_expected_calibration_error",
            "rejection_rate_at_threshold",
        },
    )
    lines = [
        r"\begin{table*}[width=\textwidth,cols=8,pos=t]",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\scriptsize",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular*}{\tblwidth}{@{\extracolsep{\fill}} l c c c c c c c @{}}",
        r"\toprule",
        r"\textbf{Model} &",
        r"\shortstack{\textbf{Brier}\\\textbf{(\%)}} &",
        r"\shortstack{\textbf{ECE}\\\textbf{(\%)}} &",
        r"\shortstack{\textbf{MC Brier}\\\textbf{(\%)}} &",
        r"\shortstack{\textbf{MC ECE}\\\textbf{(\%)}} &",
        r"\shortstack{\textbf{Coverage}\\\textbf{@0.90 (\%)}} &",
        r"\shortstack{\textbf{Retained Acc.}\\\textbf{@0.90 (\%)}} &",
        r"\shortstack{\textbf{Rejected}\\\textbf{@0.90 (\%)}} \\",
        r"\midrule",
    ]
    for idx, row in enumerate(rows):
        cells = [_format_cell(row.get(column), idx in flags[column]) for column in columns]
        model = rf"\textbf{{{_latex_escape(str(row['model']))}}}"
        if row.get("n_qubits") and row.get("n_quantum_layers"):
            model += rf" ({int(row['n_qubits'])}q, {int(row['n_quantum_layers'])}L)"
        lines.append(f"{model} & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular*}", r"\end{table*}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DenseNet/ResNet quantum-head proposed models.")
    parser.add_argument("--config", default="config_200epoch_gpu.yaml", help="Path to config YAML.")
    parser.add_argument(
        "--output-dir",
        default="outputs_quantum_backbone_comparison",
        help="Folder for the quantum-backbone experiment.",
    )
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs. Default: 200.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size if needed.")
    parser.add_argument("--mc-samples", type=int, default=None, help="Override MC dropout samples.")
    parser.add_argument("--no-gradcam", action="store_true", help="Skip Grad-CAM during evaluation.")
    parser.add_argument(
        "--no-reference-rows",
        action="store_true",
        help="Write final classification table only for the newly trained proposed variants.",
    )
    args = parser.parse_args()

    base_config = load_config(args.config)
    output_base = resolve_path(args.output_dir, base_config)
    experiment_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_quantum_backbone_comparison"
    experiment_dir = output_base / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Quantum-backbone proposed model comparison")
    print(f"Config: {args.config}")
    print(f"Epochs: {args.epochs}")
    print(f"Experiment folder: {experiment_dir}")
    print("=" * 80)

    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        variant_name = f"{variant['backbone']}_quantum_q{variant['n_qubits']}_l{variant['n_quantum_layers']}"
        print("\n" + "#" * 80)
        print(f"Training proposed variant: {variant_name}")
        print("#" * 80)

        config = dict(base_config)
        config["model_name"] = variant["model_name"]
        config["n_qubits"] = variant["n_qubits"]
        config["n_quantum_layers"] = variant["n_quantum_layers"]
        config["epochs"] = args.epochs
        config["early_stopping_patience"] = args.epochs
        config["output_dir"] = str(experiment_dir)
        config["use_class_weights"] = True
        config["freeze_backbone"] = True
        config["unfreeze_last_n_blocks"] = int(config.get("unfreeze_last_n_blocks", 5))
        if args.batch_size is not None:
            config["batch_size"] = args.batch_size
        if args.mc_samples is not None:
            config["mc_dropout_samples"] = args.mc_samples

        train_result = train_model(config, fixed_run_name=variant_name)
        eval_result = evaluate_checkpoint(
            config=config,
            checkpoint_path=train_result["best_model_path"],
            run_dir=train_result["run_dir"],
            make_gradcam=not args.no_gradcam,
        )
        row = _row_from_metrics(
            variant=variant,
            run_dir=Path(eval_result["run_dir"]),
            metrics=eval_result["metrics"],
            rejection_threshold=0.9,
        )
        rows.append(row)
        _write_csv(rows, experiment_dir / "quantum_backbone_results_partial.csv")

    _write_csv(rows, experiment_dir / "quantum_backbone_results.csv")
    save_json({"results": rows}, experiment_dir / "quantum_backbone_results.json")

    proposed_table_rows = rows
    final_table_rows = proposed_table_rows if args.no_reference_rows else [*REFERENCE_ROWS, *proposed_table_rows]

    write_classification_table(
        rows=proposed_table_rows,
        path=experiment_dir / "proposed_quantum_backbone_classification_table.tex",
        caption="Test performance of proposed quantum-head backbone variants.",
        label="tab:proposed_quantum_backbone_results",
    )
    write_classification_table(
        rows=final_table_rows,
        path=experiment_dir / "final_main_results_with_quantum_backbones.tex",
        caption="Test performance comparison with pretrained and proposed quantum-head models.",
        label="tab:final_main_results_quantum_backbones",
    )
    write_uncertainty_table(
        rows=proposed_table_rows,
        path=experiment_dir / "proposed_quantum_backbone_uncertainty_table.tex",
        caption="Uncertainty and rejection-based triage results for proposed quantum-head variants.",
        label="tab:proposed_quantum_backbone_uncertainty",
    )

    print("\n" + "=" * 80)
    print("Finished quantum-backbone comparison.")
    print(f"CSV: {experiment_dir / 'quantum_backbone_results.csv'}")
    print(f"JSON: {experiment_dir / 'quantum_backbone_results.json'}")
    print(f"Proposed classification table: {experiment_dir / 'proposed_quantum_backbone_classification_table.tex'}")
    print(f"Final main table: {experiment_dir / 'final_main_results_with_quantum_backbones.tex'}")
    print(f"Uncertainty table: {experiment_dir / 'proposed_quantum_backbone_uncertainty_table.tex'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
