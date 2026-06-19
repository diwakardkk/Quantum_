"""Train extra pretrained baselines and write paper-ready comparison tables.

Example:
    python3 scripts/run_pretrained_comparison.py --config config_200epoch_gpu.yaml

The script reuses the same dataset split, training loop, evaluation metrics,
calibration, uncertainty, and plotting code as the proposed UQ-HyQNet workflow.
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


DEFAULT_BASELINES = ["vgg16", "resnet18", "resnet50", "densenet121", "efficientnet_b0"]

MODEL_LABELS = {
    "classical_cnn": "Classical CNN",
    "mobilenet_mlp": "MobileNet MLP",
    "hybrid_quantum": "UQ-HyQNet",
    "vgg16": "VGG-16",
    "resnet18": "ResNet-18",
    "resnet50": "ResNet-50",
    "densenet121": "DenseNet-121",
    "efficientnet_b0": "EfficientNet-B0",
    "vit_b_16": "ViT-B/16",
}

EXISTING_MAIN_RESULTS = [
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


def _percent(value: Any) -> float | None:
    """Convert a metric in [0, 1] to percentage scale for the paper table."""
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return value * 100.0


def _row_from_metrics(model_name: str, run_dir: Path, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "model": MODEL_LABELS.get(model_name, model_name),
        "run_dir": str(run_dir),
        "accuracy": _percent(metrics.get("accuracy")),
        "precision_macro": _percent(metrics.get("precision_macro")),
        "recall_macro": _percent(metrics.get("recall_macro")),
        "f1_macro": _percent(metrics.get("f1_macro")),
        "roc_auc_macro": _percent(metrics.get("roc_auc_macro")),
        "brier_score": _percent(metrics.get("brier_score")),
        "expected_calibration_error": _percent(metrics.get("expected_calibration_error")),
    }


def _write_results_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model_name", "model", "run_dir", *METRIC_COLUMNS]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _best_flags(rows: list[dict[str, Any]]) -> dict[str, set[int]]:
    flags: dict[str, set[int]] = {}
    for metric in METRIC_COLUMNS:
        indexed_values = [
            (idx, float(row[metric]))
            for idx, row in enumerate(rows)
            if row.get(metric) is not None and math.isfinite(float(row[metric]))
        ]
        if not indexed_values:
            flags[metric] = set()
            continue
        values = [value for _, value in indexed_values]
        target = min(values) if metric in {"brier_score", "expected_calibration_error"} else max(values)
        flags[metric] = {idx for idx, value in indexed_values if abs(value - target) < 1e-9}
    return flags


def _format_cell(value: Any, bold: bool) -> str:
    if value is None:
        return "--"
    value = float(value)
    if not math.isfinite(value):
        return "--"
    text = f"{value:.2f}"
    return rf"\textbf{{{text}}}" if bold else text


def _latex_escape(text: str) -> str:
    return text.replace("_", r"\_")


def write_latex_table(rows: list[dict[str, Any]], path: Path, caption: str, label: str) -> None:
    flags = _best_flags(rows)
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
        metric_cells = [
            _format_cell(row.get(metric), idx in flags[metric])
            for metric in METRIC_COLUMNS
        ]
        model = rf"\textbf{{{_latex_escape(str(row['model']))}}}"
        lines.append(f"{model} & " + " & ".join(metric_cells) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular*}",
            r"\end{table*}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run pretrained CNN/ViT baselines and create comparison tables."
    )
    parser.add_argument("--config", default="config_200epoch_gpu.yaml", help="Path to config YAML")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        choices=[*DEFAULT_BASELINES, "vit_b_16"],
        help="Baselines to train. Default: VGG-16, ResNet-18, ResNet-50, DenseNet-121, EfficientNet-B0.",
    )
    parser.add_argument("--include-vit", action="store_true", help="Also train ViT-B/16.")
    parser.add_argument(
        "--output-dir",
        default="outputs_pretrained_comparison",
        help="Folder for this comparison experiment.",
    )
    parser.add_argument("--no-gradcam", action="store_true", help="Skip Grad-CAM during evaluation.")
    parser.add_argument(
        "--no-existing-rows",
        action="store_true",
        help="Do not prepend the three already-reported rows to the combined LaTeX table.",
    )
    args = parser.parse_args()

    base_config = load_config(args.config)
    models_to_run = list(args.models or DEFAULT_BASELINES)
    if args.include_vit and "vit_b_16" not in models_to_run:
        models_to_run.append("vit_b_16")

    output_base = resolve_path(args.output_dir, base_config)
    experiment_dir = output_base / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_pretrained_baselines"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Pretrained baseline comparison")
    print(f"Config: {args.config}")
    print(f"Models: {models_to_run}")
    print(f"Experiment folder: {experiment_dir}")
    print("=" * 80)

    rows: list[dict[str, Any]] = []
    for model_name in models_to_run:
        print("\n" + "#" * 80)
        print(f"Training pretrained baseline: {MODEL_LABELS.get(model_name, model_name)}")
        print("#" * 80)

        config = dict(base_config)
        config["model_name"] = model_name
        config["output_dir"] = str(experiment_dir)

        train_result = train_model(config)
        eval_result = evaluate_checkpoint(
            config=config,
            checkpoint_path=train_result["best_model_path"],
            run_dir=train_result["run_dir"],
            make_gradcam=not args.no_gradcam,
        )
        rows.append(
            _row_from_metrics(
                model_name=model_name,
                run_dir=Path(eval_result["run_dir"]),
                metrics=eval_result["metrics"],
            )
        )

        _write_results_csv(rows, experiment_dir / "pretrained_baseline_results_partial.csv")

    _write_results_csv(rows, experiment_dir / "pretrained_baseline_results.csv")
    save_json({"results": rows}, experiment_dir / "pretrained_baseline_results.json")

    write_latex_table(
        rows=rows,
        path=experiment_dir / "pretrained_baselines_table.tex",
        caption="Test performance comparison of additional pretrained baselines.",
        label="tab:pretrained_baselines",
    )

    combined_rows = rows if args.no_existing_rows else [*EXISTING_MAIN_RESULTS, *rows]
    write_latex_table(
        rows=combined_rows,
        path=experiment_dir / "main_results_with_pretrained_baselines.tex",
        caption="Test performance comparison with classical, pretrained, and proposed models.",
        label="tab:main_results_extended",
    )

    print("\n" + "=" * 80)
    print("Finished pretrained comparison.")
    print(f"CSV: {experiment_dir / 'pretrained_baseline_results.csv'}")
    print(f"JSON: {experiment_dir / 'pretrained_baseline_results.json'}")
    print(f"LaTeX pretrained table: {experiment_dir / 'pretrained_baselines_table.tex'}")
    print(f"LaTeX combined table: {experiment_dir / 'main_results_with_pretrained_baselines.tex'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
