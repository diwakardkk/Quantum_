"""Train 10-qubit, 4-layer quantum-head variants for final comparison.

This focused runner trains only the two requested variants:

1. DenseNet-121 + quantum head, 10 qubits, 4 quantum layers
2. ResNet-50 + quantum head, 10 qubits, 4 quantum layers

It reuses the same training, evaluation, uncertainty, CSV, JSON, and LaTeX
table generation utilities as ``run_quantum_backbone_comparison.py`` so the
outputs remain directly comparable with previous CNN and quantum-head results.

Example:
    python3 scripts/run_quantum_10q4l_comparison.py --config config_200epoch_gpu.yaml
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_quantum_backbone_comparison import (  # noqa: E402
    REFERENCE_ROWS,
    _row_from_metrics,
    _write_csv,
    write_classification_table,
    write_uncertainty_table,
)
from src.evaluate import evaluate_checkpoint  # noqa: E402
from src.train import train_model  # noqa: E402
from src.utils import load_config, resolve_path, save_json  # noqa: E402


VARIANTS_10Q_4L: list[dict[str, Any]] = [
    {
        "model_name": "densenet121_quantum",
        "model": "DenseNet-121 + Quantum Head",
        "backbone": "densenet121",
        "n_qubits": 10,
        "n_quantum_layers": 4,
    },
    {
        "model_name": "resnet50_quantum",
        "model": "ResNet-50 + Quantum Head",
        "backbone": "resnet50",
        "n_qubits": 10,
        "n_quantum_layers": 4,
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train DenseNet-121 and ResNet-50 quantum heads with 10 qubits and 4 layers."
    )
    parser.add_argument("--config", default="config_200epoch_gpu.yaml", help="Path to config YAML.")
    parser.add_argument(
        "--output-dir",
        default="outputs_quantum_10q4l_comparison",
        help="Folder for this 10-qubit quantum-head experiment.",
    )
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs. Default: 200.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size if needed.")
    parser.add_argument("--mc-samples", type=int, default=None, help="Override MC dropout samples.")
    parser.add_argument("--no-gradcam", action="store_true", help="Skip Grad-CAM during evaluation.")
    parser.add_argument(
        "--no-reference-rows",
        action="store_true",
        help="Write final classification table only for these 10-qubit variants.",
    )
    args = parser.parse_args()

    base_config = load_config(args.config)
    output_base = resolve_path(args.output_dir, base_config)
    experiment_dir = output_base / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_quantum_10q4l_comparison"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("10-qubit / 4-layer quantum-head comparison")
    print(f"Config: {args.config}")
    print(f"Epochs: {args.epochs}")
    print(f"Experiment folder: {experiment_dir}")
    print("=" * 80)

    rows: list[dict[str, Any]] = []
    for variant in VARIANTS_10Q_4L:
        variant_name = f"{variant['backbone']}_quantum_q{variant['n_qubits']}_l{variant['n_quantum_layers']}"
        print("\n" + "#" * 80)
        print(f"Training requested variant: {variant_name}")
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
        _write_csv(rows, experiment_dir / "quantum_10q4l_results_partial.csv")

    _write_csv(rows, experiment_dir / "quantum_10q4l_results.csv")
    save_json({"results": rows}, experiment_dir / "quantum_10q4l_results.json")

    final_table_rows = rows if args.no_reference_rows else [*REFERENCE_ROWS, *rows]
    write_classification_table(
        rows=rows,
        path=experiment_dir / "proposed_quantum_10q4l_classification_table.tex",
        caption="Test performance of 10-qubit, 4-layer proposed quantum-head variants.",
        label="tab:proposed_quantum_10q4l_results",
    )
    write_classification_table(
        rows=final_table_rows,
        path=experiment_dir / "final_main_results_with_quantum_10q4l.tex",
        caption="Test performance comparison with pretrained and 10-qubit quantum-head models.",
        label="tab:final_main_results_quantum_10q4l",
    )
    write_uncertainty_table(
        rows=rows,
        path=experiment_dir / "proposed_quantum_10q4l_uncertainty_table.tex",
        caption="Uncertainty and rejection-based triage results for 10-qubit quantum-head variants.",
        label="tab:proposed_quantum_10q4l_uncertainty",
    )

    print("\n" + "=" * 80)
    print("Finished 10-qubit / 4-layer quantum-head comparison.")
    print(f"CSV: {experiment_dir / 'quantum_10q4l_results.csv'}")
    print(f"JSON: {experiment_dir / 'quantum_10q4l_results.json'}")
    print(f"Proposed classification table: {experiment_dir / 'proposed_quantum_10q4l_classification_table.tex'}")
    print(f"Final main table: {experiment_dir / 'final_main_results_with_quantum_10q4l.tex'}")
    print(f"Uncertainty table: {experiment_dir / 'proposed_quantum_10q4l_uncertainty_table.tex'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
