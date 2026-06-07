"""Train and evaluate all configured models with one command.

Example:
    python scripts/run_all_models.py --config config_200epoch_gpu.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate import evaluate_checkpoint
from src.train import train_model
from src.utils import get_device, load_config


DEFAULT_MODELS = ["classical_cnn", "mobilenet_mlp", "hybrid_quantum"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate multiple models.")
    parser.add_argument("--config", default="config_200epoch_gpu.yaml", help="Path to config YAML")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        choices=DEFAULT_MODELS,
        help="Optional model list. Default reads models_to_run from config.",
    )
    parser.add_argument("--no-gradcam", action="store_true", help="Skip Grad-CAM during evaluation")
    args = parser.parse_args()

    base_config = load_config(args.config)
    models_to_run = args.models or base_config.get("models_to_run", DEFAULT_MODELS)
    device = get_device(base_config)

    print("=" * 80)
    print("UQ-HyQNet multi-model experiment")
    print(f"Config: {args.config}")
    print(f"Models: {models_to_run}")
    print(f"Device selected by config: {device}")
    print("Note: PennyLane default.qubit is a simulator, not real quantum hardware.")
    print("=" * 80)

    results = []
    for model_name in models_to_run:
        print("\n" + "#" * 80)
        print(f"Starting model: {model_name}")
        print("#" * 80)

        config = dict(base_config)
        config["model_name"] = model_name
        result = train_model(config)
        eval_result = evaluate_checkpoint(
            config=config,
            checkpoint_path=result["best_model_path"],
            run_dir=result["run_dir"],
            make_gradcam=not args.no_gradcam,
        )
        results.append((model_name, result["run_dir"], eval_result["metrics"].get("f1_macro")))

    print("\n" + "=" * 80)
    print("All requested models finished.")
    for model_name, run_dir, f1_macro in results:
        print(f"{model_name}: run_dir={run_dir}, test_f1_macro={f1_macro}")
    print("=" * 80)


if __name__ == "__main__":
    main()
