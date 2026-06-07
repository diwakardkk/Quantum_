"""Train and evaluate the simulated hybrid quantum-classical model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate import evaluate_checkpoint
from src.train import train_model
from src.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hybrid quantum CNN model.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    config["model_name"] = "hybrid_quantum"
    result = train_model(config)
    evaluate_checkpoint(config, result["best_model_path"], run_dir=result["run_dir"])


if __name__ == "__main__":
    main()

