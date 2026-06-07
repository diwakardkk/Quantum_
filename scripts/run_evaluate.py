"""Evaluate an already saved checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate import evaluate_checkpoint
from src.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved checkpoint.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pt or last_model.pt")
    parser.add_argument("--run-dir", default=None, help="Optional existing output run directory")
    parser.add_argument("--no-gradcam", action="store_true", help="Skip Grad-CAM generation")
    args = parser.parse_args()

    config = load_config(args.config)
    evaluate_checkpoint(
        config=config,
        checkpoint_path=args.checkpoint,
        run_dir=args.run_dir,
        make_gradcam=not args.no_gradcam,
    )


if __name__ == "__main__":
    main()

