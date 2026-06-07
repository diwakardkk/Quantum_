"""Very small fake-data test for the full project.

This does not use your real MRI dataset. It creates tiny random images and checks that
the code can build, train briefly, and evaluate each model.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.check_env import check_environment
from src.evaluate import evaluate_checkpoint
from src.train import train_model
from src.utils import load_config


def _make_fake_dataset(root: Path) -> None:
    """Create a tiny train/val/test image dataset."""
    rng = np.random.default_rng(123)
    if root.exists():
        shutil.rmtree(root)

    classes = ["class_a", "class_b"]
    counts = {"train": 4, "val": 2, "test": 2}
    for split, count in counts.items():
        for class_idx, class_name in enumerate(classes):
            class_dir = root / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            for image_idx in range(count):
                base = 70 + class_idx * 80
                arr = rng.normal(loc=base, scale=25, size=(64, 64, 3))
                arr = np.clip(arr, 0, 255).astype(np.uint8)
                Image.fromarray(arr).save(class_dir / f"{class_name}_{image_idx}.jpg")


def main() -> None:
    print("Running environment check...")
    check_environment()

    fake_data_dir = PROJECT_ROOT / "outputs" / "smoke_test" / "fake_data"
    _make_fake_dataset(fake_data_dir)

    base_config = load_config(PROJECT_ROOT / "config.yaml")
    base_config.update(
        {
            "data_dir": str(fake_data_dir),
            "output_dir": "outputs/smoke_test",
            "image_size": 64,
            "batch_size": 2,
            "epochs": 1,
            "num_workers": 0,
            "learning_rate": 1e-4,
            "pretrained_backbone": False,
            "freeze_backbone": True,
            "n_qubits": 2,
            "n_quantum_layers": 1,
            "mc_dropout_samples": 2,
            "gradcam_examples_per_class": 1,
            "device": "cpu",
        }
    )

    for model_name in ["classical_cnn", "mobilenet_mlp", "hybrid_quantum"]:
        print("\n" + "-" * 80)
        print(f"Smoke testing model: {model_name}")
        config = dict(base_config)
        config["model_name"] = model_name
        result = train_model(config, fixed_run_name=f"{model_name}")
        evaluate_checkpoint(
            config,
            result["best_model_path"],
            run_dir=result["run_dir"],
            make_gradcam=False,
        )

    print("\nSmoke test passed")
    print(f"Smoke test outputs saved under: {PROJECT_ROOT / 'outputs' / 'smoke_test'}")


if __name__ == "__main__":
    main()
