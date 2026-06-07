"""Detect the dataset and write reproducible split CSV files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import prepare_splits, print_class_distribution, write_split_csvs
from src.utils import load_config, resolve_path, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare or inspect dataset splits.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    records_by_split, class_names, info = prepare_splits(config)
    print_class_distribution(records_by_split, class_names)

    split_dir = resolve_path(config.get("data_dir", "data"), config) / "prepared_splits"
    write_split_csvs(records_by_split, class_names, split_dir)
    save_json({"class_names": class_names, **info}, split_dir / "dataset_info.json")

    print(f"\nPrepared split CSV files saved to: {split_dir}")
    print("No images were copied. The CSV files point to your original image files.")


if __name__ == "__main__":
    main()

