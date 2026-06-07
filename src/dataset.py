"""Dataset detection and dataloader creation.

The code stores image paths and labels, not image pixels, so large datasets are not
loaded into memory. Images are read only when PyTorch asks for a batch.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .utils import get_project_root, resolve_path, safe_num_workers, save_json

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLIT_NAMES = ("train", "val", "test")


def normalize_class_name(name: str) -> str:
    """Normalize common class name variants."""
    cleaned = name.strip().lower().replace("-", "_").replace(" ", "_")
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    if cleaned in {"notumor", "no_tumor", "no__tumor", "normal", "no_tumour"}:
        return "no_tumor"
    return cleaned


def _contains_images(directory: Path) -> bool:
    return any(
        path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        for path in directory.rglob("*")
    )


def _class_dirs(directory: Path) -> list[Path]:
    if not directory.exists() or not directory.is_dir():
        return []
    return sorted(
        [child for child in directory.iterdir() if child.is_dir() and _contains_images(child)],
        key=lambda p: normalize_class_name(p.name),
    )


def _find_split_dirs(root: Path) -> dict[str, Path]:
    """Find train/val/test folders case-insensitively."""
    found: dict[str, Path] = {}
    if not root.exists() or not root.is_dir():
        return found
    for child in root.iterdir():
        if child.is_dir() and child.name.lower() in SPLIT_NAMES:
            found[child.name.lower()] = child
    return found


def _candidate_dataset_roots(config: dict[str, Any]) -> list[Path]:
    """Try the configured data path plus common nearby dataset folders."""
    project_root = get_project_root(config)
    configured = resolve_path(config.get("data_dir", "data"), config)
    cwd = Path.cwd().resolve()
    candidates = [
        configured,
        project_root / "data",
        project_root / "dataset",
        project_root / "Brain_Tumor_MRI_Dataset",
        project_root.parent / "Brain_Tumor_MRI_Dataset",
        cwd / "data",
        cwd / "dataset",
        cwd / "Brain_Tumor_MRI_Dataset",
        cwd.parent / "Brain_Tumor_MRI_Dataset",
    ]

    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate not in seen:
            unique.append(candidate)
            seen.add(candidate)
    return unique


def find_dataset_root(config: dict[str, Any]) -> tuple[Path, str]:
    """Detect a supported dataset folder."""
    tried: list[str] = []
    for root in _candidate_dataset_roots(config):
        tried.append(str(root))
        split_dirs = _find_split_dirs(root)
        has_split_images = any(_class_dirs(split_dir) for split_dir in split_dirs.values())
        if split_dirs and has_split_images:
            return root, "split"
        if _class_dirs(root):
            return root, "class_root"

    message = [
        "Could not find a valid image dataset.",
        "Expected one of these structures:",
        "  data/train/<class>/*.jpg, data/val/<class>/*.jpg, data/test/<class>/*.jpg",
        "  data/<class>/*.jpg",
        "  dataset/<class>/*.jpg",
        "Also tried nearby Brain_Tumor_MRI_Dataset folders.",
        "Paths checked:",
    ]
    message.extend(f"  - {path}" for path in tried)
    raise FileNotFoundError("\n".join(message))


def _collect_records_from_class_root(root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    class_dirs = _class_dirs(root)
    if not class_dirs:
        raise FileNotFoundError(f"No class folders with images found in: {root}")

    class_names = sorted({normalize_class_name(path.name) for path in class_dirs})
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    records: list[dict[str, Any]] = []

    for class_dir in class_dirs:
        class_name = normalize_class_name(class_dir.name)
        label = class_to_idx[class_name]
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                records.append(
                    {
                        "path": str(image_path.resolve()),
                        "label": label,
                        "class_name": class_name,
                    }
                )
    return records, class_names


def _collect_records_from_split_root(root: Path) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    split_dirs = _find_split_dirs(root)
    raw_class_names: set[str] = set()
    for split_dir in split_dirs.values():
        for class_dir in _class_dirs(split_dir):
            raw_class_names.add(normalize_class_name(class_dir.name))

    if not raw_class_names:
        raise FileNotFoundError(f"No class folders with images found in split dataset: {root}")

    class_names = sorted(raw_class_names)
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    records_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLIT_NAMES}

    for split_name, split_dir in split_dirs.items():
        for class_dir in _class_dirs(split_dir):
            class_name = normalize_class_name(class_dir.name)
            label = class_to_idx[class_name]
            for image_path in sorted(class_dir.rglob("*")):
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    records_by_split[split_name].append(
                        {
                            "path": str(image_path.resolve()),
                            "label": label,
                            "class_name": class_name,
                        }
                    )
    return records_by_split, class_names


def _can_stratify(records: list[dict[str, Any]]) -> bool:
    counts = Counter(record["label"] for record in records)
    return len(counts) > 1 and min(counts.values()) >= 2


def _split_records(
    records: list[dict[str, Any]],
    test_size: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split records reproducibly, stratifying when class counts allow it."""
    if not records:
        return [], []
    if test_size <= 0:
        return records, []
    if len(records) < 2:
        return records, []

    labels = [record["label"] for record in records]
    stratify = labels if _can_stratify(records) else None
    train_records, held_out_records = train_test_split(
        records,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    return list(train_records), list(held_out_records)


def prepare_splits(config: dict[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], list[str], dict[str, Any]]:
    """Detect the dataset and create train/val/test records when needed."""
    root, mode = find_dataset_root(config)
    seed = int(config.get("seed", 42))
    val_ratio = float(config.get("val_ratio", 0.15))
    test_ratio = float(config.get("test_ratio", 0.15))

    print(f"Using dataset root: {root}")
    print(f"Detected dataset mode: {mode}")

    if mode == "split":
        records_by_split, class_names = _collect_records_from_split_root(root)
        if not records_by_split["train"]:
            raise FileNotFoundError(
                f"A split dataset was detected at {root}, but no train images were found."
            )

        # If a validation folder is missing, make a reproducible validation split
        # from the training records without copying image files.
        if not records_by_split["val"]:
            records_by_split["train"], records_by_split["val"] = _split_records(
                records_by_split["train"],
                test_size=val_ratio,
                seed=seed,
            )

        # If a test folder is missing, split it from the remaining training records.
        if not records_by_split["test"]:
            records_by_split["train"], records_by_split["test"] = _split_records(
                records_by_split["train"],
                test_size=test_ratio,
                seed=seed + 1,
            )
    else:
        all_records, class_names = _collect_records_from_class_root(root)
        train_val_records, test_records = _split_records(
            all_records,
            test_size=test_ratio,
            seed=seed,
        )
        adjusted_val_ratio = val_ratio / max(1e-8, 1.0 - test_ratio)
        train_records, val_records = _split_records(
            train_val_records,
            test_size=adjusted_val_ratio,
            seed=seed + 1,
        )
        records_by_split = {
            "train": train_records,
            "val": val_records,
            "test": test_records,
        }

    info = {
        "dataset_root": str(root),
        "dataset_mode": mode,
        "counts": split_class_counts(records_by_split, class_names),
    }
    return records_by_split, class_names, info


def split_class_counts(
    records_by_split: dict[str, list[dict[str, Any]]],
    class_names: list[str],
) -> dict[str, dict[str, int]]:
    """Count images per split and class."""
    counts: dict[str, dict[str, int]] = {}
    for split, records in records_by_split.items():
        counter = Counter(record["label"] for record in records)
        counts[split] = {
            class_name: int(counter.get(idx, 0))
            for idx, class_name in enumerate(class_names)
        }
        counts[split]["total"] = int(len(records))
    return counts


def print_class_distribution(
    records_by_split: dict[str, list[dict[str, Any]]],
    class_names: list[str],
) -> None:
    """Print an easy-to-read class distribution table."""
    counts = split_class_counts(records_by_split, class_names)
    print("\nClass distribution:")
    for split in SPLIT_NAMES:
        if split not in counts:
            continue
        print(f"  {split}:")
        for class_name in class_names:
            print(f"    {class_name}: {counts[split][class_name]}")
        print(f"    total: {counts[split]['total']}")


class RGBImagePathDataset(Dataset):
    """A lightweight dataset that reads RGB images from saved paths."""

    def __init__(self, records: list[dict[str, Any]], transform=None):
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image_path = Path(record["path"])
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(record["label"]), str(image_path)


def build_transforms(image_size: int, train: bool = False):
    """Build standard MRI image transforms.

    MobileNetV2 expects ImageNet-style normalization. The same normalization is used
    for the small CNN baseline so comparisons are simple.
    """
    transform_steps = [
        transforms.Resize((image_size, image_size)),
    ]
    if train:
        transform_steps.append(transforms.RandomHorizontalFlip(p=0.5))
    transform_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return transforms.Compose(transform_steps)


def create_dataloaders(
    config: dict[str, Any],
    output_dir: str | Path | None = None,
) -> tuple[dict[str, DataLoader], list[str], dict[str, Any]]:
    """Create PyTorch DataLoaders for train/val/test."""
    records_by_split, class_names, info = prepare_splits(config)
    print_class_distribution(records_by_split, class_names)

    if output_dir is not None:
        save_json({"class_names": class_names}, Path(output_dir) / "class_names.json")

    image_size = int(config.get("image_size", 224))
    batch_size = int(config.get("batch_size", 8))
    num_workers = safe_num_workers(config)
    pin_memory = torch.cuda.is_available()

    dataloaders: dict[str, DataLoader] = {}
    for split in SPLIT_NAMES:
        dataset = RGBImagePathDataset(
            records_by_split[split],
            transform=build_transforms(image_size=image_size, train=(split == "train")),
        )
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
        )
    return dataloaders, class_names, info


def write_split_csvs(
    records_by_split: dict[str, list[dict[str, Any]]],
    class_names: list[str],
    output_dir: str | Path,
) -> None:
    """Write reproducible split lists to CSV files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, records in records_by_split.items():
        path = output_dir / f"{split}.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "label", "class_name"])
            writer.writeheader()
            for record in records:
                writer.writerow(record)

    with (output_dir / "class_names.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "class_name"])
        for idx, class_name in enumerate(class_names):
            writer.writerow([idx, class_name])
