"""General helper functions used across the project."""

from __future__ import annotations

import json
import os
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    """Make common random operations reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    """Load a YAML config file and remember the project root."""
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    config["_config_path"] = str(path)
    config["_project_root"] = str(path.parent)
    return config


def _to_builtin(value: Any) -> Any:
    """Convert NumPy/PyTorch values into JSON-friendly Python values."""
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def save_json(data: dict[str, Any], path: str | Path) -> None:
    """Save a dictionary as formatted JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_to_builtin(data), f, indent=2)


def save_config_copy(config: dict[str, Any], run_dir: str | Path) -> None:
    """Save the exact config used for a run."""
    clean_config = {k: v for k, v in config.items() if not k.startswith("_")}
    path = Path(run_dir) / "config_used.yaml"
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(_to_builtin(clean_config), f, sort_keys=False)


def create_output_dir(
    output_dir: str | Path,
    model_name: str,
    fixed_name: str | None = None,
) -> Path:
    """Create the standard timestamped run directory."""
    output_dir = Path(output_dir)
    run_name = fixed_name or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{model_name}"
    run_dir = output_dir / run_name
    for subdir in ["checkpoints", "plots", "gradcam", "logs"]:
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)
    return run_dir


def get_project_root(config: dict[str, Any]) -> Path:
    """Return the project root from a loaded config."""
    return Path(config.get("_project_root", ".")).expanduser().resolve()


def resolve_path(path_value: str | Path, config: dict[str, Any]) -> Path:
    """Resolve relative paths from the project root, not from the shell location."""
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return get_project_root(config) / path


def get_device(config: dict[str, Any]) -> torch.device:
    """Choose CUDA automatically when available, otherwise CPU."""
    requested = str(config.get("device", "auto")).lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(requested)


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def checkpoint_size_mb(path: str | Path) -> float | None:
    """Return checkpoint size in MB if the file exists."""
    path = Path(path)
    if not path.exists():
        return None
    return path.stat().st_size / (1024 * 1024)


def copy_if_exists(src: str | Path, dst: str | Path) -> None:
    """Copy a file when it exists."""
    src = Path(src)
    if src.exists():
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def safe_num_workers(config: dict[str, Any]) -> int:
    """Use configured workers, but avoid negative values."""
    return max(0, int(config.get("num_workers", 0)))


def print_header(title: str) -> None:
    """Print a simple section header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def ensure_dir(path: str | Path) -> Path:
    """Create and return a directory path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_cpu_thread_defaults() -> None:
    """Keep CPU use friendly on small laptops unless the user configured otherwise."""
    if "OMP_NUM_THREADS" not in os.environ:
        os.environ["OMP_NUM_THREADS"] = "1"

