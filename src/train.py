"""Training loop for all UQ-HyQNet model variants."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from tqdm import tqdm

from .dataset import create_dataloaders
from .models import build_model
from .plots import plot_training_curves
from .utils import (
    count_parameters,
    create_output_dir,
    get_device,
    resolve_path,
    save_config_copy,
    save_json,
    set_seed,
)


def _run_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    max_grad_norm: float | None = None,
    desc: str = "Epoch",
) -> dict[str, Any]:
    """Run one train or validation epoch."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_labels: list[int] = []
    all_preds: list[int] = []

    progress = tqdm(dataloader, desc=desc, leave=False)
    for batch in progress:
        images = batch[0].to(device)
        labels = batch[1].to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, labels)
            if is_train:
                loss.backward()
                if max_grad_norm is not None and max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

        preds = torch.argmax(logits, dim=1)
        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += int((preds == labels).sum().item())
        total_samples += batch_size
        all_labels.extend(labels.detach().cpu().tolist())
        all_preds.extend(preds.detach().cpu().tolist())

        progress.set_postfix(
            loss=total_loss / max(1, total_samples),
            acc=total_correct / max(1, total_samples),
        )

    epoch_loss = total_loss / max(1, total_samples)
    epoch_acc = total_correct / max(1, total_samples)
    epoch_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return {
        "loss": float(epoch_loss),
        "accuracy": float(epoch_acc),
        "f1_macro": float(epoch_f1),
    }


def _save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_score: float,
    config: dict[str, Any],
    class_names: list[str],
) -> None:
    """Save a training checkpoint."""
    clean_config = {k: v for k, v in config.items() if not k.startswith("_")}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_score": best_score,
            "model_name": config.get("model_name"),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": clean_config,
            "class_names": class_names,
        },
        path,
    )


def _write_history_csv(history: list[dict[str, Any]], path: str | Path) -> None:
    """Save epoch logs as CSV."""
    if not history:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def _make_class_weight_tensor(dataloader, num_classes: int, device: torch.device) -> torch.Tensor:
    """Create inverse-frequency class weights from the training records."""
    records = getattr(dataloader.dataset, "records", [])
    counts = Counter(int(record["label"]) for record in records)
    total = sum(counts.values())
    weights = []
    for class_idx in range(num_classes):
        count = max(1, counts.get(class_idx, 0))
        weights.append(total / (num_classes * count))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_model(
    config: dict[str, Any],
    run_dir: str | Path | None = None,
    fixed_run_name: str | None = None,
) -> dict[str, Any]:
    """Train the selected model and save a full run folder."""
    set_seed(int(config.get("seed", 42)))
    model_name = str(config.get("model_name", "hybrid_quantum"))
    output_base = resolve_path(config.get("output_dir", "outputs"), config)
    run_dir = Path(run_dir) if run_dir is not None else create_output_dir(
        output_base,
        model_name=model_name,
        fixed_name=fixed_run_name,
    )
    for subdir in ["checkpoints", "plots", "gradcam", "logs"]:
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)

    save_config_copy(config, run_dir)
    dataloaders, class_names, dataset_info = create_dataloaders(config, output_dir=run_dir)
    save_json(dataset_info, run_dir / "logs" / "dataset_info.json")

    device = get_device(config)
    print(f"Using device: {device}")
    model = build_model(model_name, num_classes=len(class_names), config=config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("learning_rate", 1e-4)))
    scheduler = None
    if bool(config.get("use_lr_scheduler", False)):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=str(config.get("lr_scheduler_mode", "min")),
            factor=float(config.get("lr_scheduler_factor", 0.5)),
            patience=int(config.get("lr_scheduler_patience", 4)),
            min_lr=float(config.get("min_learning_rate", 1e-6)),
        )
    if bool(config.get("use_class_weights", False)):
        class_weights = _make_class_weight_tensor(dataloaders["train"], len(class_names), device)
        print(f"Using class-weighted loss: {class_weights.detach().cpu().tolist()}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    summary_text = (
        f"Model: {model_name}\n"
        f"Classes: {class_names}\n"
        f"Total parameters: {total_params}\n"
        f"Trainable parameters: {trainable_params}\n\n"
        f"{model}\n"
    )
    (run_dir / "model_summary.txt").write_text(summary_text, encoding="utf-8")

    epochs = int(config.get("epochs", 5))
    patience = int(config.get("early_stopping_patience", 5))
    best_score = -np.inf
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []

    best_path = run_dir / "checkpoints" / "best_model.pt"
    last_path = run_dir / "checkpoints" / "last_model.pt"

    print(f"Training {model_name} for up to {epochs} epoch(s).")
    max_grad_norm = config.get("gradient_clip_norm", None)
    max_grad_norm = float(max_grad_norm) if max_grad_norm is not None else None
    for epoch in range(1, epochs + 1):
        train_stats = _run_epoch(
            model,
            dataloaders["train"],
            criterion,
            device,
            optimizer=optimizer,
            max_grad_norm=max_grad_norm,
            desc=f"Train {epoch}/{epochs}",
        )
        val_stats = _run_epoch(
            model,
            dataloaders["val"],
            criterion,
            device,
            optimizer=None,
            desc=f"Val {epoch}/{epochs}",
        )

        if scheduler is not None:
            if str(config.get("lr_scheduler_mode", "min")) == "max":
                scheduler.step(val_stats["f1_macro"])
            else:
                scheduler.step(val_stats["loss"])

        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_acc": train_stats["accuracy"],
            "train_f1_macro": train_stats["f1_macro"],
            "val_loss": val_stats["loss"],
            "val_acc": val_stats["accuracy"],
            "val_f1_macro": val_stats["f1_macro"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        _write_history_csv(history, run_dir / "logs" / "training_log.csv")
        plot_training_curves(history, run_dir / "plots" / "training_curves.png")

        print(
            f"Epoch {epoch}: "
            f"train_loss={row['train_loss']:.4f}, train_acc={row['train_acc']:.4f}, "
            f"val_loss={row['val_loss']:.4f}, val_acc={row['val_acc']:.4f}, "
            f"val_f1={row['val_f1_macro']:.4f}"
        )

        score = row["val_f1_macro"]
        if score > best_score:
            best_score = score
            epochs_without_improvement = 0
            _save_checkpoint(best_path, model, optimizer, epoch, best_score, config, class_names)
            print(f"Saved new best checkpoint: {best_path}")
        else:
            epochs_without_improvement += 1

        _save_checkpoint(last_path, model, optimizer, epoch, best_score, config, class_names)

        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {patience} epoch(s) without improvement.")
            break

    return {
        "run_dir": run_dir,
        "best_model_path": best_path,
        "last_model_path": last_path,
        "class_names": class_names,
        "history": history,
    }
