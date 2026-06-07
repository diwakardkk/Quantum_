"""Evaluation, calibration, uncertainty, and plot generation."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .calibration import (
    brier_score_multiclass,
    expected_calibration_error,
    fit_temperature,
)
from .dataset import create_dataloaders
from .explainability import generate_gradcam_examples
from .metrics import compute_metrics, flatten_metrics_for_csv
from .models import build_model
from .plots import (
    plot_confusion_matrix,
    plot_rejection_curve,
    plot_reliability_diagram,
    plot_roc_curve,
    plot_uncertainty_histogram,
)
from .uncertainty import (
    max_softmax_probability,
    mc_dropout_inference,
    prediction_entropy,
    rejection_curve,
)
from .utils import (
    checkpoint_size_mb,
    count_parameters,
    create_output_dir,
    get_device,
    resolve_path,
    save_config_copy,
    save_json,
    set_seed,
)


@torch.no_grad()
def _collect_predictions(model, dataloader, device: torch.device) -> dict[str, Any]:
    """Collect logits, labels, paths, and inference timing."""
    model.eval()
    logits_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    paths: list[str] = []
    total_time = 0.0
    total_images = 0

    for batch in dataloader:
        images = batch[0].to(device)
        labels = batch[1].cpu().numpy()
        batch_paths = list(batch[2]) if len(batch) > 2 else [""] * len(labels)

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        logits = model(images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_time += time.perf_counter() - start
        total_images += images.size(0)

        logits_list.append(logits.detach().cpu().numpy())
        labels_list.append(labels)
        paths.extend(batch_paths)

    logits_np = np.concatenate(logits_list, axis=0)
    labels_np = np.concatenate(labels_list, axis=0)
    return {
        "logits": logits_np,
        "labels": labels_np,
        "paths": paths,
        "inference_time_per_image_sec": total_time / max(1, total_images),
    }


def _save_metrics_csv(metrics: dict[str, Any], path: str | Path) -> None:
    rows = flatten_metrics_for_csv(metrics)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


def _write_ablation_placeholder(config: dict[str, Any], path: str | Path) -> None:
    """Create the requested ablation CSV, even when no ablation run was requested."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "n_qubits": config.get("n_qubits", ""),
            "n_quantum_layers": config.get("n_quantum_layers", ""),
            "status": "placeholder",
            "note": "Run explicit ablation experiments before comparing qubit depth.",
        }
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def evaluate_checkpoint(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    run_dir: str | Path | None = None,
    make_gradcam: bool = True,
) -> dict[str, Any]:
    """Evaluate a saved checkpoint on the test set."""
    set_seed(int(config.get("seed", 42)))
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        # Older PyTorch versions do not have the weights_only argument.
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_config = checkpoint.get("config", {})
    eval_config = dict(checkpoint_config)
    eval_config.update({k: v for k, v in config.items() if k.startswith("_")})
    for key in [
        "data_dir",
        "output_dir",
        "batch_size",
        "num_workers",
        "image_size",
        "device",
        "mc_dropout_samples",
        "confidence_thresholds",
        "gradcam_examples_per_class",
    ]:
        if key in config:
            eval_config[key] = config[key]

    model_name = checkpoint.get("model_name", eval_config.get("model_name", "hybrid_quantum"))
    eval_config["model_name"] = model_name

    if run_dir is None:
        output_base = resolve_path(eval_config.get("output_dir", "outputs"), eval_config)
        run_dir = create_output_dir(output_base, model_name=f"{model_name}_eval")
    run_dir = Path(run_dir)
    for subdir in ["checkpoints", "plots", "gradcam", "logs"]:
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)
    save_config_copy(eval_config, run_dir)

    dataloaders, class_names, dataset_info = create_dataloaders(eval_config, output_dir=run_dir)
    checkpoint_classes = checkpoint.get("class_names")
    if checkpoint_classes and list(checkpoint_classes) != class_names:
        print(
            "Warning: checkpoint class names differ from detected dataset class names. "
            "Using checkpoint class names for model output labels."
        )
        class_names = list(checkpoint_classes)

    device = get_device(eval_config)
    print(f"Evaluating checkpoint on device: {device}")
    model = build_model(model_name, num_classes=len(class_names), config=eval_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    collected = _collect_predictions(model, dataloaders["test"], device)
    logits = collected["logits"]
    labels = collected["labels"]
    probabilities = torch.softmax(torch.tensor(logits), dim=1).numpy()
    preds = probabilities.argmax(axis=1)
    confidence = max_softmax_probability(probabilities)
    entropy = prediction_entropy(probabilities)

    base_metrics = compute_metrics(labels, preds, probabilities, class_names)
    base_metrics["brier_score"] = brier_score_multiclass(probabilities, labels, len(class_names))
    base_metrics["expected_calibration_error"] = expected_calibration_error(probabilities, labels)

    calibrated_probabilities = probabilities
    temperature = None
    calibrated_metrics: dict[str, Any] = {}
    try:
        scaler = fit_temperature(model, dataloaders["val"], device)
        temperature = float(scaler.temperature.detach().cpu().item())
        test_logits_tensor = torch.tensor(logits, dtype=torch.float32, device=device)
        with torch.no_grad():
            calibrated_logits = scaler(test_logits_tensor).cpu()
        calibrated_probabilities = torch.softmax(calibrated_logits, dim=1).numpy()
        calibrated_preds = calibrated_probabilities.argmax(axis=1)
        calibrated_metrics = compute_metrics(
            labels,
            calibrated_preds,
            calibrated_probabilities,
            class_names,
        )
        calibrated_metrics["brier_score"] = brier_score_multiclass(
            calibrated_probabilities,
            labels,
            len(class_names),
        )
        calibrated_metrics["expected_calibration_error"] = expected_calibration_error(
            calibrated_probabilities,
            labels,
        )
    except Exception as exc:
        print(f"Warning: temperature scaling failed and was skipped. Reason: {exc}")
        calibrated_metrics["warning"] = str(exc)

    thresholds = [float(x) for x in eval_config.get("confidence_thresholds", [0.5, 0.7, 0.9])]
    rejection_rows = rejection_curve(probabilities, labels, thresholds)
    pd.DataFrame(rejection_rows).to_csv(run_dir / "logs" / "rejection_curve.csv", index=False)

    mc_samples = int(eval_config.get("mc_dropout_samples", 0))
    mc_metrics: dict[str, Any] = {}
    if mc_samples > 0:
        print(f"Running Monte Carlo dropout inference with {mc_samples} sample(s)...")
        try:
            mc = mc_dropout_inference(model, dataloaders["test"], device, samples=mc_samples)
            mc_probs = mc["probabilities"]
            mc_preds = mc_probs.argmax(axis=1)
            mc_metrics = compute_metrics(labels, mc_preds, mc_probs, class_names)
            mc_metrics["brier_score"] = brier_score_multiclass(mc_probs, labels, len(class_names))
            mc_metrics["expected_calibration_error"] = expected_calibration_error(mc_probs, labels)
            pd.DataFrame(
                {
                    "path": collected["paths"],
                    "true_label": labels,
                    "mc_pred_label": mc_preds,
                    "mc_confidence": max_softmax_probability(mc_probs),
                    "mc_entropy": prediction_entropy(mc_probs),
                    "mc_probability_variance": mc["variance"],
                }
            ).to_csv(run_dir / "logs" / "mc_dropout_predictions.csv", index=False)
        except Exception as exc:
            print(f"Warning: MC dropout failed and was skipped. Reason: {exc}")
            mc_metrics["warning"] = str(exc)

    predictions_df = pd.DataFrame(
        {
            "path": collected["paths"],
            "true_label": labels,
            "true_class": [class_names[int(label)] for label in labels],
            "pred_label": preds,
            "pred_class": [class_names[int(pred)] for pred in preds],
            "confidence": confidence,
            "entropy": entropy,
        }
    )
    for idx, class_name in enumerate(class_names):
        predictions_df[f"prob_{class_name}"] = probabilities[:, idx]
        predictions_df[f"calibrated_prob_{class_name}"] = calibrated_probabilities[:, idx]
    predictions_df.to_csv(run_dir / "predictions.csv", index=False)

    metrics: dict[str, Any] = {
        **base_metrics,
        "model_name": model_name,
        "num_classes": len(class_names),
        "class_names": class_names,
        "temperature": temperature,
        "calibrated": calibrated_metrics,
        "mc_dropout": mc_metrics,
        "parameter_count_total": count_parameters(model, trainable_only=False),
        "parameter_count_trainable": count_parameters(model, trainable_only=True),
        "inference_time_per_image_sec": collected["inference_time_per_image_sec"],
        "model_size_mb": checkpoint_size_mb(checkpoint_path),
        "dataset_info": dataset_info,
    }
    save_json(metrics, run_dir / "metrics.json")
    _save_metrics_csv(metrics, run_dir / "metrics.csv")

    matrix = np.array(base_metrics["confusion_matrix"])
    plot_confusion_matrix(matrix, class_names, run_dir / "plots" / "confusion_matrix.png")
    plot_roc_curve(labels, probabilities, class_names, run_dir / "plots" / "roc_curve.png")
    plot_reliability_diagram(
        probabilities,
        labels,
        run_dir / "plots" / "reliability_diagram.png",
    )
    plot_uncertainty_histogram(
        probabilities,
        labels,
        run_dir / "plots" / "uncertainty_histogram.png",
    )
    plot_rejection_curve(rejection_rows, run_dir / "plots" / "rejection_curve.png")
    _write_ablation_placeholder(eval_config, run_dir / "plots" / "qubit_depth_ablation.csv")

    if make_gradcam:
        generate_gradcam_examples(
            model=model,
            dataloader=dataloaders["test"],
            output_dir=run_dir / "gradcam",
            device=device,
            class_names=class_names,
            max_per_class=int(eval_config.get("gradcam_examples_per_class", 3)),
        )

    print(f"Evaluation complete. Results saved to: {run_dir}")
    return {
        "run_dir": run_dir,
        "metrics": metrics,
    }
