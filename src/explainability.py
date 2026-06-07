"""Grad-CAM explainability utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def _denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalized tensor image back to uint8 RGB."""
    image = tensor.detach().cpu().permute(1, 2, 0).numpy()
    image = (image * IMAGENET_STD) + IMAGENET_MEAN
    image = np.clip(image, 0.0, 1.0)
    return (image * 255).astype(np.uint8)


def _sanitize_filename(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in text)


def _gradcam_for_one_image(
    model: nn.Module,
    image: torch.Tensor,
    target_class: int,
    target_layer: nn.Module,
) -> np.ndarray:
    """Compute one Grad-CAM heatmap."""
    image = image.clone().detach().requires_grad_(True)
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def forward_hook(_module, _inputs, output):
        activations.append(output)

    def backward_hook(_module, _grad_input, grad_output):
        gradients.append(grad_output[0])

    handle_fwd = target_layer.register_forward_hook(forward_hook)
    handle_bwd = target_layer.register_full_backward_hook(backward_hook)

    try:
        model.zero_grad(set_to_none=True)
        logits = model(image)
        score = logits[:, target_class].sum()
        score.backward()

        if not activations or not gradients:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        acts = activations[-1].detach()
        grads = gradients[-1].detach()
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(
            cam,
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        cam = cam[0, 0].cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam
    finally:
        handle_fwd.remove()
        handle_bwd.remove()


def generate_gradcam_examples(
    model: nn.Module,
    dataloader,
    output_dir: str | Path,
    device: torch.device,
    class_names: list[str],
    max_per_class: int = 3,
) -> None:
    """Save Grad-CAM overlays for several examples per class.

    If anything fails, this function prints a warning and returns instead of stopping
    the whole evaluation run.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not hasattr(model, "get_cam_target_layer"):
        print("Warning: this model does not expose a Grad-CAM target layer. Skipping Grad-CAM.")
        return

    try:
        target_layer = model.get_cam_target_layer()
        model.eval()
        saved_per_class = {idx: 0 for idx in range(len(class_names))}

        for batch in dataloader:
            images, labels = batch[0], batch[1]
            paths = list(batch[2]) if len(batch) > 2 else ["image"] * len(labels)

            for idx in range(images.size(0)):
                true_label = int(labels[idx].item())
                if saved_per_class.get(true_label, 0) >= max_per_class:
                    continue

                image = images[idx : idx + 1].to(device)
                logits = model(image)
                pred_label = int(torch.argmax(logits, dim=1).item())
                heatmap = _gradcam_for_one_image(
                    model=model,
                    image=image,
                    target_class=pred_label,
                    target_layer=target_layer,
                )

                rgb = _denormalize_image(images[idx])
                heatmap_uint8 = np.uint8(255 * heatmap)
                heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
                heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
                overlay = np.uint8(0.55 * rgb + 0.45 * heatmap_color)

                class_dir = output_dir / _sanitize_filename(class_names[true_label])
                class_dir.mkdir(parents=True, exist_ok=True)
                original_name = _sanitize_filename(Path(paths[idx]).stem)
                save_path = class_dir / (
                    f"{saved_per_class[true_label]:03d}_{original_name}_"
                    f"pred-{_sanitize_filename(class_names[pred_label])}.png"
                )
                cv2.imwrite(str(save_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                saved_per_class[true_label] += 1

            if all(count >= max_per_class for count in saved_per_class.values()):
                break

        print(f"Grad-CAM images saved to: {output_dir}")
    except Exception as exc:
        print(f"Warning: Grad-CAM generation failed and was skipped. Reason: {exc}")
