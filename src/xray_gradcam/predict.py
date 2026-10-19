"""Inference helpers: batch probabilities and single-image explanations for the app."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from xray_gradcam.data import build_transforms
from xray_gradcam.gradcam import GradCAM, overlay_heatmap


@torch.no_grad()
def collect_probabilities(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Return (labels, P(pneumonia)) for every image in the loader, in loader order."""
    model.eval()
    labels, probs = [], []
    for images, targets in loader:
        logits = model(images.to(device)).reshape(-1)
        probs.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(targets.numpy())
    return np.concatenate(labels).astype(int), np.concatenate(probs)


def to_gray3(image: Image.Image) -> Image.Image:
    """Grayscale, 3 channels. Colour carries no information in a radiograph, and a screenshot
    with a coloured UI should not be able to change the prediction through colour."""
    return image.convert("L").convert("RGB")


def explain_image(
    model: nn.Module, meta: dict[str, Any], image: Image.Image, device: torch.device
) -> dict[str, Any]:
    """Probability, decision at the validation threshold, CAM and the resized display image."""
    size = meta["image_size"]
    gray = to_gray3(image).resize((size, size), Image.Resampling.BILINEAR)
    tensor = build_transforms(size, train=False)(gray).unsqueeze(0).to(device)
    with GradCAM(model, meta["gradcam_layer"]) as cam_engine:
        logits, cam = cam_engine.logits_and_cam(tensor, target_class=1)
    prob = float(torch.sigmoid(logits)[0])
    display = np.asarray(gray)
    cam_np = cam[0].cpu().numpy()
    return {
        "probability": prob,
        "threshold": float(meta["threshold"]),
        "flagged": prob >= float(meta["threshold"]),
        "original": display,
        "overlay": overlay_heatmap(display, cam_np),
        "cam": cam_np,
    }
