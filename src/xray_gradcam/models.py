"""Backbones, the two-stage freezing logic and checkpoint IO.

Both architectures end in a single logit (pneumonia vs normal) so the loss is
``BCEWithLogits`` and Grad-CAM has one scalar to explain.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision import models

from xray_gradcam.utils import SUPPORTED_ARCHS

_HEAD_ATTR = {"resnet50": "fc", "efficientnet_b0": "classifier"}


def build_model(arch: str, pretrained: bool, dropout: float = 0.3) -> nn.Module:
    """torchvision backbone with its classifier replaced by dropout + one logit.

    ``pretrained=False`` is only for tests and smoke runs, where downloading weights is
    pointless. Real runs use the ImageNet weights shipped with torchvision.
    """
    if arch == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, 1))
    elif arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 1))
    else:
        raise ValueError(f"Unknown arch {arch!r}; choose from {SUPPORTED_ARCHS}")
    return model


def head_module(model: nn.Module, arch: str) -> nn.Module:
    return getattr(model, _HEAD_ATTR[arch])


def head_parameters(model: nn.Module, arch: str) -> list[nn.Parameter]:
    return list(head_module(model, arch).parameters())


def backbone_parameters(model: nn.Module, arch: str) -> list[nn.Parameter]:
    head_ids = {id(p) for p in head_parameters(model, arch)}
    return [p for p in model.parameters() if id(p) not in head_ids]


def set_backbone_trainable(model: nn.Module, arch: str, trainable: bool) -> None:
    """Freeze or unfreeze everything except the new head (the head is always trainable)."""
    for p in backbone_parameters(model, arch):
        p.requires_grad = trainable
    for p in head_parameters(model, arch):
        p.requires_grad = True


def set_train_mode(model: nn.Module, arch: str, backbone_frozen: bool) -> None:
    """Put the model in train mode, but keep a frozen backbone's BatchNorm in eval mode.

    Frozen weights with live BatchNorm would still drift, because running mean/variance
    update on every forward pass in train mode. Stage 1 should leave the pretrained
    features exactly as they were.
    """
    model.train()
    if backbone_frozen:
        for name, child in model.named_children():
            if name != _HEAD_ATTR[arch]:
                child.eval()


def get_module(model: nn.Module, dotted_name: str) -> nn.Module:
    """Resolve a name like ``layer4`` or ``features.8``; error lists valid top-level names."""
    modules = dict(model.named_modules())
    if dotted_name not in modules:
        top = sorted({n.split(".")[0] for n in modules if n})
        raise ValueError(f"Layer {dotted_name!r} not found in model. Top-level modules: {top}")
    return modules[dotted_name]


def save_checkpoint(path: str | Path, model: nn.Module, meta: dict[str, Any]) -> None:
    """Save weights plus the metadata needed to rebuild and use the model (threshold, size)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)


def load_checkpoint(
    path: str | Path, device: torch.device | str = "cpu"
) -> tuple[nn.Module, dict[str, Any]]:
    """Rebuild the model from checkpoint metadata. Uses ``weights_only=True`` so a tampered
    file cannot run arbitrary code through pickle."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. Train one with `make train` "
            "(or `python -m xray_gradcam.train --config configs/default.yaml`)."
        )
    blob = torch.load(path, map_location=device, weights_only=True)
    meta = blob["meta"]
    model = build_model(meta["arch"], pretrained=False, dropout=meta.get("dropout", 0.3))
    model.load_state_dict(blob["state_dict"])
    return model.to(device).eval(), meta
