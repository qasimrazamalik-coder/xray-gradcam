"""Two-stage fine-tuning with early stopping on validation AUROC.

Stage 1 trains only the new head on frozen ImageNet features. Stage 2 unfreezes the whole
network at a lower learning rate. The test folder is never read here: the threshold that
the app and evaluation use is chosen on the validation split at the end of training.
"""

from __future__ import annotations

import argparse
import copy
import time
from typing import Any

import torch
from torch import nn
from tqdm import tqdm

from xray_gradcam import metrics
from xray_gradcam.data import (
    DataBundle,
    make_loader,
    pos_weight_value,
    prepare_data,
)
from xray_gradcam.models import (
    backbone_parameters,
    build_model,
    head_parameters,
    save_checkpoint,
    set_backbone_trainable,
    set_train_mode,
)
from xray_gradcam.predict import collect_probabilities
from xray_gradcam.utils import (
    get_device,
    limit_cpu_threads,
    load_config,
    run_metadata,
    seed_everything,
    write_json,
)


def train_one_epoch(
    model: nn.Module,
    arch: str,
    loader: Any,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    backbone_frozen: bool,
) -> float:
    """One pass over the training data; AMP is active only on CUDA."""
    set_train_mode(model, arch, backbone_frozen)
    use_amp = device.type == "cuda"
    total, count = 0.0, 0
    for images, targets in tqdm(loader, leave=False, desc="train"):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            loss = criterion(model(images).reshape(-1), targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item() * len(targets)
        count += len(targets)
    return total / count


def run_stage(
    name: str,
    model: nn.Module,
    arch: str,
    loaders: dict[str, Any],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    patience: int,
    min_delta: float,
    device: torch.device,
    backbone_frozen: bool,
) -> dict[str, Any]:
    """Train for up to ``epochs``, stop after ``patience`` epochs without val AUROC gain.

    Returns the history and the weights of the best epoch (kept on CPU).
    """
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    best = {"auroc": -1.0, "state": None, "epoch": 0}
    history, stale = [], 0
    for epoch in range(1, epochs + 1):
        started = time.time()
        loss = train_one_epoch(
            model, arch, loaders["train"], criterion, optimizer, scaler, device, backbone_frozen
        )
        scheduler.step()
        labels, probs = collect_probabilities(model, loaders["val"], device)
        val_auroc = metrics.auroc(labels, probs)
        history.append(
            {
                "stage": name,
                "epoch": epoch,
                "train_loss": loss,
                "val_auroc": val_auroc,
                "seconds": round(time.time() - started, 1),
            }
        )
        print(f"[{name}] epoch {epoch}/{epochs} loss {loss:.4f} val_auroc {val_auroc:.4f}")
        if val_auroc > best["auroc"] + min_delta:
            best = {
                "auroc": val_auroc,
                "state": copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()}),
                "epoch": epoch,
            }
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"[{name}] early stop: no val AUROC gain for {patience} epochs")
                break
    return {"history": history, "best": best}


def build_loaders(cfg: dict[str, Any], bundle: DataBundle) -> dict[str, Any]:
    d = cfg["data"]
    common = {
        "image_size": d["image_size"],
        "batch_size": d["batch_size"],
        "num_workers": d["num_workers"],
        "seed": cfg["seed"],
    }
    return {
        "train": make_loader(
            bundle.train,
            train=True,
            balanced_sampler=cfg["train"]["imbalance"] == "sampler",
            **common,
        ),
        "val": make_loader(bundle.val, train=False, **common),
    }


def main(config_path: str) -> dict[str, Any]:
    cfg = load_config(config_path)
    limit_cpu_threads(2)
    seed_everything(cfg["seed"])
    device = get_device()
    arch = cfg["model"]["arch"]
    train_cfg = cfg["train"]

    bundle = prepare_data(cfg)
    loaders = build_loaders(cfg, bundle)
    model = build_model(arch, cfg["model"]["pretrained"], cfg["model"]["dropout"]).to(device)

    pos_weight = None
    if train_cfg["imbalance"] == "pos_weight":
        pos_weight = torch.tensor([pos_weight_value(bundle.train)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    wd = train_cfg["weight_decay"]
    s1, s2 = train_cfg["stage1"], train_cfg["stage2"]

    set_backbone_trainable(model, arch, False)
    opt1 = torch.optim.AdamW(head_parameters(model, arch), lr=s1["lr"], weight_decay=wd)
    stage1 = run_stage(
        "head",
        model,
        arch,
        loaders,
        criterion,
        opt1,
        s1["epochs"],
        s1["patience"],
        train_cfg["min_delta"],
        device,
        backbone_frozen=True,
    )
    model.load_state_dict(stage1["best"]["state"])

    set_backbone_trainable(model, arch, True)
    opt2 = torch.optim.AdamW(
        [
            {"params": backbone_parameters(model, arch), "lr": s2["lr_backbone"]},
            {"params": head_parameters(model, arch), "lr": s2["lr_head"]},
        ],
        weight_decay=wd,
    )
    stage2 = run_stage(
        "finetune",
        model,
        arch,
        loaders,
        criterion,
        opt2,
        s2["epochs"],
        s2["patience"],
        train_cfg["min_delta"],
        device,
        backbone_frozen=False,
    )

    # Keep whichever stage produced the best validation AUROC.
    final = stage2["best"] if stage2["best"]["auroc"] >= stage1["best"]["auroc"] else stage1["best"]
    model.load_state_dict(final["state"])
    labels, probs = collect_probabilities(model, loaders["val"], device)
    threshold = metrics.youden_threshold(labels, probs)

    meta = {
        "arch": arch,
        "dropout": cfg["model"]["dropout"],
        "image_size": cfg["data"]["image_size"],
        "gradcam_layer": cfg["model"]["gradcam_layer"],
        "threshold": threshold,
        "threshold_rule": "Youden's J on the validation split",
        "val_auroc": final["auroc"],
        "pretrained_init": cfg["model"]["pretrained"],
        "synthetic_data": cfg["data"]["source"] == "synthetic",
        "seed": cfg["seed"],
    }
    save_checkpoint(cfg["paths"]["checkpoint"], model.cpu(), meta)
    summary = {
        **run_metadata(cfg),
        "arch": arch,
        "imbalance": train_cfg["imbalance"],
        "pos_weight": None if pos_weight is None else float(pos_weight),
        "best_val_auroc": final["auroc"],
        "validation_threshold": threshold,
        "history": stage1["history"] + stage2["history"],
        "data": bundle.info,
        "checkpoint": cfg["paths"]["checkpoint"],
    }
    write_json(f"{cfg['paths']['results_dir']}/train_summary.json", summary)
    print(
        f"Saved {cfg['paths']['checkpoint']} (val AUROC {final['auroc']:.4f}, thr {threshold:.3f})"
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    main(parser.parse_args().config)
