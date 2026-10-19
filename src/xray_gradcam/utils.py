"""Seeding, config loading and run metadata shared by every entry point."""

from __future__ import annotations

import json
import platform
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

SUPPORTED_ARCHS = ("resnet50", "efficientnet_b0")
_REQUIRED_SECTIONS = ("seed", "data", "model", "train", "paths", "evaluation", "gradcam_grid")


def seed_everything(seed: int) -> None:
    """Seed every RNG we use, in one place, and ask cuDNN for deterministic kernels.

    Determinism costs some speed on GPU; for a portfolio-scale run the reproducibility is
    worth more than the saved minutes.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a YAML config and fail loudly if it is missing sections or names an unknown arch."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open() as handle:
        cfg = yaml.safe_load(handle)
    missing = [key for key in _REQUIRED_SECTIONS if key not in cfg]
    if missing:
        raise ValueError(f"Config {path} is missing sections: {missing}")
    if cfg["model"]["arch"] not in SUPPORTED_ARCHS:
        raise ValueError(
            f"model.arch must be one of {SUPPORTED_ARCHS}, got {cfg['model']['arch']!r}"
        )
    if cfg["data"]["source"] not in ("kaggle", "synthetic"):
        raise ValueError("data.source must be 'kaggle' or 'synthetic'")
    if cfg["train"]["imbalance"] not in ("pos_weight", "sampler", "none"):
        raise ValueError("train.imbalance must be 'pos_weight', 'sampler' or 'none'")
    return cfg


def get_device() -> torch.device:
    """Use CUDA when present, otherwise CPU. AMP is only enabled for CUDA by the caller."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def limit_cpu_threads(n: int = 2) -> None:
    """Cap intra-op threads so small runs do not oversubscribe shared machines."""
    if not torch.cuda.is_available():
        torch.set_num_threads(n)


def run_metadata(cfg: dict[str, Any]) -> dict[str, Any]:
    """Seed and package versions that go into every results JSON."""
    import sklearn
    import torchvision

    return {
        "seed": cfg["seed"],
        "data_source": cfg["data"]["source"],
        "synthetic_data": cfg["data"]["source"] == "synthetic",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "device": "cuda:" + torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write JSON with sorted, readable layout; creates parent folders."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
