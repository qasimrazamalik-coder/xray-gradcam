import copy
from pathlib import Path

import pytest
import torch

from xray_gradcam.data import make_synthetic_dataset
from xray_gradcam.utils import load_config

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


@pytest.fixture(scope="session")
def synthetic_root(tmp_path_factory) -> Path:
    """Tiny SYNTHETIC dataset in the Kaggle folder layout."""
    return make_synthetic_dataset(
        tmp_path_factory.mktemp("synthetic"),
        size=32,
        train_per_class={"NORMAL": 20, "PNEUMONIA": 40},
        test_per_class={"NORMAL": 8, "PNEUMONIA": 12},
        seed=0,
    )


@pytest.fixture
def tiny_cfg(tmp_path, synthetic_root):
    """Smoke config shrunk to run in seconds, with outputs redirected into tmp_path."""
    cfg = copy.deepcopy(load_config(ROOT / "configs" / "smoke.yaml"))
    cfg["data"]["image_size"] = 32
    cfg["data"]["batch_size"] = 16
    cfg["data"]["synthetic"] = {
        "root": str(synthetic_root),
        "train_per_class": {"NORMAL": 20, "PNEUMONIA": 40},
        "test_per_class": {"NORMAL": 8, "PNEUMONIA": 12},
    }
    cfg["train"]["stage1"] = {"epochs": 1, "lr": 0.001, "patience": 1}
    cfg["train"]["stage2"] = {"epochs": 1, "lr_backbone": 0.0001, "lr_head": 0.0003, "patience": 1}
    cfg["paths"] = {
        "checkpoint": str(tmp_path / "ckpt.pt"),
        "results_dir": str(tmp_path / "results"),
    }
    cfg["evaluation"]["bootstrap_resamples"] = 20
    cfg["gradcam_grid"]["per_category"] = 2
    return cfg
