"""Dataset discovery, stratified splitting, transforms and the synthetic fixture.

The Kaggle release ships a 16-image ``val/`` folder. Sixteen images cannot support early
stopping or threshold selection, so we ignore it, re-split the official ``train/`` folder
into train/val with stratification and a seed, and leave the official ``test/`` folder
untouched until the final evaluation.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

CLASS_NAMES = ("NORMAL", "PNEUMONIA")
IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

Sample = tuple[str, int]


# --------------------------------------------------------------------------- discovery


def find_dataset_root(root: str | Path) -> Path:
    """Locate the folder holding ``train/`` and ``test/``.

    The Kaggle archive nests a second ``chest_xray/`` folder inside the first on some
    downloads, so look one level down before giving up.
    """
    root = Path(root)
    for candidate in (root, root / "chest_xray"):
        if (candidate / "train").is_dir() and (candidate / "test").is_dir():
            return candidate
    raise FileNotFoundError(
        f"No dataset with train/ and test/ folders found under '{root}'. "
        "Run `make data` (or `python scripts/download_data.py`) first, "
        "or set data.root in the config."
    )


def list_samples(split_dir: str | Path) -> list[Sample]:
    """All (path, label) pairs under ``split_dir/{NORMAL,PNEUMONIA}``, sorted for determinism."""
    split_dir = Path(split_dir)
    samples: list[Sample] = []
    for label, name in enumerate(CLASS_NAMES):
        class_dir = split_dir / name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Expected class folder {class_dir}")
        files = sorted(p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if not files:
            raise FileNotFoundError(f"No images found in {class_dir}")
        samples.extend((str(p), label) for p in files)
    return samples


def stratified_split(
    samples: list[Sample], val_fraction: float, seed: int
) -> tuple[list[Sample], list[Sample]]:
    """Split into train/val preserving the class ratio. Same inputs and seed give the same split."""
    if not 0.0 < val_fraction < 0.5:
        raise ValueError("val_fraction must be in (0, 0.5)")
    labels = [label for _, label in samples]
    train, val = train_test_split(
        samples, test_size=val_fraction, stratify=labels, random_state=seed, shuffle=True
    )
    return sorted(train), sorted(val)


def split_fingerprint(samples: list[Sample]) -> str:
    """Short hash of a split, stored in results so two runs can be checked for the same split."""
    digest = hashlib.sha256("\n".join(f"{p}|{y}" for p, y in sorted(samples)).encode())
    return digest.hexdigest()[:12]


# --------------------------------------------------------------------------- transforms


def build_transforms(image_size: int, train: bool) -> transforms.Compose:
    """Light, anatomy-preserving augmentation for training; deterministic resize for eval.

    No vertical flip (an upside-down chest is not a plausible image) and no horizontal flip
    (it would mirror the heart to the right side, which is a rare anatomical variant and
    teaches the model something false). Small rotations, shifts, scale and brightness/contrast
    jitter mimic differences in positioning and exposure.
    """
    ops: list[Any] = [transforms.Grayscale(num_output_channels=3)]
    if train:
        ops += [
            transforms.Resize((image_size, image_size)),
            transforms.RandomAffine(degrees=7, translate=(0.04, 0.04), scale=(0.95, 1.05)),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
        ]
    else:
        ops += [transforms.Resize((image_size, image_size))]
    ops += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(ops)


class XrayDataset(Dataset):
    """Reads an image from disk, converts to 3-channel grayscale, applies a transform."""

    def __init__(self, samples: list[Sample], transform: transforms.Compose) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("L"))
        return tensor, torch.tensor(float(label))


# --------------------------------------------------------------------------- synthetic fixture


def _blob(yy: np.ndarray, xx: np.ndarray, cy: float, cx: float, sigma: float) -> np.ndarray:
    return np.exp(-(((yy - cy) ** 2) + ((xx - cx) ** 2)) / (2 * sigma**2))


def synthetic_image(label: int, size: int, rng: np.random.Generator) -> Image.Image:
    """One SYNTHETIC grayscale 'radiograph'. Not a medical image.

    Two dark ellipses stand in for lungs on a brighter body. Label 1 adds a bright blob at a
    random position inside one lung with random strength (some are faint, so classes overlap).
    The signal's location is known, which lets tests ask whether Grad-CAM can find it.
    """
    yy, xx = np.mgrid[0:size, 0:size] / (size - 1)
    img = 0.55 + 0.05 * (yy - 0.5) + rng.normal(0, 0.03, (size, size))
    for cx in (0.3, 0.7):
        img -= 0.25 * np.clip(1 - (((xx - cx) / 0.15) ** 2 + ((yy - 0.5) / 0.3) ** 2), 0, 1) ** 0.5
    img += 0.08 * _blob(yy, xx, 0.5, 0.5, 0.05)  # mediastinum
    if label == 1:
        side = rng.choice([0.3, 0.7])
        cy = 0.5 + rng.uniform(-0.15, 0.15)
        cx = side + rng.uniform(-0.05, 0.05)
        img += rng.uniform(0.15, 0.4) * _blob(yy, xx, cy, cx, rng.uniform(0.05, 0.09))
    img = np.clip(img + rng.normal(0, 0.02, img.shape), 0, 1)
    return Image.fromarray((img * 255).astype(np.uint8), mode="L")


def make_synthetic_dataset(
    root: str | Path,
    size: int,
    train_per_class: dict[str, int],
    test_per_class: dict[str, int],
    seed: int,
) -> Path:
    """Write a Kaggle-shaped folder (train/test x NORMAL/PNEUMONIA) of synthetic PNGs.

    Writing real files, rather than tensors, makes tests and smoke exercise the same loading
    code as the real dataset. Existing files are overwritten so the fixture is reproducible.
    """
    root = Path(root)
    rng = np.random.default_rng(seed)
    for split, counts in (("train", train_per_class), ("test", test_per_class)):
        for label, name in enumerate(CLASS_NAMES):
            out = root / split / name
            out.mkdir(parents=True, exist_ok=True)
            for i in range(counts[name]):
                synthetic_image(label, size, rng).save(
                    out / f"synthetic_{split}_{name}_{i:04d}.png"
                )
    (root / "README.txt").write_text(
        "SYNTHETIC images generated by xray_gradcam.data.make_synthetic_dataset.\n"
        "Not medical data. Used only for tests and smoke runs.\n"
    )
    return root


# --------------------------------------------------------------------------- bundle and loaders


@dataclass
class DataBundle:
    """Train/val/test sample lists plus bookkeeping for the results JSON."""

    train: list[Sample]
    val: list[Sample]
    test: list[Sample]
    info: dict[str, Any]


def class_counts(samples: list[Sample]) -> dict[str, int]:
    counts = Counter(label for _, label in samples)
    return {name: counts.get(i, 0) for i, name in enumerate(CLASS_NAMES)}


def prepare_data(cfg: dict[str, Any]) -> DataBundle:
    """Resolve the dataset (real or synthetic) and build the stratified train/val/test lists."""
    data_cfg = cfg["data"]
    if data_cfg["source"] == "synthetic":
        syn = data_cfg["synthetic"]
        root = make_synthetic_dataset(
            syn["root"],
            data_cfg["image_size"],
            syn["train_per_class"],
            syn["test_per_class"],
            cfg["seed"],
        )
    else:
        root = find_dataset_root(data_cfg["root"])
    train, val = stratified_split(
        list_samples(root / "train"), data_cfg["val_fraction"], cfg["seed"]
    )
    test = list_samples(root / "test")
    info = {
        "dataset_root": str(root),
        "counts": {
            "train": class_counts(train),
            "val": class_counts(val),
            "test": class_counts(test),
        },
        "split_fingerprint": {
            "train": split_fingerprint(train),
            "val": split_fingerprint(val),
            "test": split_fingerprint(test),
        },
        "official_val_folder_used": False,
    }
    return DataBundle(train=train, val=val, test=test, info=info)


def pos_weight_value(samples: list[Sample]) -> float:
    """n_negative / n_positive: the factor that balances the two classes' total loss."""
    counts = class_counts(samples)
    return counts["NORMAL"] / max(counts["PNEUMONIA"], 1)


def _worker_init(worker_id: int) -> None:
    np.random.seed(torch.initial_seed() % 2**32 + worker_id)


def make_loader(
    samples: list[Sample],
    image_size: int,
    batch_size: int,
    num_workers: int,
    train: bool,
    seed: int,
    balanced_sampler: bool = False,
) -> DataLoader:
    """DataLoader with a seeded generator. Optionally oversamples the minority class."""
    dataset = XrayDataset(samples, build_transforms(image_size, train=train))
    generator = torch.Generator().manual_seed(seed)
    sampler = None
    if train and balanced_sampler:
        labels = np.array([label for _, label in samples])
        class_weight = 1.0 / np.bincount(labels, minlength=len(CLASS_NAMES))
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(class_weight[labels], dtype=torch.double),
            num_samples=len(samples),
            replacement=True,
            generator=generator,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        worker_init_fn=_worker_init,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )
