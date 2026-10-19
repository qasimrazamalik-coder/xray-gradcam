import pytest
import torch
from torchvision import transforms

from xray_gradcam.data import (
    XrayDataset,
    build_transforms,
    class_counts,
    find_dataset_root,
    list_samples,
    pos_weight_value,
    stratified_split,
)


def test_split_is_deterministic(synthetic_root):
    samples = list_samples(synthetic_root / "train")
    assert stratified_split(samples, 0.2, seed=7) == stratified_split(samples, 0.2, seed=7)
    assert stratified_split(samples, 0.2, seed=7) != stratified_split(samples, 0.2, seed=8)


def test_split_has_no_overlap_and_covers_everything(synthetic_root):
    samples = list_samples(synthetic_root / "train")
    train, val = stratified_split(samples, 0.2, seed=1)
    assert not {p for p, _ in train} & {p for p, _ in val}
    assert sorted(train + val) == sorted(samples)


def test_split_is_stratified(synthetic_root):
    samples = list_samples(synthetic_root / "train")
    _, val = stratified_split(samples, 0.25, seed=1)
    assert class_counts(val) == {"NORMAL": 5, "PNEUMONIA": 10}


def test_official_test_folder_is_disjoint_from_train_val(synthetic_root):
    train_paths = {p for p, _ in list_samples(synthetic_root / "train")}
    test_paths = {p for p, _ in list_samples(synthetic_root / "test")}
    assert not train_paths & test_paths


def test_pos_weight_is_negative_over_positive(synthetic_root):
    assert pos_weight_value(list_samples(synthetic_root / "train")) == pytest.approx(0.5)


def test_invalid_val_fraction_rejected(synthetic_root):
    with pytest.raises(ValueError):
        stratified_split(list_samples(synthetic_root / "train"), 0.9, seed=0)


def test_missing_dataset_error_names_the_fix(tmp_path):
    with pytest.raises(FileNotFoundError, match="make data"):
        find_dataset_root(tmp_path)


@pytest.mark.parametrize("train", [True, False])
def test_transform_output_shape_and_channels(synthetic_root, train):
    samples = list_samples(synthetic_root / "train")
    x, y = XrayDataset(samples, build_transforms(48, train=train))[0]
    assert x.shape == (3, 48, 48)
    assert x.dtype == torch.float32
    assert y.item() in (0.0, 1.0)


def test_eval_transform_is_deterministic(synthetic_root):
    samples = list_samples(synthetic_root / "train")
    ds = XrayDataset(samples, build_transforms(32, train=False))
    assert torch.equal(ds[3][0], ds[3][0])


def test_no_flip_augmentations():
    ops = build_transforms(32, train=True).transforms
    flips = (transforms.RandomVerticalFlip, transforms.RandomHorizontalFlip)
    assert not any(isinstance(op, flips) for op in ops)


def test_synthetic_fixture_is_labelled(synthetic_root):
    assert "SYNTHETIC" in (synthetic_root / "README.txt").read_text()


def test_pneumonia_images_are_brighter_in_the_lungs(synthetic_root):
    """The fixture's one signal: pneumonia images carry extra brightness inside the lung area."""
    import numpy as np
    from PIL import Image

    def lung_mean(label):
        vals = []
        for path, y in list_samples(synthetic_root / "train"):
            if y == label:
                img = np.asarray(Image.open(path), dtype=float)
                vals.append(img[10:22, 4:28].mean())
        return np.mean(vals)

    assert lung_mean(1) > lung_mean(0)
