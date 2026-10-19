import numpy as np
import pytest
import torch
from PIL import Image

import app as xray_app
from xray_gradcam.models import build_model, save_checkpoint


def _png(path, size=(40, 40), mode="L"):
    Image.fromarray(np.random.default_rng(0).integers(0, 255, size, dtype=np.uint8), mode).save(
        path
    )
    return str(path)


def test_app_builds_without_weights(tmp_path):
    demo = xray_app.build_app(str(tmp_path / "missing.pt"))
    assert demo is not None


def test_empty_state_when_no_checkpoint(tmp_path):
    predictor = xray_app.Predictor(str(tmp_path / "missing.pt"))
    text, original, overlay = predictor(_png(tmp_path / "x.png"))
    assert "No trained checkpoint" in text and "make train" in text
    assert original is None and overlay is None


@pytest.mark.parametrize(
    ("name", "payload", "message"),
    [
        ("scan.gif", b"GIF89a", "Unsupported"),
        ("empty.png", b"", "empty"),
        ("fake.png", b"this is not an image", "could not be read"),
    ],
)
def test_validate_upload_rejects_bad_files(tmp_path, name, payload, message):
    path = tmp_path / name
    path.write_bytes(payload)
    with pytest.raises(ValueError, match=message):
        xray_app.validate_upload(str(path))


def test_validate_upload_rejects_none_and_oversize(tmp_path):
    with pytest.raises(ValueError, match="No image"):
        xray_app.validate_upload(None)
    big = tmp_path / "big.png"
    with big.open("wb") as handle:
        handle.truncate(xray_app.MAX_BYTES + 1)
    with pytest.raises(ValueError, match="limit"):
        xray_app.validate_upload(str(big))


def test_wrong_content_with_image_extension_rejected(tmp_path):
    path = tmp_path / "actually.png"
    Image.new("RGB", (8, 8)).save(path, format="BMP")
    with pytest.raises(ValueError, match="not a PNG or JPEG"):
        xray_app.validate_upload(str(path))


def test_prediction_with_checkpoint(tmp_path):
    torch.manual_seed(0)
    model = build_model("efficientnet_b0", pretrained=False)
    meta = {
        "arch": "efficientnet_b0", "dropout": 0.3, "image_size": 64, "gradcam_layer": "features.8",
        "threshold": 0.5, "synthetic_data": True,
    }  # fmt: skip
    save_checkpoint(tmp_path / "m.pt", model, meta)
    predictor = xray_app.Predictor(str(tmp_path / "m.pt"))
    text, original, overlay = predictor(_png(tmp_path / "x.png", (50, 70)))
    assert "Probability of pneumonia" in text and "SYNTHETIC" in text
    assert original.shape == overlay.shape == (64, 64, 3)
    # Colour input is accepted and flattened to grayscale.
    rgb = tmp_path / "rgb.png"
    Image.new("RGB", (30, 30), (200, 20, 20)).save(rgb)
    assert predictor(str(rgb))[1].shape == (64, 64, 3)
