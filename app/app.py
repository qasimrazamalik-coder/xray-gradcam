"""Gradio demo: upload a chest X-ray, see the model's probability and a Grad-CAM overlay.

Research demo only. It is not a medical device and must not be used for diagnosis.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from theme import CSS, build_theme
from xray_gradcam.models import load_checkpoint
from xray_gradcam.predict import explain_image

DEFAULT_CHECKPOINT = "checkpoints/resnet50_best.pt"
MAX_BYTES = 10 * 1024 * 1024
ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg"}
Image.MAX_IMAGE_PIXELS = 50_000_000  # reject decompression bombs well above any real X-ray

NOTICE = (
    "**Research demo. Not a medical device. Not for diagnosis.** "
    "The model was trained on a public dataset of paediatric chest X-rays and has not been "
    "validated for clinical use. A probability from this page must never inform a medical decision."
)
EMPTY_STATE = (
    "**No trained checkpoint found at `{path}`.** Train one first:\n\n"
    "```\nmake data\nmake train\n```\n\n"
    "Or point the app at an existing file: `python app/app.py --checkpoint path/to/model.pt`."
)
HOW_IT_WORKS = (
    "A torchvision CNN (ResNet-50 or EfficientNet-B0, fine-tuned in two stages) outputs one "
    "number, the probability that the image shows pneumonia. The heatmap is Grad-CAM: the "
    "gradient of that score with respect to the last convolutional feature maps, averaged per "
    "channel and used to weight them. Brighter regions pushed the score up more. The decision "
    "threshold was chosen on a held-out validation split, not on the test set."
)
LIMITATIONS = (
    "- Trained on one dataset from one hospital's paediatric patients; adult or "
    "other-scanner images may behave unpredictably.\n"
    "- A heatmap shows where the model's score is sensitive, not proof that it found a lesion. "
    "Models can rely on markers, text or borders.\n"
    "- Probabilities may be miscalibrated; see the calibration plot in the repository.\n"
    "- Images are converted to grayscale and resized; metadata is ignored."
)


def validate_upload(path: str | None) -> Image.Image:
    """Return a decoded image or raise ValueError with a message safe to show the user."""
    if not path:
        raise ValueError("No image uploaded. Choose a PNG or JPG file.")
    file = Path(path)
    if file.suffix.lower() not in ALLOWED_SUFFIXES:
        raise ValueError("Unsupported file type. Upload a .png, .jpg or .jpeg file.")
    size = file.stat().st_size
    if size == 0:
        raise ValueError("The file is empty.")
    if size > MAX_BYTES:
        raise ValueError(f"File is {size / 1e6:.1f} MB; the limit is {MAX_BYTES // 2**20} MB.")
    try:
        with Image.open(file) as image:
            if image.format not in ("PNG", "JPEG"):
                raise ValueError("The file content is not a PNG or JPEG image.")
            image.load()
            return image.copy()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("The file could not be read as an image.") from exc


class Predictor:
    """Loads the checkpoint lazily so the app starts, and shows an empty state, without one."""

    def __init__(self, checkpoint: str) -> None:
        self.checkpoint = checkpoint
        self.device = torch.device("cpu")
        self._loaded: tuple[Any, dict[str, Any]] | None = None

    @property
    def available(self) -> bool:
        return Path(self.checkpoint).is_file()

    def _model(self) -> tuple[Any, dict[str, Any]]:
        if self._loaded is None:
            self._loaded = load_checkpoint(self.checkpoint, self.device)
        return self._loaded

    def __call__(self, path: str | None) -> tuple[str, np.ndarray | None, np.ndarray | None]:
        if not self.available:
            return EMPTY_STATE.format(path=self.checkpoint), None, None
        try:
            image = validate_upload(path)
        except ValueError as exc:
            return f"**Cannot use this file.** {exc}", None, None
        model, meta = self._model()
        out = explain_image(model, meta, image, self.device)
        verdict = "above" if out["flagged"] else "below"
        table = (
            "| Quantity | Value |\n| --- | --- |\n"
            f"| Probability of pneumonia | `{out['probability']:.3f}` |\n"
            f"| Decision threshold (validation) | `{out['threshold']:.3f}` |\n"
            f"| Result | {verdict} threshold |\n"
            f"| Architecture | `{meta['arch']}` |"
        )
        if meta.get("synthetic_data"):
            table += "\n\n**This checkpoint was trained on SYNTHETIC data (pipeline check only).**"
        return table, out["original"], out["overlay"]


def build_app(checkpoint: str | None = None) -> gr.Blocks:
    """Build the UI. Works without weights: it renders the empty state instead of failing."""
    predictor = Predictor(checkpoint or os.environ.get("XRAY_CHECKPOINT", DEFAULT_CHECKPOINT))
    with gr.Blocks(title="Chest X-ray Grad-CAM") as demo:
        gr.Markdown("# Chest X-ray pneumonia classifier with Grad-CAM")
        gr.Markdown(NOTICE, elem_classes="notice")
        if not predictor.available:
            gr.Markdown(EMPTY_STATE.format(path=predictor.checkpoint), elem_classes="notice")
        with gr.Row():
            upload = gr.Image(
                label="Chest X-ray (PNG or JPG, up to 10 MB)",
                type="filepath",
                sources=["upload"],
                image_mode="RGB",
            )
        run = gr.Button("Analyse image", variant="primary", interactive=True)
        result = gr.Markdown(label="Result")
        with gr.Row():
            original = gr.Image(label="Original (grayscale)", interactive=False)
            overlay = gr.Image(label="Grad-CAM overlay", interactive=False)
        with gr.Accordion("How it works", open=False):
            gr.Markdown(HOW_IT_WORKS)
        with gr.Accordion("Limitations", open=False):
            gr.Markdown(LIMITATIONS)
        run.click(predictor, inputs=upload, outputs=[result, original, overlay])
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    build_app(args.checkpoint).launch(server_port=args.port, theme=build_theme(), css=CSS)
