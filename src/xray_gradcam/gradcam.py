"""Grad-CAM (Selvaraju et al., 2017) written against plain PyTorch hooks.

Idea: take the feature maps A^k of a late conv layer, weight each by the mean gradient of
the target score with respect to it, sum, and keep only the positive part (ReLU) because
we want regions that *increase* the score for the target class.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps
from torch import nn

from xray_gradcam.models import get_module

# Perceptually uniform, sequential colormap: equal steps in value look like equal steps in
# colour, and luminance rises monotonically. A rainbow map such as jet has bright bands and
# false edges that read as boundaries in the heatmap that the model does not have, and it
# is poorly legible for people with colour-vision deficiency. Magma also stays readable
# over a grey X-ray.
HEATMAP_CMAP = "magma"


class GradCAM:
    """Compute class-activation maps for a single-logit binary classifier.

    The forward hook stores the layer output and registers a tensor hook on it, rather than
    using ``register_full_backward_hook``. Tensor hooks are not affected by the in-place ReLU
    and SiLU activations that both torchvision backbones use.
    """

    def __init__(self, model: nn.Module, target_layer: str) -> None:
        self.model = model
        self.target_layer = target_layer
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._handle = get_module(model, target_layer).register_forward_hook(self._on_forward)

    def _on_forward(self, _module: nn.Module, _inputs: tuple, output: torch.Tensor) -> None:
        self._activations = output.detach()
        if output.requires_grad:
            output.register_hook(self._on_backward)

    def _on_backward(self, grad: torch.Tensor) -> None:
        self._gradients = grad.detach()

    def close(self) -> None:
        """Remove the hook. Call this (or use the context manager) so the model is left clean."""
        self._handle.remove()

    def __enter__(self) -> GradCAM:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def logits_and_cam(
        self, x: torch.Tensor, target_class: int = 1, upsample: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (logits of shape (B,), CAMs of shape (B,H,W) in [0,1]) for a batch.

        ``target_class=1`` explains the pneumonia logit; ``0`` explains its negation (what
        pushes the prediction towards NORMAL). With ``upsample=False`` the CAM keeps the
        layer's native resolution. A CAM that is all zeros (no positive evidence) stays zero
        instead of being rescaled into noise.
        """
        if target_class not in (0, 1):
            raise ValueError("target_class must be 0 (NORMAL) or 1 (PNEUMONIA)")
        if x.ndim != 4:
            raise ValueError(f"Expected a (B,C,H,W) tensor, got shape {tuple(x.shape)}")
        was_training = self.model.training
        self.model.eval()
        self._activations = self._gradients = None
        # requires_grad on the input guarantees a graph even if every parameter is frozen.
        inp = x.detach().clone().requires_grad_(True)
        try:
            with torch.enable_grad():
                logits = self.model(inp).reshape(-1)
                score = logits if target_class == 1 else -logits
                # Samples are independent in eval mode, so summing gives per-sample gradients.
                score.sum().backward()
        finally:
            self.model.train(was_training)
        if self._activations is None or self._gradients is None:
            raise RuntimeError(
                f"Layer {self.target_layer!r} was not on the forward path; check gradcam_layer."
            )
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self._activations).sum(dim=1))
        if upsample:
            cam = F.interpolate(
                cam.unsqueeze(1), size=x.shape[-2:], mode="bilinear", align_corners=False
            ).squeeze(1)
        # Divide by the true peak, however small: raw CAM magnitudes depend on gradient scale
        # (about 1e-14 for a randomly initialised EfficientNet), so a fixed epsilon would
        # silently skip the normalisation.
        peak = cam.flatten(1).amax(dim=1).clamp_min(torch.finfo(cam.dtype).tiny).view(-1, 1, 1)
        return logits.detach(), cam / peak

    def __call__(
        self, x: torch.Tensor, target_class: int = 1, upsample: bool = True
    ) -> torch.Tensor:
        return self.logits_and_cam(x, target_class, upsample)[1]


def overlay_heatmap(image: np.ndarray, cam: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """Blend a [0,1] CAM over an RGB uint8 image; returns an RGB uint8 array.

    Opacity scales with the CAM value, so regions the model ignored show the untouched
    X-ray and only the evidence is painted over it.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image must be (H,W,3), got {image.shape}")
    if cam.shape != image.shape[:2]:
        raise ValueError(f"cam shape {cam.shape} does not match image {image.shape[:2]}")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    colour = colormaps[HEATMAP_CMAP](np.clip(cam, 0, 1))[..., :3] * 255.0
    weight = (alpha * np.clip(cam, 0, 1))[..., None]
    blended = (1 - weight) * image.astype(np.float64) + weight * colour
    return np.clip(blended, 0, 255).astype(np.uint8)
