import numpy as np
import pytest
import torch
from torch import nn

from xray_gradcam.gradcam import GradCAM, overlay_heatmap
from xray_gradcam.models import build_model

LAYERS = {"resnet50": "layer4", "efficientnet_b0": "features.8"}


@pytest.mark.parametrize("arch", ["resnet50", "efficientnet_b0"])
def test_cam_shape_range_and_nonnegative(arch):
    torch.manual_seed(0)
    model = build_model(arch, pretrained=False).eval()
    x = torch.randn(3, 3, 64, 64)
    with GradCAM(model, LAYERS[arch]) as cam_engine:
        cam = cam_engine(x)
        small = cam_engine(x, upsample=False)
    assert cam.shape == (3, 64, 64)
    assert small.shape[0] == 3
    assert small.shape[-1] < 64
    assert torch.all(cam >= 0)
    assert torch.all(cam <= 1.0 + 1e-6)
    assert torch.allclose(cam.flatten(1).amax(dim=1), torch.ones(3), atol=1e-5)


def test_cam_matches_hand_computed_result():
    """1x1 conv, global average pool, linear: gradients are uniform, so the CAM is A / max(A)."""

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 1, 1, bias=False)
            self.fc = nn.Linear(1, 1, bias=False)
            nn.init.constant_(self.conv.weight, 1.0)
            nn.init.constant_(self.fc.weight, 2.0)

        def forward(self, x):
            return self.fc(self.conv(x).mean(dim=(2, 3)))

    x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    with GradCAM(Tiny(), "conv") as engine:
        cam = engine(x, upsample=False)
    assert torch.allclose(cam[0], torch.tensor([[0.25, 0.5], [0.75, 1.0]]))


def test_target_class_zero_flips_the_evidence():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 1, 1, bias=False)
            self.fc = nn.Linear(1, 1, bias=False)
            nn.init.constant_(self.conv.weight, 1.0)
            nn.init.constant_(self.fc.weight, 2.0)

        def forward(self, x):
            return self.fc(self.conv(x).mean(dim=(2, 3)))

    x = torch.ones(1, 1, 2, 2)
    with GradCAM(Tiny(), "conv") as engine:
        assert engine(x, target_class=1).sum() > 0
        assert engine(x, target_class=0).sum() == 0  # positive weights give no NORMAL evidence


def test_hooks_removed_and_mode_restored():
    model = build_model("resnet50", pretrained=False)
    model.train()
    with GradCAM(model, "layer4") as engine:
        engine(torch.randn(1, 3, 64, 64))
    assert model.training
    assert not model.layer4._forward_hooks


def test_works_with_frozen_parameters_and_outer_no_grad():
    model = build_model("efficientnet_b0", pretrained=False).eval()
    for p in model.parameters():
        p.requires_grad = False
    with torch.no_grad(), GradCAM(model, "features.8") as engine:
        cam = engine(torch.randn(1, 3, 64, 64))
    assert cam.shape == (1, 64, 64)


def test_bad_inputs_rejected():
    model = build_model("resnet50", pretrained=False)
    with GradCAM(model, "layer4") as engine:
        with pytest.raises(ValueError):
            engine(torch.randn(3, 64, 64))
        with pytest.raises(ValueError):
            engine(torch.randn(1, 3, 64, 64), target_class=2)
    with pytest.raises(ValueError, match="not found"):
        GradCAM(model, "layer9")


def test_overlay_shape_dtype_and_validation():
    image = np.full((16, 16, 3), 128, dtype=np.uint8)
    cam = np.zeros((16, 16))
    cam[4:8, 4:8] = 1.0
    out = overlay_heatmap(image, cam)
    assert out.shape == image.shape and out.dtype == np.uint8
    assert np.array_equal(out[0, 0], image[0, 0])  # zero evidence leaves the pixel untouched
    assert not np.array_equal(out[5, 5], image[5, 5])
    with pytest.raises(ValueError):
        overlay_heatmap(image, np.zeros((8, 8)))
