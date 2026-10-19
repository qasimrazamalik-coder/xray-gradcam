import pytest
import torch

from xray_gradcam.models import (
    build_model,
    get_module,
    head_parameters,
    load_checkpoint,
    save_checkpoint,
    set_backbone_trainable,
    set_train_mode,
)


@pytest.mark.parametrize("arch", ["resnet50", "efficientnet_b0"])
def test_head_outputs_one_logit(arch):
    model = build_model(arch, pretrained=False).eval()
    assert model(torch.zeros(2, 3, 64, 64)).shape == (2, 1)


def test_unknown_arch_rejected():
    with pytest.raises(ValueError, match="Unknown arch"):
        build_model("vgg16", pretrained=False)


@pytest.mark.parametrize("arch", ["resnet50", "efficientnet_b0"])
def test_freezing_leaves_only_the_head_trainable(arch):
    model = build_model(arch, pretrained=False)
    set_backbone_trainable(model, arch, False)
    head_ids = {id(p) for p in head_parameters(model, arch)}
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    assert trainable == head_ids
    set_backbone_trainable(model, arch, True)
    assert all(p.requires_grad for p in model.parameters())


def test_frozen_backbone_batchnorm_stays_in_eval():
    model = build_model("resnet50", pretrained=False)
    set_train_mode(model, "resnet50", backbone_frozen=True)
    assert not model.layer1.training
    assert model.fc.training
    before = model.bn1.running_mean.clone()
    model(torch.randn(2, 3, 64, 64))
    assert torch.equal(before, model.bn1.running_mean)


def test_get_module_unknown_layer_lists_options():
    with pytest.raises(ValueError, match="Top-level modules"):
        get_module(build_model("resnet50", pretrained=False), "layer9")


def test_checkpoint_round_trip(tmp_path):
    model = build_model("efficientnet_b0", pretrained=False).eval()
    meta = {"arch": "efficientnet_b0", "dropout": 0.3, "image_size": 32, "threshold": 0.4}
    save_checkpoint(tmp_path / "m.pt", model, meta)
    loaded, loaded_meta = load_checkpoint(tmp_path / "m.pt")
    x = torch.randn(1, 3, 32, 32)
    assert torch.allclose(model(x), loaded(x))
    assert loaded_meta["threshold"] == 0.4


def test_missing_checkpoint_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="make train"):
        load_checkpoint(tmp_path / "nope.pt")
