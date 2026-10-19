"""Tiny end-to-end pass (EfficientNet-B0, 32 px, SYNTHETIC data): train, evaluate, grid."""

import json

from xray_gradcam import evaluate, examples, train


def test_train_evaluate_and_grid(tiny_cfg, tmp_path):
    import yaml

    tiny_cfg["model"].update({"arch": "efficientnet_b0", "gradcam_layer": "features.8"})
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(tiny_cfg))

    summary = train.main(str(cfg_path))
    assert summary["synthetic_data"] is True
    assert summary["data"]["official_val_folder_used"] is False

    results = evaluate.main(str(cfg_path))
    assert 0.0 <= results["test"]["auroc"] <= 1.0
    assert results["threshold_source"].startswith("Youden")
    out = tmp_path / "results"
    metrics_json = json.loads((out / "metrics.json").read_text())
    assert metrics_json["seed"] == 42 and "torch" in metrics_json
    assert (out / "reliability.png").exists()
    assert "SYNTHETIC" in (out / "results_table.md").read_text()

    assert examples.main(str(cfg_path)).exists()
