"""Final evaluation on the untouched official test folder, using the validation threshold."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from xray_gradcam import metrics
from xray_gradcam.data import CLASS_NAMES, make_loader, prepare_data
from xray_gradcam.models import load_checkpoint
from xray_gradcam.predict import collect_probabilities
from xray_gradcam.utils import get_device, limit_cpu_threads, load_config, run_metadata, write_json


def score_split(
    labels: np.ndarray, probs: np.ndarray, threshold: float, cfg: dict[str, Any]
) -> dict[str, Any]:
    """All headline metrics for one split at a fixed, externally chosen threshold."""
    ev = cfg["evaluation"]
    cm = metrics.confusion_matrix_at(labels, probs, threshold)
    sens, spec = metrics.sensitivity_specificity(cm)
    return {
        "n": len(labels),
        "auroc": metrics.auroc(labels, probs),
        "auroc_ci95": metrics.bootstrap_auroc_ci(
            labels, probs, ev["bootstrap_resamples"], seed=cfg["seed"]
        ),
        "average_precision": metrics.average_precision(labels, probs),
        "threshold": threshold,
        "sensitivity": sens,
        "specificity": spec,
        "confusion_matrix": cm,
        "ece": metrics.expected_calibration_error(labels, probs, ev["ece_bins"]),
    }


def render_table(arch: str, test: dict[str, Any], synthetic: bool) -> str:
    """Markdown row block that the README table is copied from. Numbers come from the JSON."""
    ci = test["auroc_ci95"]
    banner = "SYNTHETIC DATA, NOT A RESULT\n\n" if synthetic else ""
    return (
        f"{banner}| Model | AUROC (95% CI) | Avg. precision | Sensitivity | Specificity | ECE |\n"
        "| --- | --- | --- | --- | --- | --- |\n"
        f"| {arch} | {test['auroc']:.3f} ({ci['low']:.3f}-{ci['high']:.3f}) "
        f"| {test['average_precision']:.3f} | {test['sensitivity']:.3f} "
        f"| {test['specificity']:.3f} | {test['ece']:.3f} |\n"
    )


def main(config_path: str) -> dict[str, Any]:
    cfg = load_config(config_path)
    limit_cpu_threads(2)
    device = get_device()
    model, meta = load_checkpoint(cfg["paths"]["checkpoint"], device)
    if meta["arch"] != cfg["model"]["arch"]:
        raise ValueError(
            f"Checkpoint arch {meta['arch']} does not match config {cfg['model']['arch']}"
        )

    bundle = prepare_data(cfg)
    d = cfg["data"]
    loader_args = {
        "image_size": d["image_size"],
        "batch_size": d["batch_size"],
        "num_workers": d["num_workers"],
        "train": False,
        "seed": cfg["seed"],
    }
    val_y, val_p = collect_probabilities(model, make_loader(bundle.val, **loader_args), device)
    test_y, test_p = collect_probabilities(model, make_loader(bundle.test, **loader_args), device)

    threshold = float(meta["threshold"])
    results = {
        **run_metadata(cfg),
        "arch": meta["arch"],
        "threshold_source": meta["threshold_rule"],
        "classes": list(CLASS_NAMES),
        "val": score_split(val_y, val_p, threshold, cfg),
        "test": score_split(test_y, test_p, threshold, cfg),
        "data": bundle.info,
    }
    out = Path(cfg["paths"]["results_dir"])
    write_json(out / "metrics.json", results)
    metrics.plot_reliability(
        test_y,
        test_p,
        out / "reliability.png",
        cfg["evaluation"]["ece_bins"],
        title=f"{meta['arch']}, test set" + (" (SYNTHETIC)" if results["synthetic_data"] else ""),
    )
    table = render_table(meta["arch"], results["test"], results["synthetic_data"])
    (out / "results_table.md").write_text(table)
    print(table)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    main(parser.parse_args().config)
