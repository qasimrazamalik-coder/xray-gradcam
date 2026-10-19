"""Write a grid of correct and incorrect test examples with Grad-CAM overlays.

Examples are drawn at random (seeded) from each outcome group rather than chosen by
confidence or by how good the heatmap looks, so the grid is not cherry-picked.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from xray_gradcam.data import make_loader, prepare_data
from xray_gradcam.models import load_checkpoint
from xray_gradcam.predict import collect_probabilities, explain_image
from xray_gradcam.utils import get_device, limit_cpu_threads, load_config

GROUPS = (
    ("true positive", 1, True),
    ("true negative", 0, True),
    ("false positive", 0, False),
    ("false negative", 1, False),
)


def main(config_path: str) -> Path:
    cfg = load_config(config_path)
    limit_cpu_threads(2)
    device = get_device()
    model, meta = load_checkpoint(cfg["paths"]["checkpoint"], device)
    bundle = prepare_data(cfg)
    rng = np.random.default_rng(cfg["seed"])
    k = cfg["gradcam_grid"]["per_category"]

    # Cheap forward pass over the whole test set; Grad-CAM only for the images we display.
    d = cfg["data"]
    loader = make_loader(
        bundle.test,
        d["image_size"],
        d["batch_size"],
        d["num_workers"],
        train=False,
        seed=cfg["seed"],
    )
    labels, probs = collect_probabilities(model, loader, device)
    flagged = probs >= float(meta["threshold"])

    fig, axes = plt.subplots(len(GROUPS), k, figsize=(2.6 * k, 2.8 * len(GROUPS)), squeeze=False)
    for row, (name, true_label, correct) in enumerate(GROUPS):
        pool = np.flatnonzero((labels == true_label) & ((flagged == bool(true_label)) == correct))
        chosen = rng.permutation(pool)[:k]
        for col in range(k):
            ax = axes[row, col]
            ax.axis("off")
            if col < len(chosen):
                with Image.open(bundle.test[int(chosen[col])][0]) as image:
                    out = explain_image(model, meta, image, device)
                ax.imshow(out["overlay"])
                ax.set_title(f"{name}\np={out['probability']:.2f}", fontsize=8)
            elif col == 0:
                ax.text(0.5, 0.5, f"no {name} examples", ha="center", va="center", fontsize=8)
    suffix = " (SYNTHETIC data)" if cfg["data"]["source"] == "synthetic" else ""
    fig.suptitle(
        f"Grad-CAM on test images, {meta['arch']}, layer {meta['gradcam_layer']}{suffix}",
        fontsize=10,
    )
    fig.tight_layout()
    out_path = Path(cfg["paths"]["results_dir"]) / "gradcam_examples.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Wrote {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    main(parser.parse_args().config)
