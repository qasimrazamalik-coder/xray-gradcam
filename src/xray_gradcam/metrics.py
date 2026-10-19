"""Metric functions used for model selection and the final report.

Everything here takes plain numpy arrays: ``y_true`` in {0,1} and ``y_prob`` as the
predicted probability of the positive class (PNEUMONIA).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def _check(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.shape != y_prob.shape or y_true.ndim != 1 or y_true.size == 0:
        raise ValueError("y_true and y_prob must be non-empty 1-D arrays of equal length")
    if len(np.unique(y_true)) < 2:
        raise ValueError("Both classes must be present to compute this metric")
    return y_true, y_prob


def auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true, y_prob = _check(y_true, y_prob)
    return float(roc_auc_score(y_true, y_prob))


def average_precision(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true, y_prob = _check(y_true, y_prob)
    return float(average_precision_score(y_true, y_prob))


def youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Threshold maximising sensitivity + specificity - 1.

    Call this on the validation split only. Choosing the threshold on test data would make
    the reported sensitivity/specificity optimistic.
    """
    y_true, y_prob = _check(y_true, y_prob)
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    best = int(np.argmax(tpr - fpr))
    # roc_curve prepends an infinite threshold for the "predict nothing" point.
    return float(thresholds[best]) if np.isfinite(thresholds[best]) else float(y_prob.max() + 1e-6)


def confusion_matrix_at(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, int]:
    """Counts with prediction = (prob >= threshold). Positive class is PNEUMONIA."""
    y_true, y_prob = _check(y_true, y_prob)
    pred = y_prob >= threshold
    return {
        "tn": int(np.sum((y_true == 0) & ~pred)),
        "fp": int(np.sum((y_true == 0) & pred)),
        "fn": int(np.sum((y_true == 1) & ~pred)),
        "tp": int(np.sum((y_true == 1) & pred)),
    }


def sensitivity_specificity(cm: dict[str, int]) -> tuple[float, float]:
    """Sensitivity = TP/(TP+FN), specificity = TN/(TN+FP)."""
    return cm["tp"] / (cm["tp"] + cm["fn"]), cm["tn"] / (cm["tn"] + cm["fp"])


def reliability_bins(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> list[dict[str, float]]:
    """Equal-width bins over predicted P(pneumonia): mean confidence, observed rate, count."""
    y_true, y_prob = _check(y_true, y_prob)
    idx = np.minimum((y_prob * n_bins).astype(int), n_bins - 1)
    bins = []
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            bins.append(
                {
                    "bin": b,
                    "confidence": float(y_prob[mask].mean()),
                    "observed": float(y_true[mask].mean()),
                    "count": int(mask.sum()),
                }
            )
    return bins


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """ECE on the positive-class probability: sum over bins of (n_b/N) * |observed - confidence|.

    This is the binary, positive-class variant. The top-label variant would give different
    numbers; the choice is stated in the README so results are comparable.
    """
    bins = reliability_bins(y_true, y_prob, n_bins)
    total = sum(b["count"] for b in bins)
    return float(sum(b["count"] / total * abs(b["observed"] - b["confidence"]) for b in bins))


def bootstrap_auroc_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Percentile bootstrap CI for AUROC, resampling test images with replacement.

    Resamples that happen to contain one class are skipped (and counted) because AUROC is
    undefined there. This CI reflects test-set sampling noise only; it says nothing about
    shift between hospitals or about patient-level correlation.
    """
    y_true, y_prob = _check(y_true, y_prob)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores, skipped = [], 0
    for _ in range(n_resamples):
        take = rng.integers(0, n, n)
        if len(np.unique(y_true[take])) < 2:
            skipped += 1
            continue
        scores.append(roc_auc_score(y_true[take], y_prob[take]))
    low, high = np.percentile(scores, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "low": float(low),
        "high": float(high),
        "level": 1 - alpha,
        "resamples": len(scores),
        "skipped": skipped,
    }


def plot_reliability(
    y_true: np.ndarray, y_prob: np.ndarray, path: str | Path, n_bins: int = 10, title: str = ""
) -> None:
    """Reliability diagram plus a histogram of predictions (so sparse bins are visible)."""
    bins = reliability_bins(y_true, y_prob, n_bins)
    ece = expected_calibration_error(y_true, y_prob, n_bins)
    fig, (ax, ax_hist) = plt.subplots(
        2, 1, figsize=(5, 6), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )
    ax.plot(
        [0, 1], [0, 1], color="#6b665e", linestyle="--", linewidth=1, label="perfect calibration"
    )
    ax.plot(
        [b["confidence"] for b in bins],
        [b["observed"] for b in bins],
        marker="o",
        color="#a8441f",
        label=f"model (ECE {ece:.3f})",
    )
    ax.set_ylabel("observed fraction pneumonia")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", frameon=False)
    ax.set_title(title or "Reliability diagram")
    ax_hist.hist(np.asarray(y_prob), bins=n_bins, range=(0, 1), color="#6b665e")
    ax_hist.set_xlabel("predicted probability of pneumonia")
    ax_hist.set_ylabel("images")
    for a in (ax, ax_hist):
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
