import numpy as np
import pytest

from xray_gradcam import metrics

Y = np.array([0, 0, 1, 1])
P = np.array([0.1, 0.4, 0.35, 0.8])


def test_auroc_and_ap_hand_computed():
    # 3 of the 4 (negative, positive) pairs are ranked correctly -> 0.75.
    assert metrics.auroc(Y, P) == pytest.approx(0.75)
    # Ranked: 0.8(+) 0.4(-) 0.35(+) 0.1(-): precision@1 = 1, precision@3 = 2/3 -> (1 + 2/3) / 2.
    assert metrics.average_precision(Y, P) == pytest.approx(5 / 6)


def test_confusion_matrix_sensitivity_specificity():
    cm = metrics.confusion_matrix_at(Y, P, threshold=0.5)
    assert cm == {"tn": 2, "fp": 0, "fn": 1, "tp": 1}
    sens, spec = metrics.sensitivity_specificity(cm)
    assert (sens, spec) == (0.5, 1.0)


def test_threshold_is_inclusive():
    assert metrics.confusion_matrix_at(Y, P, threshold=0.35)["tp"] == 2


def test_youden_threshold_on_separable_data():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.9])
    thr = metrics.youden_threshold(y, p)
    assert thr == pytest.approx(0.6)
    cm = metrics.confusion_matrix_at(y, p, thr)
    assert metrics.sensitivity_specificity(cm) == (1.0, 1.0)


def test_ece_hand_computed():
    y = np.array([1, 1, 0, 0])
    # Confident and right: bins 0.9 (observed 1.0) and 0.1 (observed 0.0) are each off by 0.1.
    assert metrics.expected_calibration_error(y, np.array([0.9, 0.9, 0.1, 0.1])) == pytest.approx(
        0.1
    )
    # Over-confident and wrong on half: one bin at 0.9 with observed 0.5 -> |0.5 - 0.9|.
    y2 = np.array([1, 0])
    assert metrics.expected_calibration_error(y2, np.array([0.9, 0.9])) == pytest.approx(0.4)


def test_ece_handles_probability_one():
    assert metrics.expected_calibration_error(np.array([1, 0]), np.array([1.0, 0.0])) == 0.0


def test_bootstrap_ci_is_reproducible_and_brackets_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = np.clip(y * 0.3 + rng.normal(0.35, 0.2, 200), 0, 1)
    a = metrics.bootstrap_auroc_ci(y, p, 200, seed=3)
    b = metrics.bootstrap_auroc_ci(y, p, 200, seed=3)
    assert a == b
    assert a["low"] <= metrics.auroc(y, p) <= a["high"]


def test_bootstrap_ci_degenerate_when_perfect():
    ci = metrics.bootstrap_auroc_ci(
        np.array([0, 0, 1, 1] * 5), np.array([0.1, 0.2, 0.8, 0.9] * 5), 50
    )
    assert ci["low"] == ci["high"] == 1.0


def test_single_class_fails_loudly():
    with pytest.raises(ValueError, match="Both classes"):
        metrics.auroc(np.ones(4), np.linspace(0, 1, 4))


def test_reliability_plot_written(tmp_path):
    metrics.plot_reliability(Y, P, tmp_path / "rel.png", n_bins=5)
    assert (tmp_path / "rel.png").stat().st_size > 0
