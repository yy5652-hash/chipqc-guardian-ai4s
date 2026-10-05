"""Unit tests that need no images and no network: the arithmetic the report relies on."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from chipqc.data import acquisition_date  # noqa: E402
from chipqc.degrade import degrade  # noqa: E402
from chipqc.descriptors import acquisition_descriptors, acquisition_flags  # noqa: E402
from chipqc.guardian import Guardian  # noqa: E402
from chipqc.protocol import date_folds, fit_head, head_logits, nested_evaluation, sigmoid  # noqa: E402
from chipqc.triage import PASS, REVIEW, apply_thresholds, choose_thresholds  # noqa: E402

LIMITS = {"occluded_block_frac_max": 0.25, "streak_anisotropy_max": 0.37, "log_sharpness_min": -3.4, "median_brightness_min": 0.25}


def textured(seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(0.55 + 0.18 * rng.standard_normal((768, 1024)), 0, 1).astype(np.float32)


def test_acquisition_date_is_the_first_six_characters():
    assert acquisition_date("230524_35") == "230524"


def test_date_folds_never_split_a_date():
    dates = np.repeat([f"d{i}" for i in range(20)], 7)
    for train, test in date_folds(dates, 5, seed=3):
        assert not set(dates[train]) & set(dates[test])


def test_clean_textured_frame_raises_no_flag():
    assert acquisition_flags(acquisition_descriptors(textured()), LIMITS) == []


@pytest.mark.parametrize("fault,level,word", [("occlusion", 0.5, "blacked out"), ("motion_streak", 40, "motion"), ("under_exposure", 0.2, "under-exposed")])
def test_controlled_faults_are_flagged(fault, level, word):
    flags = acquisition_flags(acquisition_descriptors(degrade(textured(), fault, level, np.random.default_rng(1))), LIMITS)
    assert any(word in f for f in flags), flags


def test_defocus_lowers_sharpness():
    sharp = acquisition_descriptors(textured())["log_sharpness"]
    blurred = acquisition_descriptors(degrade(textured(), "defocus", 4, np.random.default_rng(0)))["log_sharpness"]
    assert blurred < sharp - 1.0


def test_pass_threshold_meets_its_target_on_the_data_it_was_fitted_on():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 4000)
    p = np.clip(0.5 + 0.25 * (2 * y - 1) + 0.25 * rng.standard_normal(4000), 0, 1)
    _, t_pass = choose_thresholds(p, y, 0.10)
    passed = apply_thresholds(p, -1.0, t_pass) == PASS
    assert passed.sum() >= 30 and (1 - y[passed]).mean() <= 0.10
    assert (apply_thresholds(np.array([t_pass - 1e-6]), -1.0, t_pass) == REVIEW).all()


def test_no_threshold_when_the_target_is_unreachable():
    rng = np.random.default_rng(0)
    y, p = rng.integers(0, 2, 500), rng.random(500)            # scores carry no information
    assert choose_thresholds(p, y, 0.01)[1] > 1.0


def make_model(tmp_path, X, y, C=0.1):
    scaler, clf = fit_head(X, y, C)
    w, mu, sd = clf.coef_[0], scaler.mean_, scaler.scale_
    spec = {"name": "test", "created_utc": "2026-01-01T00:00:00Z", "backbone": "dinov2_vits14", "cell_weights": list(w / sd),
            "cell_bias": float(clf.intercept_[0] - (w * mu / sd).sum()), "platt": [0.8, 0.1], "thresholds": {"0.1": {"t_pass": 0.8}},
            "default_error_target": 0.1, "acquisition_limits": LIMITS, "embedding_mean": list(mu), "embedding_std": list(sd)}
    (tmp_path / "model.json").write_text(json.dumps(spec))
    return Guardian(tmp_path), (scaler, clf)


def test_evidence_map_averages_exactly_to_the_score(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 16)); y = (X[:, 0] + 0.5 * rng.standard_normal(300) > 0).astype(int)
    g, head = make_model(tmp_path, X, y)
    cells = rng.standard_normal((9, 12, 16)).astype(np.float32)                 # a frame as a grid of descriptors
    p, evidence = g.score_cells(cells)
    assert evidence.shape == (9, 12)
    assert p == pytest.approx(float(sigmoid(evidence.mean())), abs=1e-6)
    # and the score is the calibrated logistic regression on the mean descriptor
    logit = head_logits(head, cells.reshape(-1, 16).mean(0, keepdims=True).astype(np.float64))[0]
    assert p == pytest.approx(float(sigmoid(0.8 * logit + 0.1)), abs=1e-5)


def test_nested_protocol_scores_every_frame_once_per_repeat_and_finds_real_signal():
    rng = np.random.default_rng(0)
    dates = np.repeat([f"d{i:02d}" for i in range(30)], 20)
    y = rng.integers(0, 2, len(dates))
    X = rng.standard_normal((len(dates), 12)); X[:, 0] += 1.5 * (2 * y - 1)
    raw, cal, chosen, outcomes, decisions = nested_evaluation(X, y, dates, repeats=2)
    assert len(chosen) == 10 and ((cal > 0) & (cal < 1)).all()
    assert decisions[0.1].shape == (2, len(y))
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y, cal) > 0.9
    assert outcomes[0.1][0]["false_accept"] < 0.2
