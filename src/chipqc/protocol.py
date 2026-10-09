"""Leakage-aware evaluation.

Every frame is scored by a model that never saw its acquisition date. Outer loop: five folds over the dates,
repeated with different fold assignments. Inner loop, on the training dates only: choose the regularisation
strength, fit the probability calibrator and choose the triage thresholds. Confidence intervals resample whole
dates, because frames from one date are not independent observations.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from .triage import apply_thresholds, choose_thresholds, triage_report

C_GRID = (0.001, 0.003, 0.01, 0.03, 0.1)
NORM = "l2"                                       # the released head: unit-length embedding, then standardisation
ERROR_TARGETS = (0.05, 0.10, 0.15, 0.20)          # evaluated and reported
OFFERED_TARGETS = (0.10, 0.15, 0.20)              # what a released model offers: the 5 % target was not met on held-out dates


def normalise(X: np.ndarray, norm: str | None) -> np.ndarray:
    """`l2`: every frame embedding scaled to unit length. The scale is one number per frame, so each patch's share
    of the score is scaled equally and the evidence map stays exact."""
    return X / np.linalg.norm(X, axis=1, keepdims=True) if norm == "l2" else X


def fit_head(X: np.ndarray, y: np.ndarray, C: float, norm: str | None = NORM):
    Xn = normalise(X, norm)
    scaler = StandardScaler().fit(Xn)
    return norm, scaler, LogisticRegression(C=C, max_iter=5000).fit(scaler.transform(Xn), y)


def head_logits(head, X: np.ndarray) -> np.ndarray:
    norm, scaler, clf = head
    return clf.decision_function(scaler.transform(normalise(X, norm)))


def date_folds(dates: np.ndarray, k: int, seed: int):
    unique = np.unique(dates)
    order = dict(zip(unique, np.random.default_rng(seed).permutation(len(unique))))
    key = np.array([order[d] for d in dates])
    return list(GroupKFold(n_splits=k).split(key, key, key))


def fit_platt(z: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    clf = LogisticRegression(C=1e6, max_iter=1000).fit(z[:, None], y)
    return float(clf.coef_[0, 0]), float(clf.intercept_[0])


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def select_on_training_dates(X, y, dates, seed, inner=4, c_grid=C_GRID, norm=NORM):
    """Inner date-grouped CV: returns the chosen C and the out-of-fold logits it produced on the training dates."""
    z = {c: np.zeros(len(y)) for c in c_grid}
    for a, b in date_folds(dates, inner, seed):
        for c in c_grid:
            z[c][b] = head_logits(fit_head(X[a], y[a], c, norm), X[b])
    best = max(c_grid, key=lambda c: roc_auc_score(y, z[c]))
    return best, z[best]


def nested_evaluation(X, y, dates, repeats=3, outer=5, inner=4, c_grid=C_GRID, targets=ERROR_TARGETS, eligible=None, norm=NORM):
    """Returns out-of-fold raw and calibrated P(good) averaged over repeats, the chosen C values, per-repeat triage
    outcomes, and the per-frame decisions (target -> repeats x frames array of PASS / REVIEW / FAIL codes).

    `eligible` (repeats x frames, bool) restricts threshold selection to frames that passed the acquisition gate, so
    the error targets apply to the frames the model is actually allowed to decide."""
    n = len(y)
    raw, cal = np.zeros((repeats, n)), np.zeros((repeats, n))
    chosen, outcomes = [], {t: [] for t in targets}
    decisions = {t: np.zeros((repeats, n), dtype=int) for t in targets}
    for r in range(repeats):
        decision = {t: decisions[t][r] for t in targets}
        for tr, te in date_folds(dates, outer, r):
            c, z_in = select_on_training_dates(X[tr], y[tr], dates[tr], 100 + r, inner, c_grid, norm)
            a, b = fit_platt(z_in, y[tr])
            z = head_logits(fit_head(X[tr], y[tr], c, norm), X[te])
            raw[r, te], cal[r, te] = sigmoid(z), sigmoid(a * z + b)
            chosen.append(c)
            keep = eligible[r][tr] if eligible is not None else np.ones(len(tr), dtype=bool)
            for t in targets:
                decision[t][te] = apply_thresholds(cal[r, te], *choose_thresholds(sigmoid(a * z_in + b)[keep], y[tr][keep], t))
        for t in targets:
            outcomes[t].append(triage_report(decision[t], y))
    return raw.mean(0), cal.mean(0), chosen, outcomes, decisions


def date_bootstrap(metric, y, p, dates, n=2000, seed=0):
    unique = np.unique(dates)
    rows = {d: np.flatnonzero(dates == d) for d in unique}
    rng, values = np.random.default_rng(seed), []
    for _ in range(n):
        idx = np.concatenate([rows[d] for d in rng.choice(unique, len(unique))])
        if len(np.unique(y[idx])) == 2:
            values.append(metric(y[idx], p[idx]))
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def paired_date_bootstrap(metric, y, p_a, p_b, dates, n=2000, seed=0):
    """Interval for metric(p_a) - metric(p_b), both scored on the same resampled acquisition dates."""
    unique = np.unique(dates)
    rows = {d: np.flatnonzero(dates == d) for d in unique}
    rng, values = np.random.default_rng(seed), []
    for _ in range(n):
        idx = np.concatenate([rows[d] for d in rng.choice(unique, len(unique))])
        if len(np.unique(y[idx])) == 2:
            values.append(metric(y[idx], p_a[idx]) - metric(y[idx], p_b[idx]))
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def expected_calibration_error(y, p, bins=10):
    edges, total = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


METRICS = {
    "auroc": roc_auc_score,
    "auprc_good": average_precision_score,
    "balanced_accuracy": lambda y, p: balanced_accuracy_score(y, p > 0.5),
    "accuracy": lambda y, p: accuracy_score(y, p > 0.5),
    "brier": brier_score_loss,
}


def summarise(y, p_raw, p, dates, strata: dict[str, np.ndarray] | None = None, n_boot=2000):
    out = {}
    for name, fn in METRICS.items():
        lo, hi = date_bootstrap(fn, y, p, dates, n_boot)
        out[name] = {"value": float(fn(y, p)), "ci95": [lo, hi]}
    out["ece_raw"], out["ece_calibrated"] = expected_calibration_error(y, p_raw), expected_calibration_error(y, p)
    for label, values in (strata or {}).items():
        out[f"by_{label}"] = {str(v): {"n": int((values == v).sum()), "auroc": float(roc_auc_score(y[values == v], p[values == v]))}
                              for v in np.unique(values) if len(np.unique(y[values == v])) == 2}
    return out
