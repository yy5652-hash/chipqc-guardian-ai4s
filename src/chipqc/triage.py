"""Risk-controlled triage: which frames may be decided without a person?

Given calibrated P(good), a pass threshold is chosen on training dates so that the share of bad frames among
automatically passed frames stays under a target, with a safety margin. Everything else goes to review.
(A fail threshold is computed the same way and is used only to measure what automatic rejection would cost.)
"""
from __future__ import annotations

import numpy as np

PASS, REVIEW, FAIL = 1, 0, -1


Z_CONSERVATIVE = 1.2816          # one-sided 90 % Wilson bound: the default safety margin


def wilson_upper(errors: np.ndarray, n: np.ndarray, z: float) -> np.ndarray:
    """Upper confidence bound on an error rate observed as errors / n (Wilson score interval)."""
    if z == 0:
        return errors / n
    phat = errors / n
    return (phat + z * z / (2 * n) + z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))) / (1 + z * z / n)


def choose_thresholds(p: np.ndarray, y: np.ndarray, target_error: float, n_min: int = 30, z: float = Z_CONSERVATIVE) -> tuple[float, float]:
    """Largest PASS set (p >= t_pass) and FAIL set (p <= t_fail) whose error bound is <= target_error.

    The bound is the one-sided Wilson upper limit of the observed error, not the observed error itself: picking
    the largest set that just meets a target on the fitting data is optimistic, and the margin pays for that."""
    n = np.arange(1, len(p) + 1)
    order = np.argsort(-p)
    ok = np.flatnonzero((wilson_upper(np.cumsum(1 - y[order]), n, z) <= target_error) & (n >= n_min))
    t_pass = float(p[order][ok[-1]]) if len(ok) else 1.01
    order = np.argsort(p)
    ok = np.flatnonzero((wilson_upper(np.cumsum(y[order]), n, z) <= target_error) & (n >= n_min))
    t_fail = float(p[order][ok[-1]]) if len(ok) else -0.01
    return t_fail, t_pass


def apply_thresholds(p: np.ndarray, t_fail: float, t_pass: float) -> np.ndarray:
    return np.where(p >= t_pass, PASS, np.where(p <= t_fail, FAIL, REVIEW))


def triage_report(decision: np.ndarray, y: np.ndarray) -> dict[str, float]:
    passed, failed = decision == PASS, decision == FAIL
    auto = passed | failed
    wrong = (passed & (y == 0)) | (failed & (y == 1))
    return {
        "auto_decided": float(auto.mean()),
        "error_among_auto": float(wrong[auto].mean()) if auto.any() else 0.0,
        "passed": float(passed.mean()),
        "false_accept": float((passed & (y == 0)).sum() / max(1, passed.sum())),
        "failed": float(failed.mean()),
        "false_reject": float((failed & (y == 1)).sum() / max(1, failed.sum())),
    }
