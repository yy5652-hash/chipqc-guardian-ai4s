"""Group-safe baseline, independent calibration, and held-out evaluation."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, log_loss, roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLUMNS


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_feature_table(frame: pd.DataFrame) -> None:
    required = {"image_id", "group_id", "cell_type", "label_good", *FEATURE_COLUMNS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Feature table is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Feature table is empty; prepare actual images first")
    if frame["image_id"].isna().any() or frame["image_id"].duplicated().any():
        raise ValueError("Image IDs must be present and unique")
    expected_groups = frame["image_id"].astype(str).str.slice(0, 6)
    if not expected_groups.equals(frame["group_id"].astype(str)):
        raise ValueError("group_id must equal the first six characters of image_id")
    if not frame["label_good"].isin([0, 1]).all():
        raise ValueError("Labels must be 0=bad or 1=good")
    if frame["group_id"].nunique() < 5:
        raise ValueError("At least five independent imageID groups are required")
    values = frame.loc[:, list(FEATURE_COLUMNS)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError("Feature table contains infinite values")
    if np.isnan(values).all(axis=0).any():
        raise ValueError("At least one feature is completely missing")


def _split_score(frame: pd.DataFrame, indices: tuple[np.ndarray, np.ndarray, np.ndarray]) -> float:
    global_prevalence = float(frame["label_good"].mean())
    global_cells = frame["cell_type"].value_counts(normalize=True)
    score = 0.0
    for idx, target in zip(indices, (0.6, 0.2, 0.2)):
        part = frame.iloc[idx]
        if part["label_good"].nunique() < 2:
            return float("inf")
        score += 3 * abs(len(part) / len(frame) - target)
        score += abs(float(part["label_good"].mean()) - global_prevalence)
        part_cells = part["cell_type"].value_counts(normalize=True)
        score += 0.25 * float((part_cells.reindex(global_cells.index, fill_value=0) - global_cells).abs().sum())
    return score


def group_train_cal_test_split(frame: pd.DataFrame, seed: int = 42) -> dict[str, np.ndarray]:
    """Choose deterministic 60/20/20 group-disjoint splits.

    Candidate selection uses only group, label, and cell-type balance. The test
    images are never used for feature/model/threshold fitting.
    """
    validate_feature_table(frame)
    outer = GroupShuffleSplit(n_splits=256, test_size=0.2, random_state=seed)
    best: tuple[float, tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None
    for trial, (train_cal, test) in enumerate(outer.split(frame, frame["label_good"], frame["group_id"])):
        remainder = frame.iloc[train_cal]
        inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + trial)
        local_train, local_cal = next(inner.split(remainder, remainder["label_good"], remainder["group_id"]))
        indices = (train_cal[local_train], train_cal[local_cal], test)
        score = _split_score(frame, indices)
        if np.isfinite(score) and (best is None or score < best[0]):
            best = (score, indices)
    if best is None:
        raise ValueError("Could not form three group-disjoint splits containing both labels")
    train, calibration, test = best[1]
    sets = [set(frame.iloc[idx]["group_id"]) for idx in (train, calibration, test)]
    if any(sets[i] & sets[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise AssertionError("Group leakage detected in selected split")
    return {"train": train, "calibration": calibration, "test": test}


def fit_baseline(frame: pd.DataFrame, seed: int = 42, confidence_threshold: float = 0.8) -> dict:
    if not 0.5 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0.5 and 1.0")
    splits = group_train_cal_test_split(frame, seed)
    train = frame.iloc[splits["train"]]
    calibration = frame.iloc[splits["calibration"]]
    base = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed),
    )
    base.fit(train.loc[:, list(FEATURE_COLUMNS)], train["label_good"].astype(int))
    calibration_scores = base.decision_function(calibration.loc[:, list(FEATURE_COLUMNS)]).reshape(-1, 1)
    calibrator = LogisticRegression(max_iter=1000, random_state=seed)
    calibrator.fit(calibration_scores, calibration["label_good"].astype(int))
    return {
        "base_model": base,
        "calibrator": calibrator,
        "feature_columns": list(FEATURE_COLUMNS),
        "feature_names": list(FEATURE_COLUMNS),
        "classes": {0: "bad", 1: "good"},
        "confidence_threshold": float(confidence_threshold),
        "threshold_status": "demonstration default; not a validated risk threshold",
        "seed": int(seed),
        "split_groups": {
            name: sorted(frame.iloc[idx]["group_id"].astype(str).unique().tolist())
            for name, idx in splits.items()
        },
        "split_image_ids": {
            name: sorted(frame.iloc[idx]["image_id"].astype(str).tolist())
            for name, idx in splits.items()
        },
        "label_mapping": {"0": "bad", "1": "good"},
        "calibration_method": "Platt logistic calibration fitted on held-out groups",
    }


def select_calibration_threshold(
    artifact: dict,
    frame: pd.DataFrame,
    target_error_rate: float = 0.1,
    min_accepted: int = 20,
) -> dict:
    """Find a calibration-only empirical risk threshold, without guarantees.

    The selected threshold is exploratory: group-correlated images make the
    observed error rate optimistic as a population risk guarantee. It must not
    be called a validated deployment threshold without new external data.
    """
    if not 0 <= target_error_rate < 0.5 or min_accepted < 1:
        raise ValueError("target_error_rate must be in [0,0.5), and min_accepted positive")
    calibration_ids = set(artifact["split_image_ids"]["calibration"])
    part = frame.loc[frame["image_id"].astype(str).isin(calibration_ids)]
    if len(part) != len(calibration_ids):
        raise ValueError("Calibration feature rows are incomplete")
    probabilities = predict_good_probability(artifact, part)
    y = part["label_good"].to_numpy(dtype=int)
    confidence = np.maximum(probabilities, 1 - probabilities)
    candidates = np.unique(np.concatenate(([0.5, 1.0], confidence)))
    for threshold in candidates:
        accepted = confidence >= threshold
        n = int(accepted.sum())
        if n < min_accepted:
            continue
        errors = int(np.sum((probabilities[accepted] >= 0.5).astype(int) != y[accepted]))
        if errors / n <= target_error_rate:
            return {
                "threshold": float(threshold),
                "accepted": n,
                "total": int(len(part)),
                "observed_error_rate": float(errors / n),
                "target_error_rate": float(target_error_rate),
                "status": "exploratory calibration-set estimate; no out-of-sample risk guarantee",
            }
    return {
        "threshold": None,
        "accepted": 0,
        "total": int(len(part)),
        "observed_error_rate": None,
        "target_error_rate": float(target_error_rate),
        "status": "no calibration threshold met the empirical target and minimum coverage",
    }


def predict_good_probability(artifact: dict, frame: pd.DataFrame) -> np.ndarray:
    columns = artifact["feature_columns"]
    scores = artifact["base_model"].decision_function(frame.loc[:, columns]).reshape(-1, 1)
    probabilities = artifact["calibrator"].predict_proba(scores)[:, 1]
    return np.asarray(probabilities, dtype=float)


def selective_decision(probability_good: np.ndarray, threshold: float) -> np.ndarray:
    """Return 1=good, 0=bad, -1=human review (abstain)."""
    p = np.asarray(probability_good, dtype=float)
    if np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("Probabilities must be finite and lie in [0,1]")
    result = np.full(p.shape, -1, dtype=int)
    result[p >= threshold] = 1
    result[(p < 0.5) & (1 - p >= threshold)] = 0
    return result


def _ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for index in range(bins):
        in_bin = (p >= edges[index]) & (p < edges[index + 1] if index < bins - 1 else p <= 1)
        if in_bin.any():
            total += float(in_bin.mean()) * abs(float(y[in_bin].mean()) - float(p[in_bin].mean()))
    return total


def _metric_block(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    predicted = (p >= 0.5).astype(int)
    decision = selective_decision(p, threshold)
    accepted = decision != -1
    block = {
        "n_images": int(len(y)),
        "good_count": int(y.sum()),
        "bad_count": int(len(y) - y.sum()),
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)) if len(np.unique(y)) == 2 else None,
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "average_precision_good": float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else None,
        "brier_score": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "ece_10_bins": _ece(y, p),
        "confusion_matrix_rows_bad_good": confusion_matrix(y, predicted, labels=[0, 1]).astype(int).tolist(),
        "selective": {
            "threshold": threshold,
            "accepted_images": int(accepted.sum()),
            "review_images": int((~accepted).sum()),
            "coverage": float(accepted.mean()),
            "accepted_accuracy": float(accuracy_score(y[accepted], decision[accepted])) if accepted.any() else None,
        },
    }
    return block


def evaluate_frame(artifact: dict, frame: pd.DataFrame, split: str = "test") -> dict:
    if split not in artifact["split_image_ids"]:
        raise ValueError(f"Unknown split: {split}")
    ids = set(artifact["split_image_ids"][split])
    subset = frame.loc[frame["image_id"].astype(str).isin(ids)].copy()
    if len(subset) != len(ids):
        raise ValueError(f"Feature table is missing {len(ids) - len(subset)} images from the {split} split")
    y = subset["label_good"].to_numpy(dtype=int)
    p = predict_good_probability(artifact, subset)
    threshold = float(artifact["confidence_threshold"])
    report = {
        "split": split,
        "group_count": int(subset["group_id"].nunique()),
        "overall": _metric_block(y, p, threshold),
        "by_cell_type": {},
    }
    for cell, part in subset.groupby("cell_type", sort=True):
        positions = subset.index.get_indexer(part.index)
        report["by_cell_type"][str(cell)] = _metric_block(y[positions], p[positions], threshold)
    group_accuracies = []
    for _, part in subset.groupby("group_id"):
        positions = subset.index.get_indexer(part.index)
        group_accuracies.append(float(accuracy_score(y[positions], (p[positions] >= 0.5).astype(int))))
    report["macro_group_accuracy"] = float(np.mean(group_accuracies))
    return report


def split_summary(artifact: dict, frame: pd.DataFrame) -> dict:
    report = {}
    for split, ids in artifact["split_image_ids"].items():
        part = frame.loc[frame["image_id"].astype(str).isin(ids)]
        report[split] = {
            "images": int(len(part)),
            "groups": int(part["group_id"].nunique()),
            "good": int(part["label_good"].sum()),
            "bad": int(len(part) - part["label_good"].sum()),
            "cell_type_counts": {str(key): int(value) for key, value in Counter(part["cell_type"]).items()},
        }
    return report
