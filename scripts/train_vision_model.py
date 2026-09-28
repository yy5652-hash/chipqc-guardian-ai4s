"""Train and calibrate a group-held-out classifier on MobileNetV2 embeddings."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, brier_score_loss, confusion_matrix,
    log_loss, roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]


def logit_probability(model: ExtraTreesClassifier, matrix: np.ndarray) -> np.ndarray:
    probability = np.clip(model.predict_proba(matrix)[:, 1], 1e-6, 1 - 1e-6)
    return np.log(probability / (1 - probability)).reshape(-1, 1)


def metric_block(y: np.ndarray, probability: np.ndarray, decision_threshold: float) -> dict:
    prediction = (probability >= decision_threshold).astype(int)
    return {
        "n_images": int(len(y)),
        "good_count": int(y.sum()),
        "bad_count": int(len(y) - y.sum()),
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "brier_score": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "confusion_matrix_rows_bad_good": confusion_matrix(
            y, prediction, labels=[0, 1]
        ).astype(int).tolist(),
        "decision_threshold": float(decision_threshold),
    }


def group_bootstrap_interval(
    y: np.ndarray,
    probability: np.ndarray,
    groups: np.ndarray,
    decision_threshold: float,
    seed: int,
    resamples: int = 2000,
) -> dict:
    """Percentile intervals that resample complete held-out groups."""
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    balanced, auc = [], []
    skipped = 0
    for _ in range(resamples):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        positions = np.concatenate([np.flatnonzero(groups == group) for group in sampled])
        labels = y[positions]
        if len(np.unique(labels)) < 2:
            skipped += 1
            continue
        scores = probability[positions]
        balanced.append(balanced_accuracy_score(labels, scores >= decision_threshold))
        auc.append(roc_auc_score(labels, scores))
    return {
        "method": "percentile bootstrap resampling complete six-digit prefix groups",
        "requested_resamples": resamples,
        "valid_resamples": len(balanced),
        "skipped_single_class_resamples": skipped,
        "balanced_accuracy_95_percentile_interval": np.quantile(balanced, [0.025, 0.975]).tolist(),
        "roc_auc_95_percentile_interval": np.quantile(auc, [0.025, 0.975]).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embeddings", type=Path,
        default=ROOT / "data/processed/mobilenet_v2_embeddings.npz",
    )
    parser.add_argument("--baseline", type=Path, default=ROOT / "artifacts/baseline.joblib")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/vision_model.joblib")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/vision_evaluation.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = np.load(args.embeddings)
    matrix = data["embeddings"].astype(np.float32)
    labels = data["label_good"].astype(int)
    image_ids = data["image_id"].astype(str)
    groups = data["group_id"].astype(str)
    cell_types = data["cell_type"].astype(str)
    baseline = joblib.load(args.baseline)
    split_ids = baseline["split_image_ids"]
    indices = {
        name: np.flatnonzero(np.isin(image_ids, np.asarray(ids, dtype=str)))
        for name, ids in split_ids.items()
    }
    if sum(map(len, indices.values())) != len(image_ids):
        parser.error("Embedding rows do not match the frozen baseline split")

    # Candidate choice is based only on the calibration groups. The held-out
    # test groups are evaluated once after selection.
    candidates = ("sqrt", 0.2, 0.5)
    trials: list[dict] = []
    fitted: dict[str, ExtraTreesClassifier] = {}
    for max_features in candidates:
        model = ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=3,
            max_features=max_features,
            class_weight="balanced",
            random_state=args.seed,
            n_jobs=-1,
        )
        model.fit(matrix[indices["train"]], labels[indices["train"]])
        probability = model.predict_proba(matrix[indices["calibration"]])[:, 1]
        prediction = probability >= 0.5
        balanced = balanced_accuracy_score(labels[indices["calibration"]], prediction)
        auc = roc_auc_score(labels[indices["calibration"]], probability)
        name = str(max_features)
        fitted[name] = model
        trials.append({
            "max_features": max_features,
            "calibration_balanced_accuracy_at_0_5": float(balanced),
            "calibration_roc_auc": float(auc),
            "selection_score": float(balanced + auc),
        })
    selected = max(trials, key=lambda row: row["selection_score"])
    model = fitted[str(selected["max_features"])]

    calibration_index = indices["calibration"]
    calibrator = LogisticRegression(max_iter=1000, random_state=args.seed)
    calibrator.fit(logit_probability(model, matrix[calibration_index]), labels[calibration_index])
    calibration_probability = calibrator.predict_proba(
        logit_probability(model, matrix[calibration_index])
    )[:, 1]
    thresholds = np.linspace(0.1, 0.9, 161)
    scores = [
        balanced_accuracy_score(labels[calibration_index], calibration_probability >= threshold)
        for threshold in thresholds
    ]
    decision_threshold = float(thresholds[int(np.argmax(scores))])

    artifact = {
        "kind": "mobilenet_v2_embedding_extratrees",
        "embedding_backbone": str(data["backbone"]),
        "model": model,
        "calibrator": calibrator,
        "decision_threshold": decision_threshold,
        "split_image_ids": split_ids,
        "label_mapping": {"0": "bad", "1": "good"},
        "seed": args.seed,
    }
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "internal group-held-out evaluation; not external or clinical validation",
        "embedding_backbone": artifact["embedding_backbone"],
        "split_rule": "all images sharing the first six imageID characters stay together",
        "selection_rule": "maximize calibration balanced accuracy + ROC-AUC; test untouched until selection",
        "candidate_trials": trials,
        "selected": selected,
        "decision_threshold": decision_threshold,
        "split_counts": {
            name: {"images": int(len(index)), "groups": int(len(np.unique(groups[index])))}
            for name, index in indices.items()
        },
    }
    for name in ("calibration", "test"):
        index = indices[name]
        probability = calibrator.predict_proba(logit_probability(model, matrix[index]))[:, 1]
        report[name] = metric_block(labels[index], probability, decision_threshold)
        report[name]["by_cell_type"] = {
            cell: metric_block(
                labels[index][cell_types[index] == cell],
                probability[cell_types[index] == cell],
                decision_threshold,
            ) if len(np.unique(labels[index][cell_types[index] == cell])) == 2 else {
                "n_images": int((cell_types[index] == cell).sum()),
                "note": "single class in this split; class-balanced metrics undefined",
            }
            for cell in sorted(np.unique(cell_types[index]))
        }
        if name == "test":
            report[name]["group_bootstrap"] = group_bootstrap_interval(
                labels[index], probability, groups[index], decision_threshold, args.seed
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.output)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved vision model to {args.output}")
    print(json.dumps(report["test"], indent=2))


if __name__ == "__main__":
    main()
