"""Fit logistic regression on image features, calibrate, and score held-out groups."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chipqc.modeling import (  # noqa: E402
    evaluate_frame, file_sha256, fit_baseline, select_calibration_threshold, split_summary,
)
from chipqc.metadata import load_metadata, require_trainable_metadata  # noqa: E402


def validate_prepared_data(
    metadata: pd.DataFrame,
    manifest: pd.DataFrame,
    features: pd.DataFrame,
    matching_summary: dict,
    allow_partial: bool,
) -> dict:
    """Verify every source row is accounted for before any fitting occurs."""
    require_trainable_metadata(metadata)
    source_ids = set(metadata["image_id"].astype(str))
    manifest_ids = set(manifest["image_id"].astype(str))
    if manifest["image_id"].duplicated().any() or manifest_ids != source_ids or len(manifest) != len(metadata):
        raise ValueError("Manifest must contain each spreadsheet image ID exactly once")
    if features["image_id"].duplicated().any():
        raise ValueError("Feature table contains duplicate image IDs")
    matched_ids = set(manifest.loc[manifest["status"] == "matched", "image_id"].astype(str))
    if set(features["image_id"].astype(str)) != matched_ids:
        raise ValueError("Feature table image IDs must equal the manifest's matched image IDs")
    if (matching_summary.get("metadata_record_count") != len(metadata)
            or matching_summary.get("matched_feature_count") != len(features)):
        raise ValueError("Matching summary counts disagree with the manifest or feature table")
    actual_status = {str(key): int(value) for key, value in manifest["status"].value_counts().items()}
    if actual_status != matching_summary.get("status_counts"):
        raise ValueError("Matching summary statuses disagree with the manifest")
    source_contract = metadata.set_index("image_id").loc[:, ["group_id", "cell_type", "label_good"]]
    for name, table in (("manifest", manifest), ("features", features)):
        joined = table.set_index("image_id").loc[:, ["group_id", "cell_type", "label_good"]]
        expected = source_contract.loc[joined.index]
        # Compare aligned values, not pandas dtype/index metadata. CSV inputs can
        # use an object index while the Excel loader uses StringDtype even when
        # every image ID and contract value is identical.
        if joined["group_id"].astype(str).tolist() != expected["group_id"].astype(str).tolist():
            raise ValueError(f"{name} group IDs disagree with the spreadsheet")
        if joined["cell_type"].astype(str).tolist() != expected["cell_type"].astype(str).tolist():
            raise ValueError(f"{name} cell types disagree with the spreadsheet")
        if joined["label_good"].astype(int).tolist() != expected["label_good"].astype(int).tolist():
            raise ValueError(f"{name} labels disagree with the spreadsheet")
    subset_only = len(matched_ids) != len(source_ids)
    if subset_only and not allow_partial:
        raise ValueError(
            f"Only {len(matched_ids)}/{len(source_ids)} metadata rows matched images. "
            "Fix image matching or pass --allow-partial to label a subset-only model."
        )
    return {
        "subset_only": subset_only,
        "source_metadata_rows": len(source_ids),
        "matched_feature_rows": len(matched_ids),
        "excluded_metadata_rows": len(source_ids) - len(matched_ids),
        "matching_status_counts": actual_status,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=ROOT / "data/raw/OOC_datasheet.xlsx")
    parser.add_argument("--features", type=Path, default=ROOT / "data/processed/features.csv")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/processed/manifest.csv")
    parser.add_argument("--matching-summary", type=Path, default=ROOT / "data/processed/matching_summary.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--confidence-threshold", type=float, default=0.8)
    parser.add_argument("--calibration-target-error-rate", type=float, default=0.1)
    parser.add_argument("--allow-partial", action="store_true", help="Train on matched rows only and mark results as subset-only")
    args = parser.parse_args()
    for path in (args.metadata, args.features, args.manifest, args.matching_summary):
        if not path.exists():
            parser.error(f"Required input is missing: {path}. Run prepare_dataset.py after the image ZIP completes.")
    frame = pd.read_csv(args.features, dtype={"image_id": str, "group_id": str, "cell_type": str})
    manifest = pd.read_csv(args.manifest, dtype={"image_id": str, "group_id": str, "cell_type": str})
    matching_summary = json.loads(args.matching_summary.read_text(encoding="utf-8"))
    try:
        coverage = validate_prepared_data(
            load_metadata(args.metadata), manifest, frame, matching_summary, args.allow_partial
        )
    except ValueError as exc:
        parser.error(str(exc))
    artifact = fit_baseline(frame, seed=args.seed, confidence_threshold=args.confidence_threshold)
    artifact["feature_csv_sha256"] = file_sha256(args.features)
    report = {
        "model_type": "imputed, standardized logistic regression with independent Platt calibration",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "feature_csv_sha256": artifact["feature_csv_sha256"],
        "feature_columns": artifact["feature_columns"],
        "feature_names": artifact["feature_names"],
        "artifact_schema": {
            "file": "baseline.joblib",
            "root_type": "dict",
            "base_model": "scikit-learn Pipeline: median imputer, standard scaler, LogisticRegression",
            "calibrator": "scikit-learn LogisticRegression on base_model.decision_function score",
            "feature_names": artifact["feature_names"],
            "classes": {"0": "bad", "1": "good"},
            "probability_recipe": (
                "score = base_model.decision_function(frame[feature_names]); "
                "p_good = calibrator.predict_proba(score.reshape(-1, 1))[:, 1]"
            ),
        },
        "label_mapping": artifact["label_mapping"],
        "source_label_mapping": {"1": "good", "2": "bad"},
        **coverage,
        "source_label_mapping_evidence": (
            "Official Zenodo image ZIP preview: test/good/A549/0-1_days/221010_82.png "
            "has spreadsheet decision 1; test/bad/A549/4+_days/230529_207.png has decision 2."
        ),
        "split_rule": "All images sharing the first six imageID characters stay together",
        "split_counts": split_summary(artifact, frame),
        "split_groups": artifact["split_groups"],
        "confidence_threshold": artifact["confidence_threshold"],
        "threshold_status": artifact["threshold_status"],
        "calibration_threshold_exploration": select_calibration_threshold(
            artifact, frame, target_error_rate=args.calibration_target_error_rate
        ),
        "calibration": evaluate_frame(artifact, frame, split="calibration"),
        "held_out_test": evaluate_frame(artifact, frame, split="test"),
    }
    report["image_matching"] = matching_summary
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.output_dir / "baseline.joblib")
    (args.output_dir / "metadata.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved model and metadata to {args.output_dir}")
    print(f"Held-out test: {report['held_out_test']['overall']['n_images']} images, "
          f"{report['held_out_test']['group_count']} groups")


if __name__ == "__main__":
    main()
