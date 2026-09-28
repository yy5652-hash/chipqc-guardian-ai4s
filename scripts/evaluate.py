"""Recompute saved group-held-out metrics, optionally including perturbations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chipqc.dataset import ImageIndex  # noqa: E402
from chipqc.modeling import evaluate_frame, file_sha256  # noqa: E402
from chipqc.robustness import evaluate_robustness  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "artifacts/baseline.joblib")
    parser.add_argument("--features", type=Path, default=ROOT / "data/processed/features.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/evaluation.json")
    parser.add_argument("--split", choices=("train", "calibration", "test"), default="test")
    parser.add_argument("--robustness", action="store_true")
    parser.add_argument("--max-robustness-images", type=int, default=200)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/processed/manifest.csv")
    parser.add_argument("--images", type=Path, default=ROOT / "data/raw/OOC_image_dataset.zip")
    args = parser.parse_args()
    artifact = joblib.load(args.model)
    if file_sha256(args.features) != artifact["feature_csv_sha256"]:
        parser.error("Feature CSV checksum differs from the data used to fit this model")
    features = pd.read_csv(args.features, dtype={"image_id": str, "group_id": str, "cell_type": str})
    report = evaluate_frame(artifact, features, split=args.split)
    if args.robustness:
        if args.split != "test":
            parser.error("Robustness evaluation is defined for the held-out test split only")
        manifest = pd.read_csv(args.manifest, dtype={"image_id": str, "group_id": str, "cell_type": str})
        report["robustness"] = evaluate_robustness(
            artifact, features, manifest, ImageIndex(args.images), max_images=args.max_robustness_images,
            seed=artifact["seed"],
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.split} evaluation for {report['overall']['n_images']} images to {args.output}")


if __name__ == "__main__":
    main()
