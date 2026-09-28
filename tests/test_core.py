"""Focused regression checks for metadata, image matching, and group safety."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from chipqc.dataset import ImageIndex, prepare_feature_table  # noqa: E402
from chipqc.features import FEATURE_COLUMNS, extract_quality_features  # noqa: E402
from chipqc.metadata import audit_metadata, load_metadata  # noqa: E402


class MetadataAndFeaturesTest(unittest.TestCase):
    def test_real_metadata_counts_and_mapping(self) -> None:
        source = ROOT / "data/raw/OOC_datasheet.xlsx"
        if not source.exists():
            self.skipTest("Original spreadsheet is unavailable")
        frame = load_metadata(source)
        audit = audit_metadata(frame)
        self.assertEqual(audit["record_count"], 3072)
        self.assertEqual(audit["group_count"], 59)
        self.assertEqual(audit["invalid_image_id_count"], 0)
        self.assertEqual(audit["duplicate_image_id_count"], 0)
        self.assertEqual(audit["invalid_label_count"], 0)
        self.assertEqual(frame.loc[frame["source_decision"] == 1, "label_good"].unique().tolist(), [1])
        self.assertEqual(frame.loc[frame["source_decision"] == 2, "label_good"].unique().tolist(), [0])

    def test_features_are_finite_and_do_not_mutate_image(self) -> None:
        gradient = np.tile(np.arange(128, dtype=np.uint8), (128, 1))
        image = Image.fromarray(gradient, mode="L")
        before = image.tobytes()
        result = extract_quality_features(image)
        self.assertEqual(tuple(result), FEATURE_COLUMNS)
        self.assertTrue(all(np.isfinite(list(result.values()))))
        self.assertGreater(result["entropy_bits"], 0)
        self.assertGreater(result["gradient_mean"], 0)
        self.assertEqual(image.tobytes(), before)

    def test_zip_label_conflict_is_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "images.zip"
            image = Image.fromarray(np.full((16, 16, 3), 120, dtype=np.uint8), mode="RGB")
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("test/good/A549/220429_01.png", buffer.getvalue())
                archive.writestr("test/bad/A549/220429_02.png", buffer.getvalue())
            metadata = pd.DataFrame({
                "image_id": ["220429_01", "220429_02"],
                "cell_type": ["A549", "A549"],
                "seeding_density": [None, None],
                "time_after_seeding_h": [None, None],
                "day": [1, 1],
                "source_decision": [1, 1],
                "flow_rate": [None, None],
                "group_id": ["220429", "220429"],
                "label_good": pd.Series([1, 1], dtype="Int64"),
            })
            features, manifest, summary = prepare_feature_table(metadata, ImageIndex(archive_path))
            self.assertEqual(len(features), 1)
            self.assertEqual(summary["status_counts"], {"matched": 1, "label_conflict": 1})
            self.assertEqual(manifest.loc[1, "status"], "label_conflict")

    def test_incomplete_zip_fails_before_feature_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "images.zip"
            archive_path.write_bytes(b"PK\x03\x04incomplete")
            with self.assertRaisesRegex(ValueError, "incomplete or invalid"):
                ImageIndex(archive_path)


try:
    from chipqc.modeling import (
        evaluate_frame, fit_baseline, group_train_cal_test_split,
        select_calibration_threshold, selective_decision,
    )
    from scripts.train_baseline import validate_prepared_data
except ModuleNotFoundError as exc:
    HAS_SKLEARN = False
    SKLEARN_IMPORT_ERROR = str(exc)
else:
    HAS_SKLEARN = True
    SKLEARN_IMPORT_ERROR = ""


@unittest.skipUnless(HAS_SKLEARN, f"scikit-learn unavailable: {SKLEARN_IMPORT_ERROR}")
class ModelingTest(unittest.TestCase):
    @staticmethod
    def synthetic_frame() -> pd.DataFrame:
        rng = np.random.default_rng(17)
        rows = []
        for group_number in range(15):
            group = f"23{group_number:04d}"
            for image_number in range(10):
                good = image_number % 2
                row = {name: float(rng.normal(0, 0.1)) for name in FEATURE_COLUMNS}
                row["brightness_mean"] = float(good + rng.normal(0, 0.15))
                rows.append({
                    "image_id": f"{group}_{image_number:02d}",
                    "group_id": group,
                    "cell_type": "A549" if group_number % 2 else "HPMEC",
                    "label_good": good,
                    **row,
                })
        return pd.DataFrame(rows)

    def test_split_and_calibration_are_group_disjoint(self) -> None:
        frame = self.synthetic_frame()
        splits = group_train_cal_test_split(frame, seed=42)
        groups = {name: set(frame.iloc[indices]["group_id"]) for name, indices in splits.items()}
        self.assertFalse(groups["train"] & groups["calibration"])
        self.assertFalse(groups["train"] & groups["test"])
        self.assertFalse(groups["calibration"] & groups["test"])
        artifact = fit_baseline(frame, confidence_threshold=0.8)
        report = evaluate_frame(artifact, frame)
        self.assertEqual(report["overall"]["n_images"], len(splits["test"]))
        self.assertGreaterEqual(report["overall"]["brier_score"], 0)
        self.assertLessEqual(report["overall"]["brier_score"], 1)
        self.assertEqual(sum(map(sum, report["overall"]["confusion_matrix_rows_bad_good"])), len(splits["test"]))
        self.assertEqual(set(report["by_cell_type"]), {"A549", "HPMEC"})
        calibration_choice = select_calibration_threshold(artifact, frame, target_error_rate=0.2)
        self.assertIn("status", calibration_choice)
        self.assertEqual(calibration_choice["total"], len(splits["calibration"]))

    def test_review_boundary_is_consistent_with_binary_prediction(self) -> None:
        result = selective_decision(np.array([0.2, 0.5, 0.8]), 0.8)
        self.assertEqual(result.tolist(), [0, -1, 1])
        no_review = selective_decision(np.array([0.2, 0.5, 0.8]), 0.5)
        self.assertEqual(no_review.tolist(), [0, 1, 1])

    def test_partial_training_requires_explicit_flag(self) -> None:
        metadata = pd.DataFrame({
            "image_id": ["220429_01", "220430_01"],
            "cell_type": ["A549", "A549"],
            "seeding_density": [None, None],
            "time_after_seeding_h": [None, None],
            "day": [1, 1],
            "source_decision": [1, 2],
            "flow_rate": [None, None],
            "group_id": ["220429", "220430"],
            "label_good": pd.Series([1, 0], dtype="Int64"),
        })
        manifest = metadata.loc[:, ["image_id", "group_id", "cell_type", "label_good"]].copy()
        manifest["status"] = ["matched", "missing_image"]
        features = manifest.iloc[:1].loc[:, ["image_id", "group_id", "cell_type", "label_good"]].copy()
        summary = {
            "metadata_record_count": 2,
            "matched_feature_count": 1,
            "status_counts": {"matched": 1, "missing_image": 1},
        }
        # Match the mixed pandas string dtypes produced by Excel and CSV reads.
        metadata["image_id"] = metadata["image_id"].astype("string")
        manifest["image_id"] = manifest["image_id"].astype(object)
        features["image_id"] = features["image_id"].astype(object)
        with self.assertRaisesRegex(ValueError, "--allow-partial"):
            validate_prepared_data(metadata, manifest, features, summary, allow_partial=False)
        coverage = validate_prepared_data(metadata, manifest, features, summary, allow_partial=True)
        self.assertTrue(coverage["subset_only"])
        self.assertEqual(coverage["excluded_metadata_rows"], 1)


if __name__ == "__main__":
    unittest.main()
