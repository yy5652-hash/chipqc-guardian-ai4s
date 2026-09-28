"""Focused checks for the transparent UI measurements and manifest contract."""

from __future__ import annotations

import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chipqc_ui.manifest import manifest_csv, manifest_json, manifest_row  # noqa: E402
from chipqc_ui.model_adapter import load_baseline_model, model_feature_values, predict_baseline  # noqa: E402
from chipqc_ui.quality import QualityMetrics, classify_rule, extract_display_metrics  # noqa: E402


class QualityMetricsTests(unittest.TestCase):
    class _BaseModel:
        def decision_function(self, matrix: object) -> np.ndarray:
            return np.asarray([0.0])

    class _Calibrator:
        def predict_proba(self, scores: np.ndarray) -> np.ndarray:
            return np.asarray([[0.2, 0.8]])

    def test_flat_image_reacquire_is_a_rule_suggestion(self) -> None:
        image = Image.new("RGB", (64, 64), (128, 128, 128))
        metrics = extract_display_metrics(image)
        self.assertEqual(metrics.clarity, 0.0)
        self.assertEqual(metrics.contrast, 0.0)
        self.assertEqual(metrics.clipped_fraction, 0.0)
        self.assertEqual(classify_rule(metrics).status, "REACQUIRE")

    def test_crisp_pattern_has_more_laplacian_variance_than_blurred_pattern(self) -> None:
        checker = (np.indices((64, 64)).sum(axis=0) % 2 * 100 + 80).astype("uint8")
        crisp = extract_display_metrics(Image.fromarray(checker, mode="L"))
        blurred = extract_display_metrics(Image.new("L", (64, 64), 130))
        self.assertGreater(crisp.clarity, blurred.clarity)
        self.assertAlmostEqual(crisp.brightness, 130 / 255, places=2)
        self.assertEqual(crisp.clipped_fraction, 0.0)

    def test_optional_core_features_can_feed_the_model_contract(self) -> None:
        image = Image.open(Path(__file__).resolve().parents[1] / "demo_assets" / "synthetic_checker.pgm")
        metrics = extract_display_metrics(image)
        values = model_feature_values(image, metrics)
        self.assertIn("laplacian_variance", values)
        self.assertIn("brightness_mean", values)
        self.assertTrue(np.isfinite(values["laplacian_variance"]))

    def test_rule_levels_and_reasons(self) -> None:
        good = QualityMetrics(64, 64, 100.0, 0.5, 0.2, 0.0)
        review = QualityMetrics(64, 64, 60.0, 0.5, 0.2, 0.0)
        critical = QualityMetrics(64, 64, 20.0, 0.5, 0.2, 0.0)
        self.assertEqual(classify_rule(good).status, "PASS")
        self.assertEqual(classify_rule(review).status, "REVIEW")
        self.assertEqual(classify_rule(critical).status, "REACQUIRE")
        self.assertIn("清晰度很低", classify_rule(critical).reasons)

    def test_manifest_round_trip_keeps_rule_source_and_boundary(self) -> None:
        metrics = QualityMetrics(12, 10, 100.0, 0.5, 0.2, 0.0)
        row = manifest_row("合成图.png", metrics, classify_rule(metrics))
        csv_rows = list(csv.DictReader(io.StringIO(manifest_csv([row]).decode("utf-8-sig"))))
        json_payload = json.loads(manifest_json([row]))
        self.assertEqual(csv_rows[0]["filename"], "合成图.png")
        self.assertEqual(csv_rows[0]["decision_source"], "uncalibrated_rules")
        self.assertEqual(json_payload["items"][0]["rule_status"], "PASS")
        self.assertIn("Not for diagnosis", json_payload["use_boundary"])

    def test_missing_model_is_explicit(self) -> None:
        model = load_baseline_model(Path("/does-not-exist-chipqc-test"))
        self.assertFalse(model.available)
        self.assertIn("未发现", model.message)

    def test_project_artifact_uses_calibrator_without_changing_rule_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = root / "artifacts"
            artifacts.mkdir()
            (artifacts / "baseline.joblib").write_bytes(b"test placeholder")
            (artifacts / "metadata.json").write_text(
                json.dumps({
                    "feature_columns": ["brightness_mean"],
                    "label_mapping": {"0": "bad", "1": "good"},
                    "confidence_threshold": 0.8,
                }),
                encoding="utf-8",
            )
            fake_artifact = {
                "base_model": self._BaseModel(),
                "calibrator": self._Calibrator(),
                "feature_columns": ["brightness_mean"],
            }
            with patch.dict(sys.modules, {"joblib": SimpleNamespace(load=lambda _: fake_artifact)}):
                model = load_baseline_model(root)

        self.assertTrue(model.available)
        self.assertEqual(model.confidence_threshold, 0.8)
        image = Image.open(Path(__file__).resolve().parents[1] / "demo_assets" / "synthetic_flat.pgm")
        metrics = extract_display_metrics(image)
        prediction = predict_baseline(model, image, metrics)
        self.assertEqual(prediction.label, "good")
        self.assertAlmostEqual(prediction.probability_good, 0.8)
        self.assertEqual(classify_rule(metrics).status, "REACQUIRE")


if __name__ == "__main__":
    unittest.main()
