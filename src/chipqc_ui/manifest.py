"""Portable, explicit exports for research quality-control review."""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Mapping, Sequence

from .quality import QualityMetrics, RuleDecision


MANIFEST_FIELDS = (
    "filename",
    "width_px",
    "height_px",
    "clarity_laplacian_variance",
    "brightness_mean_0_1",
    "contrast_std_0_1",
    "clipped_fraction_0_1",
    "rule_status",
    "rule_reasons",
    "model_output",
    "model_probability_good",
    "model_confidence",
    "decision_source",
)

USE_BOUNDARY = (
    "Uncalibrated research/demo quality-control preview only. "
    "Not for diagnosis, patient care, chip acceptance, or automated acquisition decisions."
)


def manifest_row(
    filename: str,
    metrics: QualityMetrics,
    decision: RuleDecision,
    model_output: str = "",
    model_confidence: float | None = None,
    model_probability_good: float | None = None,
) -> dict[str, str | int | float]:
    return {
        "filename": filename,
        "width_px": metrics.width,
        "height_px": metrics.height,
        "clarity_laplacian_variance": round(metrics.clarity, 3),
        "brightness_mean_0_1": round(metrics.brightness, 4),
        "contrast_std_0_1": round(metrics.contrast, 4),
        "clipped_fraction_0_1": round(metrics.clipped_fraction, 4),
        "rule_status": decision.status,
        "rule_reasons": "; ".join(decision.reasons),
        "model_output": model_output,
        "model_probability_good": "" if model_probability_good is None else round(model_probability_good, 4),
        "model_confidence": "" if model_confidence is None else round(model_confidence, 4),
        "decision_source": "uncalibrated_rules",
    }


def manifest_csv(rows: Sequence[Mapping[str, object]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8-sig")


def manifest_json(rows: Sequence[Mapping[str, object]]) -> bytes:
    payload = {
        "manifest_type": "chipqc_guardian_demo",
        "decision_source": "uncalibrated_rules",
        "use_boundary": USE_BOUNDARY,
        "items": list(rows),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
