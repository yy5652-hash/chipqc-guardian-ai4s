"""Transparent image quality heuristics for a pre-model demonstration.

These measurements are deliberately simple and have not been calibrated against
acquisition hardware, human labels, or clinical outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps


MAX_ANALYSIS_SIDE = 1024


@dataclass(frozen=True)
class QualityMetrics:
    width: int
    height: int
    clarity: float
    brightness: float
    contrast: float
    clipped_fraction: float

    def feature_values(self) -> dict[str, float]:
        """Return stable names for the optional baseline model contract."""
        return {
            "clarity": self.clarity,
            "brightness": self.brightness,
            "contrast": self.contrast,
            "clipped_fraction": self.clipped_fraction,
        }


@dataclass(frozen=True)
class RuleDecision:
    status: str
    reasons: tuple[str, ...]


def extract_display_metrics(image: Image.Image) -> QualityMetrics:
    """Measure a bounded RGB preview, retaining original pixel dimensions."""
    upright = ImageOps.exif_transpose(image)
    width, height = upright.size
    if width < 1 or height < 1:
        raise ValueError("图像尺寸必须大于零")

    preview = upright.convert("RGB")
    preview.thumbnail((MAX_ANALYSIS_SIDE, MAX_ANALYSIS_SIDE), Image.Resampling.LANCZOS)
    gray = np.asarray(preview.convert("L"), dtype=np.float32)

    brightness = float(gray.mean() / 255.0)
    contrast = float(gray.std() / 255.0)
    clipped_fraction = float(np.mean((gray <= 5.0) | (gray >= 250.0)))

    if min(gray.shape) < 3:
        clarity = 0.0
    else:
        center = gray[1:-1, 1:-1]
        laplacian = (
            4.0 * center
            - gray[:-2, 1:-1]
            - gray[2:, 1:-1]
            - gray[1:-1, :-2]
            - gray[1:-1, 2:]
        )
        clarity = float(np.var(laplacian))

    return QualityMetrics(
        width=width,
        height=height,
        clarity=clarity,
        brightness=brightness,
        contrast=contrast,
        clipped_fraction=clipped_fraction,
    )


def classify_rule(metrics: QualityMetrics) -> RuleDecision:
    """Apply illustrative thresholds, with the most serious status winning."""
    critical: list[str] = []
    review: list[str] = []

    if metrics.clarity < 25.0:
        critical.append("清晰度很低")
    elif metrics.clarity < 70.0:
        review.append("清晰度偏低")

    if metrics.brightness < 0.15:
        critical.append("整体过暗")
    elif metrics.brightness > 0.90:
        critical.append("整体过亮")
    elif metrics.brightness < 0.30 or metrics.brightness > 0.80:
        review.append("亮度接近边界")

    if metrics.contrast < 0.045:
        critical.append("对比度很低")
    elif metrics.contrast < 0.10:
        review.append("对比度偏低")

    if metrics.clipped_fraction > 0.65:
        critical.append("黑白剪切像素过多")
    elif metrics.clipped_fraction > 0.30:
        review.append("黑白剪切像素偏多")

    if critical:
        return RuleDecision("REACQUIRE", tuple(critical + review))
    if review:
        return RuleDecision("REVIEW", tuple(review))
    return RuleDecision("PASS", ("演示阈值内未发现明显采集质量问题",))
