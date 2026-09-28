"""Deterministic classical image quality descriptors.

Every feature is computed from pixels alone. Images are resized to a maximum
side of 512 before pixel statistics, so the feature scale is reproducible.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


MAX_SIDE = 512
FEATURE_COLUMNS = (
    "width", "height", "aspect_ratio", "brightness_mean", "brightness_std",
    "brightness_p05", "brightness_p95", "dark_fraction", "bright_fraction",
    "contrast_p90_p10", "entropy_bits", "saturation_mean", "saturation_std",
    "red_mean", "green_mean", "blue_mean", "red_std", "green_std", "blue_std",
    "laplacian_variance", "gradient_mean", "gradient_p90", "edge_fraction",
    "illumination_tile_cv",
)


def _open_image(image: Image.Image | str | Path | bytes | np.ndarray) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.copy()
    if isinstance(image, (str, Path)):
        with Image.open(image) as source:
            return source.copy()
    if isinstance(image, bytes):
        with Image.open(BytesIO(image)) as source:
            return source.copy()
    if isinstance(image, np.ndarray):
        array = np.asarray(image)
        if array.dtype != np.uint8:
            raise ValueError("NumPy image arrays must have uint8 pixel values")
        return Image.fromarray(array)
    raise TypeError("image must be a PIL image, path, encoded bytes, or uint8 array")


def extract_quality_features(image: Image.Image | str | Path | bytes | np.ndarray) -> dict[str, float]:
    """Extract finite, fixed-name image quality features for inference/training."""
    source = _open_image(image)
    source = ImageOps.exif_transpose(source).convert("RGB")
    width, height = source.size
    if min(width, height) < 3:
        raise ValueError("Image must be at least 3 by 3 pixels")
    source.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
    rgb = np.asarray(source, dtype=np.float32) / 255.0
    gray = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    saturation = np.asarray(source.convert("HSV"), dtype=np.float32)[:, :, 1] / 255.0
    p05, p10, p90, p95 = np.percentile(gray, (5, 10, 90, 95))
    histogram = np.bincount(np.clip(np.rint(gray * 255), 0, 255).astype(np.uint8).ravel(), minlength=256)
    probabilities = histogram[histogram > 0] / histogram.sum()
    entropy = float(-(probabilities * np.log2(probabilities)).sum())

    center = gray[1:-1, 1:-1]
    laplacian = gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4 * center
    gx = (gray[1:-1, 2:] - gray[1:-1, :-2]) / 2
    gy = (gray[2:, 1:-1] - gray[:-2, 1:-1]) / 2
    gradient = np.hypot(gx, gy)
    tile_means = [
        float(tile.mean())
        for tile in np.array_split(gray, 4, axis=0)
        for tile in np.array_split(tile, 4, axis=1)
        if tile.size
    ]
    values = {
        "width": float(width),
        "height": float(height),
        "aspect_ratio": float(width / height),
        "brightness_mean": float(gray.mean()),
        "brightness_std": float(gray.std()),
        "brightness_p05": float(p05),
        "brightness_p95": float(p95),
        "dark_fraction": float((gray < 0.05).mean()),
        "bright_fraction": float((gray > 0.95).mean()),
        "contrast_p90_p10": float(p90 - p10),
        "entropy_bits": entropy,
        "saturation_mean": float(saturation.mean()),
        "saturation_std": float(saturation.std()),
        "red_mean": float(rgb[:, :, 0].mean()),
        "green_mean": float(rgb[:, :, 1].mean()),
        "blue_mean": float(rgb[:, :, 2].mean()),
        "red_std": float(rgb[:, :, 0].std()),
        "green_std": float(rgb[:, :, 1].std()),
        "blue_std": float(rgb[:, :, 2].std()),
        "laplacian_variance": float(laplacian.var()),
        "gradient_mean": float(gradient.mean()),
        "gradient_p90": float(np.percentile(gradient, 90)),
        "edge_fraction": float((gradient > 0.10).mean()),
        "illumination_tile_cv": float(np.std(tile_means) / max(np.mean(tile_means), 1e-6)),
    }
    if tuple(values) != FEATURE_COLUMNS or not all(np.isfinite(list(values.values()))):
        raise RuntimeError("Feature extraction produced an invalid feature vector")
    return values
