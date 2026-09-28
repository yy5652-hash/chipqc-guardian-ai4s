"""Deterministic image perturbations for a separate robustness audit."""

from __future__ import annotations

from io import BytesIO

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter

from .dataset import ImageIndex
from .features import extract_quality_features
from .modeling import predict_good_probability, selective_decision


PERTURBATIONS = ("brightness_0.75", "brightness_1.25", "contrast_0.75", "blur_1.5", "jpeg_quality_50")


def perturb_image(image: Image.Image, name: str) -> Image.Image:
    """Apply one reproducible, non-destructive acquisition-like disturbance."""
    rgb = image.convert("RGB")
    if name == "brightness_0.75":
        return ImageEnhance.Brightness(rgb).enhance(0.75)
    if name == "brightness_1.25":
        return ImageEnhance.Brightness(rgb).enhance(1.25)
    if name == "contrast_0.75":
        return ImageEnhance.Contrast(rgb).enhance(0.75)
    if name == "blur_1.5":
        return rgb.filter(ImageFilter.GaussianBlur(radius=1.5))
    if name == "jpeg_quality_50":
        buffer = BytesIO()
        rgb.save(buffer, format="JPEG", quality=50)
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            return compressed.copy()
    raise ValueError(f"Unknown perturbation: {name}")


def evaluate_robustness(
    artifact: dict,
    features: pd.DataFrame,
    manifest: pd.DataFrame,
    index: ImageIndex,
    max_images: int | None = 200,
    seed: int = 42,
) -> dict:
    """Compare perturbed and original predictions on held-out test images.

    This is a stress test, not a second test set or a retraining procedure.
    The default sample is deterministic and its actual size is reported.
    """
    test_ids = set(artifact["split_image_ids"]["test"])
    eligible = manifest.loc[(manifest["status"] == "matched") & manifest["image_id"].isin(test_ids)].copy()
    if max_images is not None:
        if max_images < 1:
            raise ValueError("max_images must be positive")
        if len(eligible) > max_images:
            rng = np.random.default_rng(seed)
            eligible = eligible.iloc[np.sort(rng.choice(len(eligible), size=max_images, replace=False))]
    feature_by_id = features.set_index("image_id")
    records: dict[str, list[tuple[int, float, float, int, int]]] = {name: [] for name in PERTURBATIONS}
    errors: dict[str, int] = {name: 0 for name in PERTURBATIONS}
    for row in eligible.itertuples(index=False):
        original = feature_by_id.loc[[row.image_id]].reset_index()
        original_p = float(predict_good_probability(artifact, original)[0])
        original_decision = int(selective_decision(np.array([original_p]), artifact["confidence_threshold"])[0])
        with Image.open(BytesIO(index.read(row.member))) as image:
            image.load()
            for name in PERTURBATIONS:
                try:
                    changed = perturb_image(image, name)
                    changed_features = pd.DataFrame([extract_quality_features(changed)])
                    changed_p = float(predict_good_probability(artifact, changed_features)[0])
                    changed_decision = int(selective_decision(np.array([changed_p]), artifact["confidence_threshold"])[0])
                    records[name].append((int(row.label_good), original_p, changed_p, original_decision, changed_decision))
                except Exception:
                    errors[name] += 1
    result = {"sampled_test_images": int(len(eligible)), "seed": seed, "max_images": max_images, "perturbations": {}}
    for name, rows in records.items():
        if not rows:
            result["perturbations"][name] = {"n": 0, "error_count": errors[name]}
            continue
        values = np.asarray(rows, dtype=float)
        y = values[:, 0].astype(int)
        original_p, changed_p = values[:, 1], values[:, 2]
        changed_decision = values[:, 4].astype(int)
        accepted = changed_decision != -1
        result["perturbations"][name] = {
            "n": len(rows),
            "error_count": errors[name],
            "prediction_flip_rate": float(np.mean((original_p >= 0.5) != (changed_p >= 0.5))),
            "mean_absolute_probability_shift": float(np.mean(np.abs(changed_p - original_p))),
            "accuracy": float(np.mean((changed_p >= 0.5) == y)),
            "selective_coverage": float(accepted.mean()),
            "selective_accuracy": float(np.mean(changed_decision[accepted] == y[accepted])) if accepted.any() else None,
            "decision_change_rate_including_review": float(np.mean(values[:, 3] != values[:, 4])),
        }
    return result
