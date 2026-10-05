"""Controlled acquisition faults applied to real frames (for validating the acquisition gate)."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

FAULTS = {
    "defocus": {"unit": "Gaussian sigma, px", "levels": [1, 2, 4, 8], "rule": "defocus"},
    "motion_streak": {"unit": "streak length, px", "levels": [5, 10, 20, 40], "rule": "motion_streak"},
    "occlusion": {"unit": "share of field blacked out", "levels": [0.1, 0.25, 0.4, 0.6], "rule": "occluded"},
    "under_exposure": {"unit": "exposure gain", "levels": [0.7, 0.5, 0.35, 0.2], "rule": "under_exposed"},
}


def degrade(grey: np.ndarray, fault: str, level: float, rng: np.random.Generator) -> np.ndarray:
    """grey: float32 frame in [0, 1] at 1024 x 768 (frames.to_grey). Returns the degraded frame."""
    if fault == "defocus":
        return np.asarray(Image.fromarray((grey * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(level)), dtype=np.float32) / 255.0
    if fault == "motion_streak":
        kernel, axis = np.ones(int(level), dtype=np.float32) / int(level), int(rng.integers(0, 2))
        return np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="same"), axis, grey).astype(np.float32)
    if fault == "occlusion":
        out, (h, w), side = grey.copy(), grey.shape, int(rng.integers(0, 4))
        n = int(round((h if side < 2 else w) * level))
        if side < 2:
            out[(slice(0, n) if side == 0 else slice(h - n, h)), :] = 0.02 + 0.01 * rng.random((n, w), dtype=np.float32)
        else:
            out[:, (slice(0, n) if side == 2 else slice(w - n, w))] = 0.02 + 0.01 * rng.random((h, n), dtype=np.float32)
        return out
    if fault == "under_exposure":
        return np.clip(grey * level, 0, 1).astype(np.float32)
    raise ValueError(fault)
