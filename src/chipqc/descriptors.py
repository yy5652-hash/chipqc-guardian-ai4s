"""Transparent acquisition descriptors and the acquisition gate.

Each descriptor is a physical quantity an operator can verify on the frame itself. They are computed on the
grey frame at 1024 x 768 and never enter the culture-quality classifier; they decide whether a frame is a
usable observation at all.
"""
from __future__ import annotations

import numpy as np

BLOCK = 32
DARK_BLOCK = 0.12        # a 32 x 32 block whose mean is below this is blacked out
#: Default limits. `occluded` is a fixed physical choice; the others are the 5 % tails of the 3,072 reference frames
#: (scripts/fit_acquisition_limits.py) and are stored with each released model.
DEFAULT_LIMITS = {"occluded_block_frac_max": 0.25}


def acquisition_descriptors(grey: np.ndarray) -> dict[str, float]:
    """grey: float32 768 x 1024 in [0, 1] (frames.to_grey)."""
    h, w = grey.shape
    blocks = grey.reshape(h // BLOCK, BLOCK, w // BLOCK, BLOCK).swapaxes(1, 2)
    lit = blocks.mean((2, 3)) > DARK_BLOCK
    gx, gy = np.diff(grey, axis=1)[:-1, :], np.diff(grey, axis=0)[:, :-1]
    lap = grey[1:-1, 1:-1] * 4 - grey[:-2, 1:-1] - grey[2:, 1:-1] - grey[1:-1, :-2] - grey[1:-1, 2:]
    by, bx = (h - 2) // BLOCK, (w - 2) // BLOCK
    lap_var = lap[: by * BLOCK, : bx * BLOCK].reshape(by, BLOCK, bx, BLOCK).swapaxes(1, 2).var((2, 3))
    lit_inner = lit[:by, :bx]
    sharp = float(np.median(lap_var[lit_inner])) if lit_inner.any() else 0.0
    ex, ey = float((gx ** 2).mean()), float((gy ** 2).mean())
    return {
        "occluded_block_frac": float(1.0 - lit.mean()),          # share of the field that is blacked out
        "saturated_frac": float((grey > 0.97).mean()),           # share of clipped highlights
        "median_brightness": float(np.median(grey)),
        "log_sharpness": float(np.log10(sharp + 1e-9)),          # local Laplacian variance in lit blocks (defocus lowers it)
        "streak_anisotropy": abs(ex - ey) / (ex + ey + 1e-9),    # one-directional smear (motion) raises it
    }


def acquisition_flags(d: dict[str, float], limits: dict[str, float]) -> list[str]:
    """Human-readable reasons to re-acquire the frame; empty when the frame is a usable observation.

    Clipped highlights are reported as a descriptor but are not a reason to re-acquire: an empty, sparsely
    covered channel is bright for biological reasons, and that is the culture-quality model's call."""
    flags = []
    if d["occluded_block_frac"] > limits["occluded_block_frac_max"]:
        flags.append(f"{d['occluded_block_frac']:.0%} of the field is blacked out")
    if "streak_anisotropy_max" in limits and d["streak_anisotropy"] > limits["streak_anisotropy_max"]:
        flags.append("motion streak suspected (one-directional smear beyond the reference limit)")
    if "log_sharpness_min" in limits and d["log_sharpness"] < limits["log_sharpness_min"]:
        flags.append("defocus suspected (local sharpness below the reference limit)")
    if "median_brightness_min" in limits and d["median_brightness"] < limits["median_brightness_min"]:
        flags.append("under-exposed (median brightness below the reference limit)")
    return flags
