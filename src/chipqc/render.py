"""Pictures for people: the evidence map over the frame, and a decision badge."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

GOOD_RGB = np.array([42, 120, 214], dtype=np.float32)      # blue: this region pulls the score towards good
BAD_RGB = np.array([227, 73, 72], dtype=np.float32)        # red: pulls towards bad
DECISION_RGB = {"PASS": (12, 130, 40), "REVIEW": (170, 110, 0), "REACQUIRE": (170, 70, 40)}       # labelled badges, never colour alone


def evidence_overlay(image: Image.Image, evidence: np.ndarray, width: int = 960, strength: float = 0.55, scale: float | None = None) -> Image.Image:
    """Tint each region by its share of the log-odds. `scale` is the log-odds magnitude drawn at full strength
    (default: the 98th percentile of |evidence| in this frame, at least 1)."""
    height = int(round(width * image.height / image.width))
    base = np.asarray(image.convert("L").resize((width, height), Image.BILINEAR), dtype=np.float32)[..., None].repeat(3, axis=2)
    scale = scale or max(1.0, float(np.percentile(np.abs(evidence), 98)))
    e = np.asarray(Image.fromarray(evidence.astype(np.float32), mode="F").resize((width, height), Image.BICUBIC))
    alpha = (np.clip(np.abs(e) / scale, 0, 1) * strength)[..., None]
    tint = np.where(e[..., None] >= 0, GOOD_RGB, BAD_RGB)
    return Image.fromarray(np.clip(base * (1 - alpha) + tint * alpha, 0, 255).astype(np.uint8))


def _font(size: int):
    for name in ("Helvetica.ttc", "Arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def labelled(image: Image.Image, text: str, colour=(0, 0, 0)) -> Image.Image:
    out = image.copy()
    d = ImageDraw.Draw(out)
    f = _font(max(14, image.width // 42))
    box = d.textbbox((0, 0), text, font=f)
    d.rectangle([0, 0, box[2] + 16, box[3] + 12], fill=colour)
    d.text((8, 5), text, font=f, fill=(255, 255, 255))
    return out


def side_by_side(image: Image.Image, evidence: np.ndarray, decision: str, p_good: float, title: str = "", width: int = 760) -> Image.Image:
    height = int(round(width * image.height / image.width))
    left = labelled(image.convert("RGB").resize((width, height), Image.LANCZOS), title or "frame")
    right = labelled(evidence_overlay(image, evidence, width), f"{decision}   P(good) {p_good:.2f}", DECISION_RGB.get(decision, (0, 0, 0)))
    sheet = Image.new("RGB", (2 * width + 8, height), "white")
    sheet.paste(left, (0, 0)); sheet.paste(right, (width + 8, 0))
    return sheet
