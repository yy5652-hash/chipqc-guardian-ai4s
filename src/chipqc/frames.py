"""Reading a brightfield frame the way the model saw every training frame.

The whole field of view is kept and resized to 1792 x 1344 px (a 0.87x downscale of the 2056 x 1542 source
frames), which the released backbone reads as 4 x 4 tiles of 448 x 336 px.
"""
from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, Union

import numpy as np
from PIL import Image

MODEL_SIZE = (1792, 1344)          # width, height shown to the released model: a 0.87x downscale of the 2056 x 1542 source frames
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
Source = Union[str, Path, BinaryIO, Image.Image]


def open_frame(source: Source) -> Image.Image:
    """RGB frame in landscape orientation (a portrait frame is rotated: orientation carries no meaning here)."""
    image = source if isinstance(source, Image.Image) else Image.open(source)
    image = image.convert("RGB")
    return image.rotate(90, expand=True) if image.height > image.width else image


def to_model_input(image: Image.Image, size: tuple[int, int] = MODEL_SIZE) -> np.ndarray:
    """RGB frame -> float32 array 3 x H x W, ImageNet-normalised. All reference frames are 4:3; other aspect
    ratios are resized to 4:3."""
    x = np.asarray(image.resize(size, Image.BICUBIC), dtype=np.float32) / 255.0
    return ((x - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1).copy()


def to_grey(image: Image.Image, size: tuple[int, int] = (1024, 768)) -> np.ndarray:
    """Grey float32 frame in [0, 1] at the fixed size the acquisition descriptors are defined on."""
    return np.asarray(image.convert("L").resize(size, Image.BILINEAR), dtype=np.float32) / 255.0
