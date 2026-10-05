"""Embeddings for the representations compared in the report (section 5.3) -> features/<name>.npz

    python scripts/extract_ablation_features.py                # all of them, about one hour on a laptop GPU
    python scripts/extract_ablation_features.py --only mobilenet_v2_672

The released model's own features come from scripts/extract_features.py. Each variant here changes one thing:
how much of the frame the backbone sees, at what resolution, or which backbone reads it.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from chipqc.backbone import FrameEncoder  # noqa: E402
from chipqc.data import load_manifest  # noqa: E402
from chipqc.frames import IMAGENET_MEAN, IMAGENET_STD, open_frame, to_model_input  # noqa: E402

#: name -> (backbone, frame size shown to the model or "crop224", ViT tile grid)
VARIANTS = {
    "mobilenet_v2_224crop": ("mobilenet_v2", "crop224", None),      # the representation of our first submission
    "mobilenet_v2_672": ("mobilenet_v2", (672, 504), None),
    "mobilenet_v2_1344": ("mobilenet_v2", (1344, 1008), None),
    "mobilenet_v2_2048": ("mobilenet_v2", (2048, 1536), None),
    "resnet50_1344": ("resnet50", (1344, 1008), None),
    "convnext_tiny_1344": ("convnext_tiny", (1344, 1008), None),
    "dinov2_vits14_2x2": ("dinov2_vits14", (896, 672), (2, 2)),
    "dinov2_vits14_4x4": ("dinov2_vits14", (1792, 1344), (4, 4)),
    "dinov2_vitb14": ("dinov2_vitb14", (1344, 1008), (3, 3)),
}


def centre_crop_224(image: Image.Image) -> np.ndarray:
    """Shorter side to 232 px, then the central 224 x 224 px (torchvision's ImageNet evaluation transform)."""
    s = 232 / min(image.size)
    im = image.resize((round(image.width * s), round(image.height * s)), Image.BILINEAR)
    x0, y0 = (im.width - 224) // 2, (im.height - 224) // 2
    x = np.asarray(im.crop((x0, y0, x0 + 224, y0 + 224)), dtype=np.float32) / 255.0
    return ((x - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1).copy()


class Frames(torch.utils.data.Dataset):
    def __init__(self, images, paths, size):
        self.images, self.paths, self.size = images, paths, size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        image = open_frame(self.images / self.paths[i])
        return centre_crop_224(image) if self.size == "crop224" else to_model_input(image, self.size)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", type=Path, default=ROOT / "data/raw/OOC_image_dataset")
    ap.add_argument("--only", choices=list(VARIANTS))
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    man = load_manifest(ROOT / "data/manifest.csv")
    for name, (backbone, size, grid) in VARIANTS.items():
        out = ROOT / f"features/{name}.npz"
        if (a.only and name != a.only) or (not a.only and out.exists() and np.load(out)["embeddings"].dtype == np.float16):
            continue
        enc = FrameEncoder(backbone, grid=grid)
        batch = 2 if size != "crop224" and size[0] >= 1792 or backbone == "dinov2_vitb14" else 32 if size == "crop224" else 6
        loader = torch.utils.data.DataLoader(Frames(a.images, man.path.tolist(), size), batch_size=batch, num_workers=a.workers)
        parts, t0 = [], time.time()
        for x in loader:
            parts.append(enc.embed(x.numpy()))
        np.savez_compressed(out, embeddings=np.concatenate(parts).astype(np.float16), image_id=man.image_id.to_numpy().astype(str), backbone=name)
        print(f"{name}: {len(man)} x {enc.dim} in {time.time() - t0:.0f} s -> {out.name}", flush=True)
        del enc


if __name__ == "__main__":
    main()
