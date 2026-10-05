"""Whole-frame embeddings for every frame in data/manifest.csv -> features/<backbone>.npz.

The released feature files let anyone re-run the full evaluation without downloading the 6.7 GB image archive.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from chipqc.backbone import FrameEncoder  # noqa: E402
from chipqc.data import load_manifest  # noqa: E402
from chipqc.frames import open_frame, to_model_input  # noqa: E402


class Frames(torch.utils.data.Dataset):
    def __init__(self, images: Path, paths: list[str]):
        self.images, self.paths = images, paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return to_model_input(open_frame(self.images / self.paths[i]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", type=Path, default=ROOT / "data/raw/OOC_image_dataset")
    ap.add_argument("--manifest", type=Path, default=ROOT / "data/manifest.csv")
    ap.add_argument("--backbone", default="dinov2_vits14")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    man = load_manifest(a.manifest)
    enc = FrameEncoder(a.backbone)
    loader = torch.utils.data.DataLoader(Frames(a.images, man.path.tolist()), batch_size=a.batch, num_workers=a.workers)
    parts, t0 = [], time.time()
    for i, batch in enumerate(loader):
        parts.append(enc.embed(batch.numpy()))
        if (i + 1) % 100 == 0:
            print(f"{(i + 1) * a.batch}/{len(man)} frames, {(i + 1) * a.batch / (time.time() - t0):.1f} per second", flush=True)
    out = a.out or ROOT / f"features/{a.backbone}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, embeddings=np.concatenate(parts).astype(np.float32), image_id=man.image_id.to_numpy().astype(str), backbone=a.backbone)
    print(f"saved {out} ({len(man)} x {enc.dim}) in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
