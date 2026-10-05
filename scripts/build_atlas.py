"""Thumbnails of the reference frames (shown as "most similar good / bad examples") -> models/guardian-v2/atlas/"""
import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from chipqc.data import load_manifest  # noqa: E402


def one(args):
    src, dst = args
    with Image.open(src) as im:
        im.convert("L").resize((144, 108), Image.LANCZOS).save(dst, quality=60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", type=Path, default=ROOT / "data/raw/OOC_image_dataset")
    ap.add_argument("--out", type=Path, default=ROOT / "models/guardian-v2/atlas")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    man = load_manifest(ROOT / "data/manifest.csv")
    with Pool(8) as pool:
        pool.map(one, [(a.images / p, a.out / f"{i}.jpg") for i, p in zip(man.image_id, man.path)], chunksize=32)
    print(len(man), "thumbnails ->", a.out)
