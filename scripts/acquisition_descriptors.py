"""Acquisition descriptors for every frame in the manifest -> features/acquisition_descriptors.csv."""
import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from chipqc.data import load_manifest  # noqa: E402
from chipqc.descriptors import acquisition_descriptors  # noqa: E402
from chipqc.frames import open_frame, to_grey  # noqa: E402


def one(path):
    return acquisition_descriptors(to_grey(open_frame(path)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", type=Path, default=ROOT / "data/raw/OOC_image_dataset")
    ap.add_argument("--manifest", type=Path, default=ROOT / "data/manifest.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "features/acquisition_descriptors.csv")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    man = load_manifest(a.manifest)
    with Pool(a.workers) as pool:
        rows = pool.map(one, [a.images / p for p in man.path], chunksize=16)
    out = pd.concat([man[["image_id"]], pd.DataFrame(rows)], axis=1)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.round(6).to_csv(a.out, index=False)
    print(f"{len(out)} frames -> {a.out}")
