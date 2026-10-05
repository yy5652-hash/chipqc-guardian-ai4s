"""Write data/manifest.csv from the extracted Zenodo archive and its spreadsheet."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from chipqc.data import build_manifest  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--images", type=Path, default=ROOT / "data/raw/OOC_image_dataset")
ap.add_argument("--datasheet", type=Path, default=ROOT / "data/raw/OOC_datasheet.xlsx")
ap.add_argument("--out", type=Path, default=ROOT / "data/manifest.csv")
a = ap.parse_args()
m = build_manifest(a.images, a.datasheet)
a.out.parent.mkdir(parents=True, exist_ok=True)
m.to_csv(a.out, index=False)
print(f"{len(m)} frames, {m.date.nunique()} acquisition dates, {m.cell_line.nunique()} cell lines, "
      f"{int(m.label_good.sum())} good / {int((1 - m.label_good).sum())} bad -> {a.out}")
