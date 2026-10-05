"""Dataset manifest for the public Organ-on-a-Chip image dataset (Movčana et al., Zenodo 10.5281/zenodo.10203721).

One row per frame, joined from the authors' spreadsheet and the extracted archive. The acquisition date is the
first six characters of the image ID (YYMMDD); every evaluation in this project holds out whole dates.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ZENODO_RECORD = "10203721"
ARCHIVE_MD5 = "8f7e058996203d48eb03b2d86c0a2e4d"
COLUMNS = ["image_id", "date", "frame_index", "cell_line", "label_good", "hours_after_seeding", "day", "day_bin",
           "seeding_density", "flow_rate", "author_split", "path"]


def acquisition_date(image_id: str) -> str:
    return str(image_id)[:6]


def build_manifest(images_root: Path, datasheet: Path) -> pd.DataFrame:
    """Join the spreadsheet with the extracted archive. Raises if any row lacks exactly one image or the
    folder label disagrees with the spreadsheet label."""
    sheet = pd.read_excel(datasheet, dtype={"imageID": str})
    sheet = sheet.rename(columns={"imageID": "image_id", "cell type": "cell_line", "seeding density, cells/ml": "seeding_density",
                                  "time after seeding, h": "hours_after_seeding", "Decision 1/2 (good/bad)": "decision", "flow rate": "flow_rate"})
    sheet = sheet[sheet.image_id.notna()].copy()
    files = {}
    for p in sorted(Path(images_root).rglob("*.png")):
        if p.stem in files:
            raise ValueError(f"image ID {p.stem} appears twice in the archive")
        files[p.stem] = p
    missing = [i for i in sheet.image_id if i not in files]
    if missing:
        raise ValueError(f"{len(missing)} spreadsheet rows have no image, e.g. {missing[:3]}")
    rows = []
    for r in sheet.itertuples(index=False):
        p = files[r.image_id]
        rel = p.relative_to(images_root)
        split, folder_label, _, day_bin = rel.parts[-5], rel.parts[-4], rel.parts[-3], rel.parts[-2]
        good = int(r.decision == 1)
        if (folder_label == "good") != bool(good):
            raise ValueError(f"{r.image_id}: spreadsheet says {r.decision}, archive folder says {folder_label}")
        rows.append(dict(image_id=r.image_id, date=acquisition_date(r.image_id), frame_index=int(r.image_id.split("_")[1]),
                         cell_line=str(r.cell_line).strip(), label_good=good, hours_after_seeding=r.hours_after_seeding, day=r.day,
                         day_bin=day_bin.replace("_days", ""), seeding_density=r.seeding_density, flow_rate=r.flow_rate,
                         author_split=split, path=rel.as_posix()))
    out = pd.DataFrame(rows, columns=COLUMNS)
    if out.image_id.duplicated().any():
        raise ValueError("duplicate image IDs in the spreadsheet")
    return out


def load_manifest(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"image_id": str, "date": str})
