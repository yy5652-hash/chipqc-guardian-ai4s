"""Match spreadsheet IDs to images and extract reproducible quality features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chipqc.dataset import ImageIndex, prepare_feature_table  # noqa: E402
from chipqc.metadata import load_metadata  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=ROOT / "data/raw/OOC_datasheet.xlsx")
    parser.add_argument("--images", type=Path, default=ROOT / "data/raw/OOC_image_dataset.zip")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed")
    args = parser.parse_args()
    images = ImageIndex(args.images)
    features, manifest, summary = prepare_feature_table(load_metadata(args.metadata), images)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output_dir / "features.csv", index=False, float_format="%.10g")
    manifest.to_csv(args.output_dir / "manifest.csv", index=False)
    (args.output_dir / "matching_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Extracted {len(features)} feature rows from {summary['source_image_count']} image files")
    print(f"Matching statuses: {summary['status_counts']}")


if __name__ == "__main__":
    main()
