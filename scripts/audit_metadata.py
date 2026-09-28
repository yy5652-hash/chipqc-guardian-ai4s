"""Audit OOC spreadsheet fields and collection-day leakage groups."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chipqc.metadata import audit_metadata, load_metadata  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=ROOT / "data/raw/OOC_datasheet.xlsx")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/metadata_audit.json")
    args = parser.parse_args()
    report = audit_metadata(load_metadata(args.metadata))
    report["source_spreadsheet"] = str(args.metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Audited {report['record_count']} records and {report['group_count']} groups; saved {args.output}")
    for field in ("invalid_image_id_count", "duplicate_image_id_count", "invalid_label_count"):
        if report[field]:
            print(f"WARNING: {field}={report[field]}")


if __name__ == "__main__":
    main()
