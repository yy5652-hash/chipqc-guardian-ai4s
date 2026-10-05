"""Run ChipQC Guardian on frames or folders of frames.

    python inference.py examples/                     # the bundled example run
    python inference.py /path/to/run --target 0.05 --out qc/

Writes <out>/audit.csv and <out>/audit.json (one record per frame) and, unless --no-maps is given,
<out>/maps/<frame>.jpg with the frame next to its evidence map. Nothing is uploaded anywhere.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from chipqc.frames import open_frame  # noqa: E402
from chipqc.guardian import Guardian  # noqa: E402
from chipqc.render import side_by_side  # noqa: E402

SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def collect(inputs: list[Path]) -> list[Path]:
    files = []
    for p in inputs:
        files += sorted(q for q in p.rglob("*") if q.suffix.lower() in SUFFIXES) if p.is_dir() else [p]
    return files


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", type=Path, help="frames or folders")
    ap.add_argument("--model", type=Path, default=ROOT / "models/guardian-v2")
    ap.add_argument("--target", default=None, choices=["0.1", "0.15", "0.2"], help="largest acceptable share of bad frames among passed frames (default 0.1)")
    ap.add_argument("--out", type=Path, default=Path("chipqc_out"))
    ap.add_argument("--no-maps", action="store_true")
    a = ap.parse_args()

    files = collect(a.inputs)
    if not files:
        sys.exit("no image files found")
    g = Guardian(a.model)
    target = a.target or g.default_target
    a.out.mkdir(parents=True, exist_ok=True)
    if not a.no_maps:
        (a.out / "maps").mkdir(exist_ok=True)
    records = []
    for f in files:
        image = open_frame(f)
        r = g.assess(image, error_target=target)
        records.append({"frame": str(f), "decision": r.decision, "p_good": round(r.p_good, 4), "reason": "; ".join(r.reasons),
                        **{k: round(v, 4) for k, v in r.acquisition.items()}, "pass_threshold": g.pass_threshold(target), "error_target": float(target),
                        "model": g.spec["name"], "model_created_utc": g.spec["created_utc"], "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
                        "assessed_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})
        if not a.no_maps:
            side_by_side(image, r.evidence, r.decision, r.p_good, f.name).save(a.out / "maps" / f"{f.stem}.jpg", quality=88)
        print(f"{r.decision:9s} P(good) {r.p_good:.2f}  {f.name}  ({r.reasons[0]})")
    with open(a.out / "audit.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(records[0]))
        w.writeheader(); w.writerows(records)
    (a.out / "audit.json").write_text(json.dumps({"model": g.spec["name"], "backbone": g.spec["backbone"], "error_target": float(target), "pass_threshold": g.pass_threshold(target),
                                                 "acquisition_limits": g.limits, "frames": records}, indent=1))
    counts = {d: sum(r["decision"] == d for r in records) for d in ("PASS", "REVIEW", "REACQUIRE")}
    print(f"{len(records)} frames: {counts['PASS']} PASS, {counts['REVIEW']} REVIEW, {counts['REACQUIRE']} REACQUIRE -> {a.out}/audit.csv")


if __name__ == "__main__":
    main()
