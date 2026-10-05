"""Does the acquisition gate catch controlled faults? -> results/degradation_study.json

Sixty frames per cell line are degraded at four severities per fault. For each fault we report how often the
released gate rule fires on the degraded frames, and how often the same rule fires on the untouched originals.
Originals are not guaranteed fault-free, so the false-alarm figure is an upper bound.
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from chipqc.data import load_manifest  # noqa: E402
from chipqc.degrade import FAULTS, degrade  # noqa: E402
from chipqc.descriptors import acquisition_descriptors  # noqa: E402
from chipqc.frames import open_frame, to_grey  # noqa: E402


def one(args):
    path, seed = args
    rng, grey = np.random.default_rng(seed), to_grey(open_frame(path))
    rows = [("none", 0, acquisition_descriptors(grey))]
    for fault, spec in FAULTS.items():
        for level in spec["levels"]:
            rows.append((fault, level, acquisition_descriptors(degrade(grey, fault, level, rng))))
    return rows


def fires(d, limits):
    return {"occluded": d["occluded_block_frac"] > limits["occluded_block_frac_max"], "motion_streak": d["streak_anisotropy"] > limits["streak_anisotropy_max"],
            "defocus": d["log_sharpness"] < limits["log_sharpness_min"], "under_exposed": d["median_brightness"] < limits["median_brightness_min"]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", type=Path, default=ROOT / "data/raw/OOC_image_dataset")
    ap.add_argument("--model", type=Path, default=ROOT / "models/guardian-v2")
    ap.add_argument("--per-line", type=int, default=60)
    a = ap.parse_args()
    limits = json.loads((a.model / "model.json").read_text())["acquisition_limits"]
    man = load_manifest(ROOT / "data/manifest.csv")
    pick = man.groupby("cell_line", group_keys=False).sample(a.per_line, random_state=1, replace=False).reset_index(drop=True)
    with Pool(8) as pool:
        res = pool.map(one, [(a.images / p, i) for i, p in enumerate(pick.path)], chunksize=4)
    out = {"frames": int(len(pick)), "limits": limits, "faults": {}}
    originals = [fires(rows[0][2], limits) for rows in res]
    for fault, spec in FAULTS.items():
        rule = spec["rule"]
        rows = []
        for level in spec["levels"]:
            hit = [fires(d, limits) for rr in res for (f, lv, d) in rr if f == fault and lv == level]
            rows.append({"level": level, "caught_by_its_rule": float(np.mean([h[rule] for h in hit])), "caught_by_any_rule": float(np.mean([any(h.values()) for h in hit]))})
        out["faults"][fault] = {"unit": spec["unit"], "rule": rule, "rule_fires_on_originals": float(np.mean([o[rule] for o in originals])), "levels": rows}
        print(fault, "| originals", round(out["faults"][fault]["rule_fires_on_originals"], 3), "|", "  ".join(f"{r['level']}: {r['caught_by_its_rule']:.2f}" for r in rows))
    out["any_rule_fires_on_originals"] = float(np.mean([any(o.values()) for o in originals]))
    (ROOT / "results/degradation_study.json").write_text(json.dumps(out, indent=1))
