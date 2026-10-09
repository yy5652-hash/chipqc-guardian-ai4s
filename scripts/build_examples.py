"""Bundle a small example run: real frames from two acquisition dates that the demo model never saw.

Frames are chosen by rule from the demo model's own outcomes so that every outcome appears, including a case
where the model and the experts disagree. Source: Movčana et al., Zenodo 10.5281/zenodo.10203721, CC BY 4.0.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from chipqc.data import load_manifest  # noqa: E402
from chipqc.descriptors import acquisition_flags  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--images", type=Path, default=ROOT / "data/raw/OOC_image_dataset")
ap.add_argument("--model", type=Path, default=ROOT / "models/guardian-v2-demo")
ap.add_argument("--features", type=Path, default=ROOT / "features/dinov2_vits14_4x4_l4.npz")
a = ap.parse_args()

spec = json.loads((a.model / "model.json").read_text())
man = load_manifest(ROOT / "data/manifest.csv")
X = np.load(a.features)["embeddings"]
d = pd.read_csv(ROOT / "features/acquisition_descriptors.csv")
A, B = spec["platt"]
man["p"] = 1 / (1 + np.exp(-(A * (X @ np.asarray(spec["cell_weights"]) + spec["cell_bias"]) + B)))
man["n_flags"] = [len(acquisition_flags(r, spec["acquisition_limits"])) for r in d.drop(columns="image_id").to_dict("records")]
t_pass = spec["thresholds"][str(spec["default_error_target"])]["t_pass"]
held = man[man.date.isin(spec["held_out_dates"])].copy()
held["outcome"] = np.where(held.n_flags > 0, "REACQUIRE", np.where(held.p >= t_pass, "PASS", "REVIEW"))
rng = np.random.default_rng(3)
take = lambda q, n: q.sample(min(n, len(q)), random_state=int(rng.integers(1 << 30)))
picks = pd.concat([
    take(held[(held.outcome == "PASS") & (held.label_good == 1) & (held.cell_line == "HPMEC")], 1),
    take(held[(held.outcome == "PASS") & (held.label_good == 1) & (held.cell_line == "A549")], 1),
    take(held[(held.outcome == "PASS") & (held.label_good == 0)], 1),                                   # a false accept, shown on purpose
    take(held[(held.outcome == "REVIEW") & (held.p > 0.5) & (held.label_good == 1)], 1),
    take(held[(held.outcome == "REVIEW") & (held.p < 0.3) & (held.label_good == 0) & (held.cell_line == "CACO")], 1),
    take(held[(held.outcome == "REVIEW") & (held.p < 0.3) & (held.label_good == 0) & (held.cell_line == "HPMEC")], 1),
    take(held[(held.outcome == "REVIEW") & (held.p < 0.3) & (held.label_good == 1)], 1),                # low score, experts said good
    held[held.image_id.isin(["230524_30", "230524_35"])],                                               # motion streak, occluded field
    take(held[(held.outcome == "REACQUIRE") & (held.cell_line == "A549")], 1),
]).drop_duplicates("image_id")
out = ROOT / "examples"
out.mkdir(exist_ok=True)
for f in list(out.glob("*.jpg")) + list(out.glob("*.png")):
    f.unlink()
frames = []
for r in picks.itertuples():
    with Image.open(a.images / r.path) as im:                      # lossless, so the bundled copy scores like the original
        im.convert("RGB").resize(tuple(spec["frame_size"]), Image.BICUBIC).save(out / f"{r.image_id}.png", optimize=True)
    frames.append({"file": f"{r.image_id}.png", "image_id": r.image_id, "cell_line": r.cell_line, "expert_label": "good" if r.label_good else "bad", "culture_day_bin": r.day_bin})
note = (f"{len(frames)} real frames from acquisition dates {' and '.join(spec['held_out_dates'])} of the Organ-on-a-Chip Image Dataset (Movčana et al., Zenodo, "
        f"doi:10.5281/zenodo.10203721, CC BY 4.0), resized to {spec['frame_size'][0]} × {spec['frame_size'][1]} px. They are scored by a model fitted without those two dates.")
(out / "examples.json").write_text(json.dumps({"note": note, "frames": frames}, indent=1))
print(pd.DataFrame(frames).assign(p=picks.p.round(2).to_numpy(), outcome=picks.outcome.to_numpy())[["image_id", "cell_line", "expert_label", "p", "outcome"]].to_string(index=False))
print(dict(held.outcome.value_counts()), "of", len(held), "held-out frames | pass threshold", t_pass)
