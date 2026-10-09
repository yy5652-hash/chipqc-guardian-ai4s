"""Fit the released model on all reference frames and write models/<name>/model.json + reference.npz.

Regularisation strength, calibrator and pass thresholds come from date-grouped cross-fitting over all 59
dates, so each of them is chosen on predictions for dates the scoring model had not seen. The evaluation that
supports the release is produced separately by evaluate.py (nested, nothing tuned on scored dates).
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from chipqc.backbone import BACKBONES  # noqa: E402
from chipqc.data import load_manifest  # noqa: E402
from chipqc.protocol import C_GRID, OFFERED_TARGETS, date_folds, fit_head, fit_platt, head_logits, sigmoid  # noqa: E402
from chipqc.triage import choose_thresholds  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--features", type=Path, default=ROOT / "features/dinov2_vits14_4x4_l4.npz")
ap.add_argument("--descriptors", type=Path, default=ROOT / "features/acquisition_descriptors.csv")
ap.add_argument("--manifest", type=Path, default=ROOT / "data/manifest.csv")
ap.add_argument("--out", type=Path, default=ROOT / "models/guardian-v2")
ap.add_argument("--default-error-target", type=float, default=0.10)
ap.add_argument("--exclude-dates", nargs="*", default=[], help="hold these acquisition dates out (used for the demo model, so bundled examples are unseen)")
ap.add_argument("--atlas", default="atlas", help="folder with reference thumbnails, relative to the model folder")
a = ap.parse_args()

man = load_manifest(a.manifest)
f = np.load(a.features, allow_pickle=False)
assert (f["image_id"] == man.image_id.to_numpy().astype(str)).all(), "feature rows must follow the manifest"
keep = ~man.date.isin(a.exclude_dates).to_numpy()
man = man[keep].reset_index(drop=True)
X, y, dates = f["embeddings"].astype(np.float64)[keep], man.label_good.to_numpy(), man.date.to_numpy()

# cross-fitted logits for every C (3 repeats of 5 date folds)
z = {c: np.zeros(len(y)) for c in C_GRID}
for r in range(3):
    for tr, te in date_folds(dates, 5, r):
        for c in C_GRID:
            z[c][te] += head_logits(fit_head(X[tr], y[tr], c), X[te]) / 3
C = max(C_GRID, key=lambda c: roc_auc_score(y, z[c]))
A, B = fit_platt(z[C], y)
p_cross = sigmoid(A * z[C] + B)
thresholds = {}
for t in OFFERED_TARGETS:                               # a released model has pass thresholds only: there is no automatic FAIL
    thresholds[str(t)] = {"t_pass": round(choose_thresholds(p_cross, y, t)[1], 4)}

scaler, clf = fit_head(X, y, C)
w, mu, sd = clf.coef_[0], scaler.mean_, scaler.scale_
cell_weights = w / sd                                   # frame logit = mean over cells of (cell . cell_weights + cell_bias)
cell_bias = float(clf.intercept_[0] - (w * mu / sd).sum())

d = pd.read_csv(a.descriptors)[keep].reset_index(drop=True)
assert (d.image_id.astype(str) == man.image_id).all()
limits = {"occluded_block_frac_max": 0.25,                                         # fixed physical choice
          "streak_anisotropy_max": round(float(np.percentile(d.streak_anisotropy, 95)), 4),   # 5 % tail of the reference frames
          "log_sharpness_min": round(float(np.percentile(d.log_sharpness, 2)), 4),            # 2 % tail
          "median_brightness_min": round(float(np.percentile(d.median_brightness, 5)), 4)}    # 5 % tail

def evaluation_summary():
    """Headline numbers of the date-held-out evaluation (results/*.json), embedded so the model carries its evidence."""
    r = ROOT / "results"
    if not (r / "headline.json").exists() or not (r / "system_outcomes.json").exists():
        return None
    h, so = json.loads((r / "headline.json").read_text()), json.loads((r / "system_outcomes.json").read_text())
    return {"protocol": h["protocol"], "frames": h["frames"], "dates": h["dates"], "auroc": h["auroc"], "balanced_accuracy": h["balanced_accuracy"],
            "ece_calibrated": h["ece_calibrated"], "system": {t: {"summary": v["summary"]} for t, v in so.items()}}


spec = {
    "name": a.out.name, "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "backbone": str(f["backbone"]), "tile": list(BACKBONES[str(f["backbone"])]["tile"]), "grid": list(BACKBONES[str(f["backbone"])]["grid"]),
    "frame_size": [BACKBONES[str(f["backbone"])]["tile"][i] * BACKBONES[str(f["backbone"])]["grid"][i] for i in (0, 1)],
    "cell_weights": [round(float(v), 8) for v in cell_weights], "cell_bias": round(cell_bias, 8),
    "platt": [round(A, 6), round(B, 6)], "thresholds": thresholds, "default_error_target": a.default_error_target,
    "acquisition_limits": limits, "atlas": a.atlas, "held_out_dates": list(a.exclude_dates),
    "embedding_mean": [round(float(v), 6) for v in mu], "embedding_std": [round(float(v), 6) for v in sd],
    "training": {"frames": int(len(y)), "dates": int(len(np.unique(dates))), "good": int(y.sum()), "bad": int((1 - y).sum()), "C": C,
                 "cross_fitted_auroc": round(float(roc_auc_score(y, z[C])), 4),
                 "data": "Movčana et al., Organ-on-a-Chip (OOC) Image Dataset, Zenodo 10.5281/zenodo.10203721, CC BY 4.0"},
    "evaluation": evaluation_summary(),
}
a.out.mkdir(parents=True, exist_ok=True)
(a.out / "model.json").write_text(json.dumps(spec, indent=1))
unit = (X - mu) / sd
np.savez_compressed(a.out / "reference.npz", embeddings=unit.astype(np.float16), image_id=man.image_id.to_numpy().astype(str),
                    label_good=y.astype(np.int8), cell_line=man.cell_line.to_numpy().astype(str))
print(f"C={C}  cross-fitted AUROC {spec['training']['cross_fitted_auroc']}  platt {spec['platt']}  thresholds {thresholds}")
print(f"acquisition limits {limits}")
print(f"wrote {a.out}/model.json and reference.npz")
