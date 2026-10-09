"""A new camera: how badly does the model transfer, and how many labelled sessions repair it?
-> results/camera_transfer.json   (needs features/camera_format.csv, written by scripts/camera_formats.py)

The reference dataset mixes a grey 2056 x 1542 camera and a colour camera. We treat one as the "new" instrument:
train on the other camera plus k acquisition dates of the new one, test on the new camera's remaining dates.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from chipqc.data import load_manifest  # noqa: E402
from chipqc.protocol import fit_head, head_logits  # noqa: E402

MAIN = sys.argv[1] if len(sys.argv) > 1 else "dinov2_vits14_4x4_l4"
man = load_manifest(ROOT / "data/manifest.csv")
cam = pd.read_csv(ROOT / "features/camera_format.csv").camera.str.startswith("grey").to_numpy()
X = np.load(ROOT / f"features/{MAIN}.npz")["embeddings"].astype(np.float64)
y, dates = man.label_good.to_numpy(), man.date.to_numpy()
oof = np.load(ROOT / f"results/oof/{MAIN}.npz")["p_good"]
C = 0.01
out = {"within_camera_dates_held_out": {"grey": {"frames": int(cam.sum()), "dates": int(len(np.unique(dates[cam]))), "auroc": float(roc_auc_score(y[cam], oof[cam]))},
                                        "colour": {"frames": int((~cam).sum()), "dates": int(len(np.unique(dates[~cam]))), "auroc": float(roc_auc_score(y[~cam], oof[~cam]))}}, "adaptation": {}}
rng = np.random.default_rng(0)
for target_name, target in (("colour", ~cam), ("grey", cam)):
    target_dates = np.unique(dates[target])
    # dates that contain both cameras count as target dates here; frames of the other camera on those dates stay out of training
    source = ~target & ~np.isin(dates, target_dates)
    rows = []
    for k in (0, 1, 2, 4, 8):
        aucs, frames = [], []
        for _ in range(30 if k else 1):
            chosen = rng.choice(target_dates, k, replace=False) if k else np.array([])
            tr = source | (target & np.isin(dates, chosen))
            te = target & ~np.isin(dates, chosen)
            if len(np.unique(y[te])) < 2:
                continue
            aucs.append(roc_auc_score(y[te], head_logits(fit_head(X[tr], y[tr], C), X[te])))
            frames.append(int((target & np.isin(dates, chosen)).sum()))
        rows.append({"labelled_dates_from_new_camera": k, "mean_labelled_frames": float(np.mean(frames)), "auroc_mean": float(np.mean(aucs)),
                     "auroc_p10": float(np.percentile(aucs, 10)), "auroc_p90": float(np.percentile(aucs, 90))})
        print(target_name, rows[-1])
    out["adaptation"][target_name] = {"new_camera_dates": int(len(target_dates)), "new_camera_frames": int(target.sum()), "rows": rows}
(ROOT / "results/camera_transfer.json").write_text(json.dumps(out, indent=1))

# zero labelled frames from the new camera, every representation we have features for
mixed = np.isin(dates, [d for d in np.unique(dates) if len(set(cam[dates == d])) > 1])
rows = {}
for path in sorted((ROOT / "features").glob("*.npz")):
    F = np.load(path)["embeddings"].astype(np.float64)
    rows["main" if path.stem == MAIN else path.stem] = {
        "grey_to_colour": float(roc_auc_score(y[~cam], head_logits(fit_head(F[cam & ~mixed], y[cam & ~mixed], C), F[~cam]))),
        "colour_to_grey": float(roc_auc_score(y[cam], head_logits(fit_head(F[~cam & ~mixed], y[~cam & ~mixed], C), F[cam])))}
(ROOT / "results/camera_backbones.json").write_text(json.dumps(rows, indent=1))
print({k: (round(v["grey_to_colour"], 3), round(v["colour_to_grey"], 3)) for k, v in rows.items()})
