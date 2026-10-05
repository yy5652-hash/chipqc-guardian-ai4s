"""Reproduce every reported number from the released features (no images needed).

    python evaluate.py                 # all analyses -> results/*.json and results/oof/*.npz
    python evaluate.py --only headline

Each analysis writes one JSON file; the README, the model card and the technical report quote those files.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from chipqc.data import load_manifest  # noqa: E402
from chipqc.protocol import (C_GRID, ERROR_TARGETS, date_bootstrap, date_folds, fit_head, head_logits, nested_evaluation,  # noqa: E402
                             paired_date_bootstrap, select_on_training_dates, sigmoid, summarise)
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

MAIN = "dinov2_vits14"
#: feature file -> how the frame was shown to the backbone
REPRESENTATIONS = {
    "mobilenet_v2_224crop": "MobileNetV2, 224 px centre crop (first submission)",
    "mobilenet_v2_672": "MobileNetV2, whole frame, 672 px",
    "mobilenet_v2_1344": "MobileNetV2, whole frame, 1344 px",
    "mobilenet_v2_2048": "MobileNetV2, whole frame, 2048 px (native)",
    "resnet50_1344": "ResNet-50, whole frame, 1344 px",
    "convnext_tiny_1344": "ConvNeXt-Tiny, whole frame, 1344 px",
    "dinov2_vits14_2x2": "DINOv2 ViT-S/14, whole frame at 896 px, 2 x 2 tiles",
    "dinov2_vits14": "DINOv2 ViT-S/14, whole frame at 1344 px, 3 x 3 tiles (released model)",
    "dinov2_vits14_4x4": "DINOv2 ViT-S/14, whole frame at 1792 px, 4 x 4 tiles",
    "dinov2_vitb14": "DINOv2 ViT-B/14, whole frame at 1344 px, 3 x 3 tiles",
}


def features(name: str, man) -> np.ndarray | None:
    path = ROOT / f"features/{name}.npz"
    if not path.exists():
        return None
    f = np.load(path, allow_pickle=False)
    assert (f["image_id"] == man.image_id.to_numpy().astype(str)).all(), f"{name}: rows must follow data/manifest.csv"
    return f["embeddings"].astype(np.float64)


def mean_outcomes(outcomes):
    return {str(t): {k: float(np.mean([o[k] for o in rows])) for k in rows[0]} for t, rows in outcomes.items()}


def protocol_row(X, y, dates, strata=None, repeats=3, n_boot=2000):
    raw, cal, chosen, outcomes, decisions = nested_evaluation(X, y, dates, repeats=repeats)
    s = summarise(y, raw, cal, dates, strata, n_boot)
    s["chosen_C"] = {str(c): chosen.count(c) for c in C_GRID}
    s["triage"] = mean_outcomes(outcomes)
    protocol_row.last_decisions = decisions
    return s, raw, cal


def headline(man, out):
    X, y, dates = features(MAIN, man), man.label_good.to_numpy(), man.date.to_numpy()
    s, raw, cal = protocol_row(X, y, dates, {"cell_line": man.cell_line.to_numpy(), "day_bin": man.day_bin.to_numpy()})
    s.update(main=MAIN, representation=REPRESENTATIONS[MAIN], frames=int(len(y)), dates=int(len(np.unique(dates))),
             protocol="5 folds over acquisition dates x 3 repeats; C, calibrator and thresholds chosen on training dates only; 95 % intervals resample dates")
    (out / "oof").mkdir(exist_ok=True)
    np.savez_compressed(out / "oof" / f"{MAIN}.npz", p_good=cal.astype(np.float32), p_good_uncalibrated=raw.astype(np.float32), image_id=man.image_id.to_numpy().astype(str),
                        **{f"decision_at_{t}": d.astype(np.int8) for t, d in protocol_row.last_decisions.items()})
    return s


def representations(man, out):
    y, dates = man.label_good.to_numpy(), man.date.to_numpy()
    rows, oof = {}, {}
    for name, text in REPRESENTATIONS.items():
        X = features(name, man)
        if X is None:
            continue
        s, _, oof[name] = protocol_row(X, y, dates)
        rows[name] = {"representation": text, "dim": int(X.shape[1]), **{k: s[k] for k in ("auroc", "balanced_accuracy", "accuracy", "brier")}, "triage_at_10pct": s["triage"]["0.1"]}
        if name != MAIN:
            np.savez_compressed(out / "oof" / f"{name}.npz", p_good=oof[name].astype(np.float32), image_id=man.image_id.to_numpy().astype(str))
    for name in rows:                                               # the same resampled dates score both models, so the interval is for the difference itself
        if name != MAIN:
            rows[name]["auroc_minus_released"] = {"value": rows[name]["auroc"]["value"] - rows[MAIN]["auroc"]["value"],
                                                  "ci95": list(paired_date_bootstrap(roc_auc_score, y, oof[name], oof[MAIN], dates))}
    return rows


def baselines(man, out):
    y, dates = man.label_good.to_numpy(), man.date.to_numpy()
    onehot = lambda v: (v[:, None] == np.unique(v)[None, :]).astype(np.float64)
    meta = np.concatenate([onehot(man.cell_line.to_numpy()), onehot(man.day_bin.to_numpy())], axis=1)
    rows = {"always_good": {"accuracy": float(y.mean()), "balanced_accuracy": 0.5, "auroc": 0.5}}
    s, _, _ = protocol_row(meta, y, dates, n_boot=1000)
    rows["metadata_only"] = {"inputs": "cell line and culture-day bin, no image", **{k: s[k] for k in ("auroc", "balanced_accuracy", "accuracy")}}
    import pandas as pd
    d = pd.read_csv(ROOT / "features/acquisition_descriptors.csv").drop(columns="image_id").to_numpy(dtype=np.float64)
    s, _, _ = protocol_row(d, y, dates, n_boot=1000)
    rows["acquisition_descriptors_only"] = {"inputs": "five transparent acquisition descriptors", **{k: s[k] for k in ("auroc", "balanced_accuracy", "accuracy")}}
    return rows


def leakage(man, out):
    X, y, dates, split = features(MAIN, man), man.label_good.to_numpy(), man.date.to_numpy(), man.author_split.to_numpy()
    tr, va, te = split == "train", split == "val", split == "test"
    c = max(C_GRID, key=lambda c: roc_auc_score(y[va], head_logits(fit_head(X[tr], y[tr], c), X[va])))
    z = head_logits(fit_head(X[tr | va], y[tr | va], c), X[te])
    authors = {"test_frames": int(te.sum()), "C": c, "accuracy": float(accuracy_score(y[te], z > 0)), "auroc": float(roc_auc_score(y[te], z)),
               "precision_good": float(precision_score(y[te], z > 0)), "recall_good": float(recall_score(y[te], z > 0)),
               "balanced_accuracy": float(balanced_accuracy_score(y[te], z > 0)), "test_dates": int(len(set(dates[te]))),
               "test_dates_also_in_training": int(len(set(dates[te]) & set(dates[tr | va]))),
               "published_baseline": {"accuracy": 0.81, "precision": 0.79, "recall": 0.78, "source": "Movčana et al., Data 9(2):28, 2024"}}
    p = np.zeros(len(y))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        p[b] = head_logits(fit_head(X[a], y[a], 0.01), X[b])
    return {"authors_split": authors, "random_image_folds": {"auroc": float(roc_auc_score(y, p)), "accuracy": float(accuracy_score(y, p > 0))}}


def first_submission_split(man, out):
    """The frozen 35 / 12 / 12 date split published with our first submission: fit on its training and calibration
    dates, score its 12 test dates once. The first submission reported AUROC 0.772 and balanced accuracy 0.712 there."""
    split = json.loads((ROOT / "data/first_submission_split.json").read_text())["split_groups"]
    y, dates = man.label_good.to_numpy(), man.date.to_numpy()
    tr, te = np.isin(dates, split["train"] + split["calibration"]), np.isin(dates, split["test"])
    rows = {"test_frames": int(te.sum()), "test_dates": int(len(split["test"])), "first_submission": {"auroc": 0.772, "balanced_accuracy": 0.712, "accuracy": 0.718}}
    for name in (MAIN, "mobilenet_v2_224crop"):
        X = features(name, man)
        if X is None:
            continue
        c, z_in = select_on_training_dates(X[tr], y[tr], dates[tr], 0)
        from chipqc.protocol import fit_platt
        a, b = fit_platt(z_in, y[tr])
        p = sigmoid(a * head_logits(fit_head(X[tr], y[tr], c), X[te]) + b)
        lo, hi = date_bootstrap(roc_auc_score, y[te], p, dates[te], 2000)
        rows[name] = {"C": c, "auroc": float(roc_auc_score(y[te], p)), "ci95": [lo, hi], "balanced_accuracy": float(balanced_accuracy_score(y[te], p > 0.5)), "accuracy": float(accuracy_score(y[te], p > 0.5))}
    return rows


def unseen_cell_line(man, out):
    """Leave one cell line out of training entirely. Also run for a supervised backbone, as a comparison."""
    y, dates, line = man.label_good.to_numpy(), man.date.to_numpy(), man.cell_line.to_numpy()

    def run(name, oof):
        X, rows = features(name, man), {}
        for cell in np.unique(line):
            tr, te = line != cell, line == cell
            c, _ = select_on_training_dates(X[tr], y[tr], dates[tr], 0)
            z = head_logits(fit_head(X[tr], y[tr], c), X[te])
            lo, hi = date_bootstrap(roc_auc_score, y[te], z, dates[te], 1000)
            rows[cell] = {"frames": int(te.sum()), "dates": int(len(np.unique(dates[te]))), "auroc_line_never_seen": float(roc_auc_score(y[te], z)), "ci95": [lo, hi],
                          "auroc_line_seen_dates_held_out": float(roc_auc_score(y[te], oof[te]))}
        return rows

    rows = run(MAIN, np.load(out / "oof" / f"{MAIN}.npz")["p_good"])
    other = "convnext_tiny_1344"
    if features(other, man) is not None and (out / "oof" / f"{other}.npz").exists():
        rows["_comparison"] = {"representation": REPRESENTATIONS[other], "lines": run(other, np.load(out / "oof" / f"{other}.npz")["p_good"])}
    return rows


def label_structure(man, out):
    y, dates, idx = man.label_good.to_numpy(), man.date.to_numpy(), man.frame_index.to_numpy()
    same, lengths = [], []
    for d in np.unique(dates):
        r = np.flatnonzero(dates == d); r = r[np.argsort(idx[r])]
        if len(r) > 1:
            same.append(y[r][1:] == y[r][:-1])
            change = np.flatnonzero(np.diff(y[r]) != 0)
            lengths.extend(np.diff(np.r_[0, change + 1, len(r)]).tolist())
    same = np.concatenate(same)
    per_date = np.array([y[dates == d].mean() for d in np.unique(dates)])
    return {"adjacent_frames_share_label": float(same.mean()), "expected_if_independent": float(y.mean() ** 2 + (1 - y.mean()) ** 2),
            "mean_run_length": float(np.mean(lengths)), "median_run_length": float(np.median(lengths)),
            "dates": int(len(per_date)), "dates_all_good": int((per_date == 1).sum()), "dates_all_bad": int((per_date == 0).sum()),
            "good_fraction_by_date_min_max": [float(per_date.min()), float(per_date.max())]}


def acquisition_gate(man, out):
    """How often each acquisition rule fires on the reference frames and what the experts said about those frames."""
    import pandas as pd
    from chipqc.descriptors import acquisition_flags
    spec = json.loads((ROOT / "models/guardian-v2/model.json").read_text())
    limits = spec["acquisition_limits"]
    d = pd.read_csv(ROOT / "features/acquisition_descriptors.csv")
    y = man.label_good.to_numpy()
    rules = {"occluded": d.occluded_block_frac > limits["occluded_block_frac_max"], "motion_streak": d.streak_anisotropy > limits["streak_anisotropy_max"],
             "defocus": d.log_sharpness < limits["log_sharpness_min"], "under_exposed": d.median_brightness < limits["median_brightness_min"]}
    any_flag = np.zeros(len(y), dtype=bool)
    rows = {}
    for name, hit in rules.items():
        hit = hit.to_numpy(); any_flag |= hit
        rows[name] = {"frames": int(hit.sum()), "share": float(hit.mean()), "expert_bad_rate": float(1 - y[hit].mean())}
    rows["any_rule"] = {"frames": int(any_flag.sum()), "share": float(any_flag.mean()), "expert_bad_rate": float(1 - y[any_flag].mean())}
    rows["no_rule"] = {"frames": int((~any_flag).sum()), "share": float((~any_flag).mean()), "expert_bad_rate": float(1 - y[~any_flag].mean())}
    return {"limits": limits, "overall_bad_rate": float(1 - y.mean()), "rules": rows}


def gate_flags(d, dates, repeats=3):
    """repeats x frames: does the acquisition gate fire? Limits are fitted on each fold's training dates."""
    flagged = np.zeros((repeats, len(dates)), dtype=bool)
    for r in range(repeats):
        for tr, te in date_folds(dates, 5, r):                         # the protocol's own folds
            x = d.iloc[te]
            flagged[r, te] = ((x.occluded_block_frac > 0.25) | (x.streak_anisotropy > np.percentile(d.streak_anisotropy[tr], 95))
                              | (x.log_sharpness < np.percentile(d.log_sharpness[tr], 2)) | (x.median_brightness < np.percentile(d.median_brightness[tr], 5))).to_numpy()
    return flagged


def system_outcomes(man, out):
    """The assembled system on dates it never saw: acquisition gate, then automatic PASS at a target error, else REVIEW.
    Pass thresholds are chosen on training-date frames that passed the gate. Also reported: what an automatic FAIL
    would have cost (the reason the system does not have one) and how well the score orders the review queue."""
    import pandas as pd
    d = pd.read_csv(ROOT / "features/acquisition_descriptors.csv")
    X, y, dates = features(MAIN, man), man.label_good.to_numpy(), man.date.to_numpy()
    flagged = gate_flags(d, dates)
    _, cal, _, _, decisions = nested_evaluation(X, y, dates, eligible=~flagged)
    n, result = len(y), {}
    for t in ERROR_TARGETS:
        rows = []
        for r in range(flagged.shape[0]):
            dec = decisions[t][r]
            passed, reacq = (dec == 1) & ~flagged[r], flagged[r]
            review = ~passed & ~reacq
            would_fail = (dec == -1) & ~flagged[r]
            rows.append({"pass_frames": passed.sum(), "pass_bad": (passed & (y == 0)).sum(), "reacquire_frames": reacq.sum(), "reacquire_bad": (reacq & (y == 0)).sum(),
                         "review_frames": review.sum(), "review_bad": (review & (y == 0)).sum(), "would_fail_frames": would_fail.sum(), "would_fail_good": (would_fail & (y == 1)).sum(),
                         "review_queue_auroc": roc_auc_score(y[review], cal[review]),
                         "bad_found_in_first_third_of_queue": float((y[review][np.argsort(cal[review])[: review.sum() // 3]] == 0).sum() / max(1, (y[review] == 0).sum()))})
        m = {k: float(np.mean([row[k] for row in rows])) for k in rows[0]}
        result[str(t)] = {
            "PASS": {"frames": m["pass_frames"], "expert_good": m["pass_frames"] - m["pass_bad"], "expert_bad": m["pass_bad"]},
            "REVIEW": {"frames": m["review_frames"], "expert_good": m["review_frames"] - m["review_bad"], "expert_bad": m["review_bad"]},
            "REACQUIRE": {"frames": m["reacquire_frames"], "expert_good": m["reacquire_frames"] - m["reacquire_bad"], "expert_bad": m["reacquire_bad"]},
            "summary": {"passed_automatically": m["pass_frames"] / n, "bad_among_passed": m["pass_bad"] / max(1.0, m["pass_frames"]),
                        "sent_to_reacquire": m["reacquire_frames"] / n, "bad_among_reacquire": m["reacquire_bad"] / max(1.0, m["reacquire_frames"]),
                        "left_for_a_person": m["review_frames"] / n, "bad_among_review": m["review_bad"] / max(1.0, m["review_frames"]),
                        "review_queue_auroc": m["review_queue_auroc"], "bad_found_in_first_third_of_queue": m["bad_found_in_first_third_of_queue"],
                        "automatic_fail_if_it_existed": {"frames": m["would_fail_frames"] / n, "good_among_failed": m["would_fail_good"] / max(1.0, m["would_fail_frames"])}}}
    return result


ANALYSES = {"headline": headline, "representations": representations, "baselines": baselines, "leakage": leakage, "first_submission_split": first_submission_split,
            "unseen_cell_line": unseen_cell_line, "label_structure": label_structure, "acquisition_gate": acquisition_gate,
            "system_outcomes": system_outcomes}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", choices=list(ANALYSES))
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    a = ap.parse_args()
    a.out.mkdir(exist_ok=True)
    man = load_manifest(ROOT / "data/manifest.csv")
    for name, fn in ANALYSES.items():
        if a.only and name != a.only:
            continue
        t0 = time.time()
        (a.out / f"{name}.json").write_text(json.dumps(fn(man, a.out), indent=1))
        print(f"{name}: {time.time() - t0:.0f} s -> results/{name}.json", flush=True)
