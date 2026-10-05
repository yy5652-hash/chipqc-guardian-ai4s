"""Draw every figure of the technical report from results/*.json -> reports/figures/*.png

    python scripts/make_figures.py                     # charts only
    python scripts/make_figures.py --images data/raw/OOC_image_dataset   # also the panels that show real frames
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from chipqc.data import load_manifest  # noqa: E402

R, OUT = ROOT / "results", ROOT / "reports/figures"
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7", "#e34948"   # validated categorical slots
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "font.size": 9,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False, "lines.linewidth": 2.0,
})
load = lambda name: json.loads((R / f"{name}.json").read_text())


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print("wrote", f"reports/figures/{name}.png")


def hbars(ax, labels, values, colour=BLUE, ci=None, fmt="{:.3f}", thickness=0.46, xlim=None):
    """Thin horizontal bars from one baseline, value at the tip, optional interval whiskers."""
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, height=thickness, color=colour, linewidth=0)
    if ci is not None:
        for yy, c in zip(y, ci):
            if c:
                ax.plot(c, [yy, yy], color=INK, linewidth=1.0, solid_capstyle="butt")
    for yy, v, c in zip(y, values, ci if ci is not None else [None] * len(values)):
        ax.text((c[1] if c else v) + 0.006 * ((xlim[1] - xlim[0]) if xlim else 1), yy, fmt.format(v), va="center", ha="left", color=INK, fontsize=8.5)
    ax.set_yticks(y, labels)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    if xlim:
        ax.set_xlim(*xlim)


def fig_protocols():
    lk, hd = load("leakage"), load("headline")
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.4, 2.0), gridspec_kw={"width_ratios": [1, 1.2], "wspace": 1.05})
    au = lk["authors_split"]
    hbars(a, ["Published baseline\n(Movčana et al. 2024)", "This work,\nsame split"], [au["published_baseline"]["accuracy"], au["accuracy"]], colour=BLUE, fmt="{:.2f}", thickness=0.36, xlim=(0.5, 1.0))
    a.set_xlabel("accuracy on the authors' test folder")
    a.set_title("Same split as the dataset paper")
    labels = ["Authors' folders\n(57 of 57 test dates also in training)", "Random image folds", "Whole acquisition dates held out\n(this work's headline protocol)"]
    hbars(b, labels, [au["auroc"], lk["random_image_folds"]["auroc"], hd["auroc"]["value"]], colour=BLUE, ci=[None, None, hd["auroc"]["ci95"]], thickness=0.4, xlim=(0.5, 1.0))
    b.set_xlabel("AUROC, same features and classifier")
    b.set_title("What the hold-out scheme does to the number")
    save(fig, "fig_protocols")


def fig_representations():
    rep = load("representations")
    order = [k for k in ("mobilenet_v2_224crop", "mobilenet_v2_672", "mobilenet_v2_1344", "mobilenet_v2_2048", "resnet50_1344", "convnext_tiny_1344", "dinov2_vits14_2x2", "dinov2_vits14", "dinov2_vits14_4x4", "dinov2_vitb14") if k in rep]
    fig, ax = plt.subplots(figsize=(7.6, 0.42 * len(order) + 1.0))
    hbars(ax, [rep[k]["representation"] for k in order], [rep[k]["auroc"]["value"] for k in order], ci=[rep[k]["auroc"]["ci95"] for k in order], xlim=(0.5, 0.95))
    ax.set_xlabel("AUROC on held-out acquisition dates (bar: point estimate, line: 95 % interval over dates)")
    ax.set_title("Seeing the whole frame at high resolution is what moves the result")
    save(fig, "fig_representations")


def roc(y, p):
    order = np.argsort(-p)
    tp, fp = np.cumsum(y[order]), np.cumsum(1 - y[order])
    return np.r_[0, fp / fp[-1]], np.r_[0, tp / tp[-1]]


def fig_roc_calibration(man):
    y = man.label_good.to_numpy()
    main = np.load(R / "oof/dinov2_vits14.npz")
    rep = load("representations")
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 3.6), gridspec_kw={"wspace": 0.32})
    series = [("dinov2_vits14", BLUE, "released model, whole frame"), ("mobilenet_v2_224crop", ORANGE, "224 px centre crop (first submission)")]
    for name, colour, text in series:
        if not (R / f"oof/{name}.npz").exists():
            continue
        p = np.load(R / f"oof/{name}.npz")["p_good"]
        fpr, tpr = roc(y, p)
        a.plot(fpr, tpr, color=colour, label=f"{text}: {rep[name]['auroc']['value']:.3f}")
    a.plot([0, 1], [0, 1], color=AXIS, linewidth=1.0)
    a.set_xlabel("false positive rate (bad frames scored above the threshold)"); a.set_ylabel("true positive rate")
    a.set_title("ROC on held-out dates"); a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), title="AUROC", alignment="left"); a.set_aspect("equal")
    edges = np.linspace(0, 1, 11)
    for key, colour, text in (("p_good_uncalibrated", ORANGE, "before calibration"), ("p_good", BLUE, "after calibration")):
        p = main[key]
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo)
            if m.sum() >= 15:
                xs.append(p[m].mean()); ys.append(y[m].mean())
        b.plot(xs, ys, color=colour, marker="o", markersize=5.5, markeredgecolor=SURFACE, markeredgewidth=1.4, label=text)
    b.plot([0, 1], [0, 1], color=AXIS, linewidth=1.0)
    hd = load("headline")
    b.set_xlabel("predicted probability that the frame is good"); b.set_ylabel("share of frames experts called good")
    b.set_title(f"Calibration (expected error {hd['ece_raw']:.3f} before, {hd['ece_calibrated']:.3f} after)"); b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2)); b.set_aspect("equal")
    save(fig, "fig_roc_calibration")


def fig_risk_coverage(man):
    y = man.label_good.to_numpy()
    p = np.load(R / "oof/dinov2_vits14.npz")["p_good"]
    so = load("system_outcomes")
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 3.3), gridspec_kw={"wspace": 0.35, "width_ratios": [1, 1.15]})
    order = np.argsort(-p)
    n = np.arange(1, len(p) + 1)
    risk = np.cumsum(1 - y[order]) / n
    a.plot(n[30:] / len(p), risk[30:], color=BLUE)
    for t in ("0.05", "0.1", "0.15", "0.2"):
        s = so[t]
        x, e = s["summary"]["passed_automatically"], s["summary"]["bad_among_passed"]
        a.plot([x], [e], marker="o", markersize=7, color=BLUE, markeredgecolor=SURFACE, markeredgewidth=1.6)
        a.annotate(f"target {float(t):.0%}", (x, e), textcoords="offset points", xytext=(6, -11), fontsize=8, color=INK2)
    a.set_xlabel("share of frames passed automatically"); a.set_ylabel("share of passed frames that experts called bad")
    a.set_title("Automatic PASS: what you give up for coverage"); a.set_xlim(0, 1); a.set_ylim(0, 0.5)
    s = so["0.1"]
    names = ["PASS", "REVIEW", "REACQUIRE"]
    yy = np.arange(len(names))[::-1]
    good, bad = [s[k]["expert_good"] for k in names], [s[k]["expert_bad"] for k in names]
    gap = 0.004 * len(p)
    b.barh(yy, good, height=0.46, color=BLUE, linewidth=0, label="experts: good")
    b.barh(yy, [max(0, v - gap) for v in bad], left=[g + gap for g in good], height=0.46, color=RED, linewidth=0, label="experts: bad")
    for y0, g, bd in zip(yy, good, bad):
        b.text(g + bd + 0.012 * len(p), y0, f"{(g + bd) / len(p):.0%} of frames", va="center", fontsize=8.5, color=INK)
    b.set_yticks(yy, names); b.tick_params(axis="y", length=0); b.grid(axis="y", visible=False)
    b.set_xlabel("frames (of 3,072, each scored on a date the model never saw)"); b.set_xlim(0, len(p) * 0.78)
    b.set_title("Where the frames go at a 10 % target"); b.legend(loc="lower right")
    save(fig, "fig_risk_coverage")


def fig_cell_lines():
    u = {k: v for k, v in load("unseen_cell_line").items() if not k.startswith("_")}
    names = sorted(u, key=lambda k: -u[k]["frames"])
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    y = np.arange(len(names))[::-1].astype(float)
    ax.barh(y + 0.19, [u[k]["auroc_line_seen_dates_held_out"] for k in names], height=0.34, color=BLUE, linewidth=0, label="cell line seen in training, dates held out")
    ax.barh(y - 0.19, [u[k]["auroc_line_never_seen"] for k in names], height=0.34, color=ORANGE, linewidth=0, label="cell line never seen in training")
    for yy, k in zip(y, names):
        ax.text(u[k]["auroc_line_seen_dates_held_out"] + 0.006, yy + 0.19, f"{u[k]['auroc_line_seen_dates_held_out']:.2f}", va="center", fontsize=8)
        ax.text(u[k]["auroc_line_never_seen"] + 0.006, yy - 0.19, f"{u[k]['auroc_line_never_seen']:.2f}", va="center", fontsize=8)
    ax.set_yticks(y, [f"{k}  (n = {u[k]['frames']:,})" for k in names]); ax.tick_params(axis="y", length=0); ax.grid(axis="y", visible=False)
    drop = max(u[k]["auroc_line_seen_dates_held_out"] - u[k]["auroc_line_never_seen"] for k in names)
    ax.set_xlim(0.5, 1.08); ax.set_xlabel("AUROC")
    ax.set_title("Cell lines withheld from training: no loss on any of the six" if drop <= 0.03 else "Cell lines withheld from training")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    save(fig, "fig_cell_lines")


def fig_label_runs(man):
    ls = load("label_structure")
    dates = ["230525", "230320", "230524", "230425"]
    fig, axes = plt.subplots(len(dates), 1, figsize=(9.2, 0.62 * len(dates) + 0.9), gridspec_kw={"hspace": 0.9})
    for ax, d in zip(axes, dates):
        s = man[man.date == d].sort_values("frame_index")
        yv = s.label_good.to_numpy()
        ax.bar(np.arange(len(yv)), np.ones(len(yv)), width=0.86, color=[BLUE if v else RED for v in yv], linewidth=0)
        ax.set_xlim(-0.5, 229.5); ax.set_ylim(0, 1); ax.set_yticks([]); ax.grid(False)
        for sp in ("left", "bottom"): ax.spines[sp].set_visible(False)
        ax.set_xticks([]); ax.set_title(f"acquisition date {d}: {len(yv)} frames in acquisition order", fontsize=8.5, fontweight="normal", color=INK2, pad=3)
    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=RED)]
    fig.legend(handles, ["expert label: good", "expert label: bad"], loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(f"Labels come in runs: neighbouring frames agree {ls['adjacent_frames_share_label']:.0%} of the time (independent frames would agree {ls['expected_if_independent']:.0%})",
                 x=0.125, ha="left", fontsize=10, fontweight="bold", y=1.02)
    save(fig, "fig_label_runs")


def fig_acquisition():
    ds, gate = load("degradation_study"), load("acquisition_gate")
    titles = {"defocus": "Defocus", "motion_streak": "Motion streak", "occlusion": "Occluded field", "under_exposure": "Under-exposure"}
    fig, axes = plt.subplots(1, 6, figsize=(12.4, 2.6), gridspec_kw={"wspace": 0.42, "width_ratios": [1, 1, 1, 1, 0.75, 1.3]})
    axes[4].axis("off")
    for ax, (fault, spec) in zip(axes[:4], ds["faults"].items()):
        lv = [r["level"] for r in spec["levels"]]
        ax.plot(range(len(lv)), [r["caught_by_its_rule"] for r in spec["levels"]], color=BLUE, marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.axhline(spec["rule_fires_on_originals"], color=MUTED, linewidth=1.0)
        ax.text(len(lv) - 1, spec["rule_fires_on_originals"] + 0.03, "untouched frames", ha="right", fontsize=7.5, color=INK2)
        ax.set_xticks(range(len(lv)), [str(v) for v in lv]); ax.set_ylim(0, 1.05); ax.set_xlabel(spec["unit"], fontsize=8)
        ax.set_title(titles[fault], fontsize=9)
    axes[0].set_ylabel("share of frames the rule flags")
    rules = [("occluded", "Occluded > 25 %"), ("motion_streak", "Motion streak"), ("under_exposed", "Under-exposed"), ("defocus", "Defocus"), ("no_rule", "No rule fires")]
    hbars(axes[5], [f"{t}\n({gate['rules'][k]['frames']:,} frames)" for k, t in rules], [gate["rules"][k]["expert_bad_rate"] for k, _ in rules], fmt="{:.0%}", xlim=(0, 1.15))
    axes[5].set_xlabel("share experts called bad", fontsize=8); axes[5].set_title("On the real frames", fontsize=9)
    fig.suptitle("The acquisition gate: controlled faults on real frames (left) and what experts said about flagged real frames (right)", x=0.09, ha="left", fontsize=10, fontweight="bold", y=1.06)
    save(fig, "fig_acquisition")


def fig_camera():
    ct = load("camera_transfer")
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.0), gridspec_kw={"wspace": 0.12}, sharey=True)
    for ax, key, title in zip(axes, ("colour", "grey"), ("New camera: colour", "New camera: grey")):
        rows = ct["adaptation"][key]["rows"]
        x = [r["labelled_dates_from_new_camera"] for r in rows]
        ax.fill_between(range(len(x)), [r["auroc_p10"] for r in rows], [r["auroc_p90"] for r in rows], color=BLUE, alpha=0.10, linewidth=0)
        ax.plot(range(len(x)), [r["auroc_mean"] for r in rows], color=BLUE, marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5)
        within = ct["within_camera_dates_held_out"][key]["auroc"]
        ax.axhline(within, color=MUTED, linewidth=1.0)
        ax.text(len(x) - 1, within + 0.008, f"trained with this camera's other dates: {within:.2f}", ha="right", fontsize=8, color=INK2)
        ax.text(0.08, rows[0]["auroc_mean"] - 0.022, f"{rows[0]['auroc_mean']:.2f}", fontsize=8.5); ax.text(len(x) - 1.02, rows[-1]["auroc_mean"] - 0.026, f"{rows[-1]['auroc_mean']:.2f}", fontsize=8.5, ha="right")
        ax.set_xticks(range(len(x)), [f"{k}\n{int(round(r['mean_labelled_frames'])):,} frames" for k, r in zip(x, rows)], fontsize=8)
        ax.set_xlabel("labelled acquisition dates of the new camera added to training"); ax.set_title(title); ax.set_ylim(0.5, 0.9)
    axes[0].set_ylabel("AUROC on the new camera's remaining dates")
    save(fig, "fig_camera")


def fig_system():
    fig, ax = plt.subplots(figsize=(10.6, 2.9))
    ax.set_xlim(0, 106); ax.set_ylim(0, 29); ax.axis("off")
    def box(x, y, w, h, title, body, edge=AXIS, fill="#ffffff"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.1", linewidth=1.1, edgecolor=edge, facecolor=fill))
        ax.text(x + 1.2, y + h - 2.1, title, fontsize=9, fontweight="bold", color=INK, va="top")
        ax.text(x + 1.2, y + h - 5.6, body, fontsize=7.6, color=INK2, va="top", linespacing=1.35)
    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11, linewidth=1.1, color=INK2))
    box(0, 9, 15, 13, "Frame", "brightfield field of view\n2056 × 1542 px\n(+ optional cell line, day)")
    box(20, 9, 22, 13, "1  Acquisition gate", "five physical descriptors:\nblacked-out area, motion streak,\ndefocus, exposure")
    box(47, 9, 26, 13, "2  Culture-quality score", "whole frame at 1344 px, 96 × 72 patch\ndescriptors (DINOv2), linear head: calibrated\nP(good) and an exact evidence map")
    box(78, 9, 26, 13, "3  Risk-controlled decision", "pass threshold fitted on training dates\nfor a chosen error target;\neverything else goes to a person")
    arrow(15.6, 15.5, 19.6, 15.5); arrow(42.6, 15.5, 46.6, 15.5); arrow(73.6, 15.5, 77.6, 15.5)
    for x, text in ((31, "REACQUIRE\nnot a usable observation"), (91, "PASS   or   REVIEW (queue ordered by score)")):
        ax.text(x, 3.2, text, ha="center", va="center", fontsize=8.4, fontweight="bold", color=INK)
    arrow(31, 8.6, 31, 5.6); arrow(91, 8.6, 91, 4.6)
    ax.text(53, 26.5, "Every output is written to an audit record: decision, probability, thresholds, descriptor values, model version, image hash", ha="center", fontsize=8.2, color=INK2)
    save(fig, "fig_system")


def image_panels(man, images):
    from chipqc.frames import open_frame
    from chipqc.guardian import Guardian
    from chipqc.render import labelled, side_by_side
    from PIL import Image
    # examples: one good and one bad frame per cell line (first acquisition date that has both)
    tw, th = 420, 315
    lines = sorted(man.cell_line.unique(), key=lambda k: -(man.cell_line == k).sum())
    sheet = Image.new("RGB", (len(lines) * (tw + 6) - 6, 2 * (th + 6) - 6), "white")
    for c, line in enumerate(lines):
        sub = man[man.cell_line == line]
        for r, lab in enumerate((1, 0)):
            row = sub[sub.label_good == lab].sample(1, random_state=7).iloc[0]
            im = open_frame(images / row.path).resize((tw, th), Image.LANCZOS)
            sheet.paste(labelled(im, f"{line} · {'good' if lab else 'bad'} · {row.image_id}", (30, 90, 170) if lab else (190, 50, 60)), (c * (tw + 6), r * (th + 6)))
    OUT.mkdir(parents=True, exist_ok=True)
    sheet.save(OUT / "fig_dataset.jpg", quality=90); print("wrote reports/figures/fig_dataset.jpg")
    # evidence maps and disagreements: frames from the two dates the demo model never saw
    import pandas as pd
    from chipqc.descriptors import acquisition_flags
    g = Guardian(ROOT / "models/guardian-v2-demo")
    held = man[man.date.isin(g.spec["held_out_dates"])].copy()
    X = np.load(ROOT / f"features/{g.spec['backbone']}.npz")["embeddings"]
    A, B = g.platt
    held["p"] = 1 / (1 + np.exp(-(A * (X[held.index] @ g.v + g.b) + B)))
    d = pd.read_csv(ROOT / "features/acquisition_descriptors.csv").drop(columns="image_id")
    held["clean"] = [not acquisition_flags(r, g.limits) for r in d.iloc[held.index].to_dict("records")]

    def panel(ids, name, width=640):
        rows = []
        for iid in ids:
            row = man[man.image_id == iid].iloc[0]
            im = open_frame(images / row.path)
            a = g.assess(im, with_neighbours=False)
            rows.append(side_by_side(im, a.evidence, a.decision, a.p_good, f"{iid} · {row.cell_line} · experts: {'good' if row.label_good else 'bad'}", width=width))
        sheet = Image.new("RGB", (rows[0].width * 2 + 10, (rows[0].height + 10) * ((len(rows) + 1) // 2) - 10), "white")
        for k, r in enumerate(rows):
            sheet.paste(r, ((k % 2) * (r.width + 10), (k // 2) * (r.height + 10)))
        sheet.save(OUT / f"{name}.jpg", quality=90); print(f"wrote reports/figures/{name}.jpg", ids)

    clean = held[held.clean]
    panel([clean[clean.label_good == 1].sort_values("p").image_id.iloc[-1], clean[clean.label_good == 0].sort_values("p").image_id.iloc[0], "230524_30", "230524_35"], "fig_evidence")
    panel(list(clean[clean.label_good == 0].sort_values("p").image_id.iloc[-2:]) + list(clean[clean.label_good == 1].sort_values("p").image_id.iloc[:2]), "fig_disagreements")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", type=Path)
    ap.add_argument("--only")
    a = ap.parse_args()
    man = load_manifest(ROOT / "data/manifest.csv")
    charts = {"system": fig_system, "protocols": fig_protocols, "representations": fig_representations, "roc": lambda: fig_roc_calibration(man),
              "risk": lambda: fig_risk_coverage(man), "cell_lines": fig_cell_lines, "camera": fig_camera, "runs": lambda: fig_label_runs(man), "acquisition": fig_acquisition}
    for name, fn in charts.items():
        if not a.only or a.only == name:
            fn()
    if a.images and (not a.only or a.only == "images"):
        image_panels(man, a.images)
