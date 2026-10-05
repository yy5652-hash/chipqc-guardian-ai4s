"""ChipQC Guardian review console.

    streamlit run app.py

Upload brightfield frames from an organ-on-a-chip run (or open the bundled examples). Each frame gets one of
three outcomes: PASS, REVIEW or REACQUIRE, with the evidence behind it. Nothing leaves your machine.
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from chipqc.guardian import PASS, REACQUIRE, REVIEW, Guardian  # noqa: E402
from chipqc.render import evidence_overlay  # noqa: E402

MODELS = {"released": ROOT / "models/guardian-v2", "examples": ROOT / "models/guardian-v2-demo"}
BADGE = {PASS: ("#0c8228", "PASS", "may be used without a person looking"), REVIEW: ("#a86c00", "REVIEW", "a person decides"),
         REACQUIRE: ("#aa4628", "REACQUIRE", "not a usable observation")}

st.set_page_config(page_title="ChipQC Guardian", layout="wide")


@st.cache_resource(show_spinner="Loading the model…")
def guardian(which: str) -> Guardian:
    path = MODELS[which] if MODELS[which].exists() else MODELS["released"]
    g = Guardian(path)
    _ = g.encoder          # load the backbone once (assigned, so Streamlit does not print it)
    return g


def badge(decision: str) -> str:
    colour, name, meaning = BADGE[decision]
    return f"<span style='background:{colour};color:#fff;padding:3px 10px;border-radius:4px;font-weight:600'>{name}</span> <span style='color:#52514e'>{meaning}</span>"


def thumbnail(g: Guardian, image_id: str):
    p = g.atlas_dir / f"{image_id}.jpg"
    return Image.open(p) if p.exists() else None


g_released = guardian("released")
spec = g_released.spec
ev = spec.get("evaluation") or {}

with st.sidebar:
    st.markdown("### ChipQC Guardian")
    st.caption("Quality gate for organ-on-a-chip brightfield frames")
    target = st.select_slider("Target: at most this share of passed frames may be bad", options=["0.1", "0.15", "0.2"], value=str(spec["default_error_target"]),
                              format_func=lambda v: f"{float(v):.0%}")
    t_pass = g_released.pass_threshold(target)
    st.caption(f"Pass threshold at this target: P(good) ≥ {t_pass:.2f}")
    if ev:
        s = ev.get("system", {}).get(target, {}).get("summary")
        st.markdown("**Measured on dates the model never saw**")
        st.caption(f"AUROC {ev['auroc']['value']:.3f} (95 % interval {ev['auroc']['ci95'][0]:.3f}–{ev['auroc']['ci95'][1]:.3f}) over {ev['frames']:,} frames from {ev['dates']} acquisition dates.")
        if s:
            st.caption(f"At this target {s['passed_automatically']:.0%} of frames passed automatically and {s['bad_among_passed']:.1%} of those were bad; "
                       f"{s['sent_to_reacquire']:.0%} were sent back for re-acquisition; {s['left_for_a_person']:.0%} went to a person.")
    st.markdown("**Scope**")
    st.caption("The score estimates what cell-biology experts called a good or bad culture in the reference dataset (six cell lines, one laboratory). "
               "It is not a measurement of viability, barrier function or drug response.")

st.title("ChipQC Guardian")
st.write("Each frame passes an **acquisition gate** (is this a usable observation?), then a **culture-quality score** with an exact evidence map, "
         "then a **risk-controlled decision**. A person keeps every decision the evidence does not support.")

examples_dir = ROOT / "examples"
example_meta = json.loads((examples_dir / "examples.json").read_text()) if (examples_dir / "examples.json").exists() else {}
tab_upload, tab_examples = st.tabs(["Your frames", "Bundled examples"])
frames, which_model = [], "released"
with tab_upload:
    files = st.file_uploader("Brightfield frames (PNG, JPEG or TIFF)", type=["png", "jpg", "jpeg", "tif", "tiff"], accept_multiple_files=True)
    for f in files or []:
        data = f.getvalue()
        frames.append((f.name, ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB"), hashlib.sha256(data).hexdigest(), None))
with tab_examples:
    if example_meta:
        st.caption(example_meta["note"])
        if st.toggle("Open the bundled example run", value=not frames):
            which_model = "examples"
            for item in example_meta["frames"]:
                data = (examples_dir / item["file"]).read_bytes()
                frames.append((item["file"], Image.open(io.BytesIO(data)).convert("RGB"), hashlib.sha256(data).hexdigest(), item))
    else:
        st.caption("No bundled examples in this checkout.")

if not frames:
    st.info("Add frames above, or open the bundled examples.")
    st.stop()

g = guardian(which_model)
records, results = [], []
progress = st.progress(0.0, text="Scoring frames…")
for k, (name, image, digest, meta) in enumerate(frames):
    a = g.assess(image, error_target=target)
    results.append((name, image, a, meta))
    records.append({"frame": name, "decision": a.decision, "p_good": round(a.p_good, 4), "reason": a.reasons[0], **{k2: round(v, 4) for k2, v in a.acquisition.items()},
                    "pass_threshold": g.pass_threshold(target), "error_target": float(target), "model": g.spec["name"], "model_created_utc": g.spec["created_utc"],
                    "sha256": digest, "assessed_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})
    progress.progress((k + 1) / len(frames), text=f"Scored {k + 1} of {len(frames)}")
progress.empty()

table = pd.DataFrame(records)
if which_model == "examples":
    st.caption("Showing the bundled example run. These frames come from two acquisition dates that the model scoring them never saw.")
counts = table.decision.value_counts()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Frames", len(table))
c2.metric("PASS", int(counts.get(PASS, 0)))
c3.metric("REVIEW", int(counts.get(REVIEW, 0)))
c4.metric("REACQUIRE", int(counts.get(REACQUIRE, 0)))
order = {REACQUIRE: 0, REVIEW: 1, PASS: 2}
st.dataframe(table.assign(_o=table.decision.map(order)).sort_values(["_o", "p_good"]).drop(columns="_o")[["frame", "decision", "p_good", "reason"]], hide_index=True, width="stretch")
d1, d2 = st.columns(2)
d1.download_button("Audit record (CSV)", table.to_csv(index=False).encode(), "chipqc_audit.csv", "text/csv")
d2.download_button("Audit record (JSON)", json.dumps({"model": g.spec["name"], "thresholds": g.spec["thresholds"][target], "acquisition_limits": g.spec["acquisition_limits"], "frames": records}, indent=1).encode(),
                   "chipqc_audit.json", "application/json")

st.subheader("Frame by frame (most urgent first)")
for name, image, a, meta in sorted(results, key=lambda r: (order[r[2].decision], r[2].p_good)):
    with st.container(border=True):
        head = f"**{name}** &nbsp; {badge(a.decision)} &nbsp; P(good) **{a.p_good:.2f}**"
        if meta:
            head += f" &nbsp; <span style='color:#52514e'>experts: {meta['expert_label']} · {meta['cell_line']} · held out from this model</span>"
        st.markdown(head, unsafe_allow_html=True)
        st.caption(" · ".join(a.reasons))
        left, right = st.columns(2)
        left.image(image, caption="Frame", width="stretch")
        right.image(evidence_overlay(image, a.evidence), caption="Evidence map: blue regions pull the score towards good, red towards bad. The map averages exactly to the score.", width="stretch")
        with st.expander("Acquisition descriptors and similar reference frames"):
            lim = g.limits
            rows = [("Blacked-out share of the field", a.acquisition["occluded_block_frac"], f"re-acquire above {lim['occluded_block_frac_max']:.2f}"),
                    ("Motion streak (anisotropy)", a.acquisition["streak_anisotropy"], f"re-acquire above {lim['streak_anisotropy_max']:.2f}"),
                    ("Local sharpness (log10)", a.acquisition["log_sharpness"], f"re-acquire below {lim['log_sharpness_min']:.2f}"),
                    ("Median brightness", a.acquisition["median_brightness"], f"re-acquire below {lim['median_brightness_min']:.2f}"),
                    ("Clipped highlights", a.acquisition["saturated_frac"], "reported only")]
            st.dataframe(pd.DataFrame(rows, columns=["Descriptor", "Value", "Limit"]).round(3), hide_index=True, width="stretch")
            cols = st.columns(6)
            for col, nb in zip(cols, a.neighbours):
                th = thumbnail(g, nb["image_id"])
                if th is not None:
                    col.image(th, width="stretch")
                col.caption(f"{nb['label']} · {nb['cell_line']} · similarity {nb['similarity']:.2f}")
st.caption("Reference data: Movčana et al., Organ-on-a-Chip (OOC) Image Dataset, Zenodo, doi:10.5281/zenodo.10203721, CC BY 4.0.")
