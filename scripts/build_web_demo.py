"""Assemble the static browser demo in docs/ (served by GitHub Pages).

    python scripts/build_web_demo.py

Writes docs/model/guardian_tiles.onnx (the demo model: backbone, classifier and calibrator in one graph),
docs/model/web_model.json (thresholds, acquisition limits, and the Python reference result for every bundled
frame) and copies the bundled frames. The page itself (docs/index.html, app.js, core.js) is hand-written.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from chipqc.frames import open_frame  # noqa: E402
from chipqc.guardian import Guardian  # noqa: E402

DOCS, MODEL = ROOT / "docs", ROOT / "models/guardian-v2-demo"
(DOCS / "model").mkdir(parents=True, exist_ok=True)
(DOCS / "examples").mkdir(exist_ok=True)
subprocess.run([sys.executable, str(ROOT / "scripts/export_onnx.py"), "--model", str(MODEL), "--out", str(DOCS / "model")], check=True)

g = Guardian(MODEL)
meta = json.loads((ROOT / "examples/examples.json").read_text())
frames = []
for item in meta["frames"]:
    src = ROOT / "examples" / item["file"]
    shutil.copy(src, DOCS / "examples" / item["file"])
    image = open_frame(src)
    image.resize((336, 252), Image.LANCZOS).save(DOCS / "examples" / (src.stem + "_thumb.jpg"), quality=80)
    a = g.assess(image, with_neighbours=False)
    frames.append({**item, "thumb": src.stem + "_thumb.jpg", "reference": {"p_good": round(a.p_good, 6), "decision": a.decision, "reasons": a.reasons,
                                                                          "acquisition": {k: round(v, 6) for k, v in a.acquisition.items()}}})
ev = g.spec.get("evaluation") or {}
web = {"name": g.spec["name"], "created_utc": g.spec["created_utc"], "held_out_dates": g.spec["held_out_dates"], "backbone": "DINOv2 ViT-S/14 with registers (frozen)",
       "frame_size": g.spec["frame_size"], "tile": g.spec["tile"], "grid": g.spec["grid"], "thresholds": g.spec["thresholds"], "default_error_target": str(g.spec["default_error_target"]),
       "acquisition_limits": g.limits, "note": meta["note"], "frames": frames,
       "evaluation": {"auroc": ev.get("auroc"), "frames": ev.get("frames"), "dates": ev.get("dates"), "system": ev.get("system")}}
(DOCS / "model/web_model.json").write_text(json.dumps(web, indent=1))
(DOCS / ".nojekyll").write_text("")
print("docs/model/web_model.json:", len(frames), "frames;", {f["image_id"]: (f["reference"]["decision"], f["reference"]["p_good"]) for f in frames})
