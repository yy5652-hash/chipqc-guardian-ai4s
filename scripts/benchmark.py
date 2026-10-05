"""Seconds per frame for the released model on this machine -> results/runtime.json"""
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from chipqc.frames import open_frame  # noqa: E402
from chipqc.guardian import Guardian  # noqa: E402

frames = [open_frame(p) for p in sorted((ROOT / "examples").glob("*.png"))]
out = {"machine": platform.platform(), "processor": platform.processor(), "frames": len(frames)}
for device in ("mps", "cuda", "cpu"):
    try:
        g = Guardian(ROOT / "models/guardian-v2", device=device)
        g.assess(frames[0])                                # warm-up
        t0 = time.time()
        for f in frames:
            g.assess(f)
        out[f"{'gpu' if device != 'cpu' else 'cpu'}_s"] = f"{(time.time() - t0) / len(frames):.2f}"
        out[f"{'gpu' if device != 'cpu' else 'cpu'}_device"] = device
    except Exception as e:                                  # device not present on this machine
        out[f"{device}_unavailable"] = str(e)[:80]
(ROOT / "results/runtime.json").write_text(json.dumps(out, indent=1))
print(out)
