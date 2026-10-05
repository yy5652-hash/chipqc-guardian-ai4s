"""Run the browser demo in headless Chromium on the bundled frames and compare it with the Python reference.

    python scripts/check_web_demo.py [--backend wasm|webgpu]    # -> docs/model/browser_parity.json, reports/figures/web_demo.png

Serves docs/ on localhost, opens index.html?selftest=1 and waits for the page to score all bundled frames.
Needs `pip install playwright && playwright install chromium`.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", default="wasm", choices=["wasm", "webgpu"])
    ap.add_argument("--port", type=int, default=8799)
    a = ap.parse_args()
    from playwright.sync_api import sync_playwright

    server = subprocess.Popen([sys.executable, "-m", "http.server", str(a.port), "--directory", str(ROOT / "docs"), "--bind", "127.0.0.1"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(1.0)
        with sync_playwright() as p:
            browser = p.chromium.launch(args=["--enable-unsafe-webgpu", "--enable-features=Vulkan,WebGPU", "--use-angle=metal"] if a.backend == "webgpu" else [])
            page = browser.new_page(viewport={"width": 1280, "height": 1500}, device_scale_factor=2)
            errors = []
            page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(f"http://127.0.0.1:{a.port}/index.html?selftest=1&backend={a.backend}")
            try:
                page.wait_for_function("window.__selftest !== undefined", timeout=1_800_000)
            except Exception:
                print("page status:", page.inner_text("#status"), "| progress:", page.inner_text("#progress"), "| errors:", errors[:5])
                raise
            result = page.evaluate("window.__selftest")
            page.screenshot(path=str(ROOT / "reports/figures/web_demo.png"), full_page=True)
            browser.close()
    finally:
        server.terminate()
    rows, worst_p, worst_d = [], 0.0, 0.0
    for f in result["frames"]:
        ref = f["reference"]
        dp = abs(f["p_good"] - ref["p_good"])
        dd = max(abs(f["acquisition"][k] - v) for k, v in ref["acquisition"].items())
        same_gate = bool(f["flags"]) == (ref["decision"] == "REACQUIRE")
        worst_p, worst_d = max(worst_p, dp), max(worst_d, dd)
        rows.append({"image_id": f["image_id"], "p_good_browser": round(f["p_good"], 6), "p_good_python": ref["p_good"], "difference": round(dp, 6),
                     "largest_descriptor_difference": float(f"{dd:.2e}"), "same_gate_outcome": same_gate, "seconds": round(f["seconds"], 1)})
    out = {"backend": result["backend"], "browser": "Chromium (headless)", "frames": rows, "largest_p_good_difference": round(worst_p, 6),
           "largest_descriptor_difference": float(f"{worst_d:.2e}"), "all_gate_outcomes_match": all(r["same_gate_outcome"] for r in rows),
           "mean_seconds_per_frame": round(sum(r["seconds"] for r in rows) / len(rows), 1)}
    (ROOT / "docs/model/browser_parity.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "frames"}))


if __name__ == "__main__":
    main()
