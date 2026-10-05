"""Start the review console, open it in a headless browser and save screenshots or a screen recording.

    python scripts/capture_console.py --out reports/figures        # PNG screenshots for the report
    python scripts/capture_console.py --video demo/recording       # a WebM recording of a scripted walkthrough

Needs `pip install playwright && playwright install chromium`. The console runs on this machine only.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "reports/figures")
    ap.add_argument("--video", type=Path)
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    from playwright.sync_api import sync_playwright

    env = dict(os.environ, STREAMLIT_BROWSER_GATHER_USAGE_STATS="false")
    server = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"), "--server.port", str(a.port), "--server.headless", "true", "--server.address", "127.0.0.1"],
                              cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        url = f"http://127.0.0.1:{a.port}"
        for _ in range(120):
            try:
                urllib.request.urlopen(url + "/_stcore/health", timeout=1)
                break
            except Exception:
                time.sleep(0.5)
        with sync_playwright() as p:
            browser = p.chromium.launch()
            if a.video:
                record(browser, url, a.video)
            else:
                screenshots(browser, url, a.out)
            browser.close()
    finally:
        server.terminate()


def card(page, text):
    """The innermost bordered block that mentions `text`."""
    return page.locator('div[data-testid="stVerticalBlock"]').filter(has=page.get_by_text(text)).last


def screenshots(browser, url, out):
    out.mkdir(parents=True, exist_ok=True)
    page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=2)
    page.goto(url)
    page.get_by_text("Frame by frame (most urgent first)").wait_for(timeout=240_000)
    page.wait_for_timeout(2500)
    page.screenshot(path=str(out / "console_overview.png"))
    for name, text in (("console_frame_pass", "may be used without a person looking"), ("console_frame_review", "likely bad: review first"), ("console_frame_reacquire", "not a usable observation")):
        block = card(page, text)
        block.scroll_into_view_if_needed(); page.wait_for_timeout(1200)
        block.screenshot(path=str(out / f"{name}.png"))
    print("saved", sorted(q.name for q in out.glob("console_*.png")))


def record(browser, url, out):
    """A scripted walkthrough at 1920 x 1080. Writes the WebM and a JSON list of (label, seconds since start).
    The page jumps between positions instead of scrolling, so the picture never moves while it is on screen."""
    out.mkdir(parents=True, exist_ok=True)
    warm = browser.new_page()                                   # load the model before the recording starts
    warm.goto(url); warm.get_by_text("Frame by frame (most urgent first)").wait_for(timeout=240_000); warm.close()
    ctx = browser.new_context(viewport={"width": 1920, "height": 1080}, record_video_dir=str(out), record_video_size={"width": 1920, "height": 1080})
    page = ctx.new_page()
    t0, marks = time.time(), []
    mark = lambda label: marks.append((label, round(time.time() - t0, 2)))
    page.goto(url)
    page.get_by_text("Frame by frame (most urgent first)").wait_for(timeout=240_000)
    page.evaluate("document.body.style.zoom = 1.3")
    page.wait_for_timeout(1500); mark("overview")
    page.wait_for_timeout(10500)

    def show(label, text, hold):
        block = card(page, text)
        box = block.bounding_box()
        block.evaluate("e => { e.style.scrollMarginTop = '84px'; e.scrollIntoView({behavior: 'instant', block: 'start'}); }")   # room for a caption above the card
        page.wait_for_timeout(1400); mark(label)
        page.wait_for_timeout(int(hold * 1000))
        return block

    show("streak", "motion streak suspected", 6.0)
    show("occluded", "of the field is blacked out", 6.5)
    show("review", "230425_180.png", 15.5)
    block = show("pass", "230425_21.png", 6.5)
    block.get_by_text("Acquisition descriptors and similar reference frames").click()
    page.wait_for_timeout(900)
    block.get_by_text("Acquisition descriptors and similar reference frames").evaluate("e => { e.style.scrollMarginTop = '84px'; e.scrollIntoView({behavior: 'instant', block: 'start'}); }")
    page.wait_for_timeout(1200); mark("details")
    page.wait_for_timeout(8500)
    page.get_by_text("Audit record (CSV)").evaluate("e => e.scrollIntoView({behavior: 'instant', block: 'center'})")
    page.wait_for_timeout(1300); mark("audit")
    page.wait_for_timeout(5000)
    video = page.video
    ctx.close()
    path = Path(video.path())
    path.rename(out / "console.webm")
    (out / "console_marks.json").write_text(json.dumps(marks, indent=1))
    print("recorded", out / "console.webm", marks)


if __name__ == "__main__":
    main()
