"""Build a captioned AI4S demo storyboard from observed UI screenshots.

All microscopy-like pixels on the UI screenshots are synthetic 8x8 test assets.
The background cover is conceptual artwork, not experimental data.
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUT = ROOT.parents[1] / "outputs"
SIZE = (1280, 720)
FONT = "/System/Library/Fonts/Helvetica.ttc"
WHITE = (238, 248, 253)
MUTED = (160, 189, 207)
TEAL = (56, 216, 213)
BG = (7, 24, 40)


def font(size):
    return ImageFont.truetype(FONT, size)


def base(section):
    im = Image.new("RGB", SIZE, BG)
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, 1280, 7), fill=TEAL)
    d.text((72, 35), "CHIPQC GUARDIAN  /  AI4S OPEN INNOVATION", font=font(22), fill=TEAL)
    d.text((72, 88), section, font=font(44), fill=WHITE)
    return im, d


def line(d, text, y, size=27, color=MUTED):
    d.text((72, y), text, font=font(size), fill=color)


def screenshot(im, filename, box=(72, 180, 1136, 420)):
    shot = Image.open(HERE / filename).convert("RGB")
    shot.thumbnail((box[2], box[3]), Image.Resampling.LANCZOS)
    x = box[0] + (box[2] - shot.width) // 2
    y = box[1] + (box[3] - shot.height) // 2
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((x - 7, y - 7, x + shot.width + 7, y + shot.height + 7), 12, fill=(36, 74, 93))
    im.paste(shot, (x, y))


slides = []
cover = Image.open(OUT / "ChipQC_Guardian_Cover.png").convert("RGB")
cover = cover.resize(SIZE, Image.Resampling.LANCZOS)
d = ImageDraw.Draw(cover)
d.rounded_rectangle((55, 434, 812, 670), 18, fill=(6, 22, 37))
d.text((88, 465), "ChipQC Guardian", font=font(60), fill=WHITE)
d.text((90, 548), "Human-reviewed organ-on-chip image quality", font=font(29), fill=TEAL)
d.text((90, 602), "AI4S Open Innovation  |  End-to-End System", font=font(21), fill=MUTED)
slides.append((cover, 10))

im, d = base("The problem")
line(d, "Brightfield image review is repetitive, but quality errors matter.", 184, 32, WHITE)
line(d, "A score alone is not an operational decision.", 245, 31)
d.rounded_rectangle((72, 344, 1208, 528), 20, fill=(14, 48, 69))
line(d, "Original image   +   transparent quality descriptors   +   human review", 390, 28, TEAL)
line(d, "This prototype does not diagnose disease or assess drug efficacy.", 575, 24)
slides.append((im, 10))

im, d = base("Working research prototype")
screenshot(im, "ui-before.jpg", (72, 180, 1136, 390))
line(d, "Streamlit app with a loaded research model and explicit use limits.", 605, 25, WHITE)
line(d, "The input assets shown next are synthetic, not biological specimens.", 646, 23)
slides.append((im, 12))

im, d = base("Upload and inspect — synthetic checker")
screenshot(im, "ui-result-card.jpg", (72, 178, 1136, 390))
line(d, "Rule preview: PASS. Research model output is displayed separately.", 608, 24, WHITE)
line(d, "PASS here is not a validated decision for real samples.", 647, 23)
slides.append((im, 13))

im, d = base("Defect review — synthetic flat image")
screenshot(im, "ui-flat-card.jpg", (72, 178, 1136, 390))
line(d, "Low clarity and contrast trigger a REACQUIRE preview.", 608, 24, WHITE)
line(d, "Model disagreement is visible; a researcher makes the final call.", 647, 23)
slides.append((im, 13))

im, d = base("Frozen, acquisition-grouped evaluation")
line(d, "3,072 images fully matched; 59 date-like acquisition groups.", 185, 29, WHITE)
line(d, "12 held-out groups / 670 images; no test tuning.", 232, 28)
for idx, (label, value) in enumerate((
    ("Balanced accuracy", "0.712"), ("AUROC", "0.772"),
    ("Macro-F1", "0.712"), ("Brier score", "0.204"),
)):
    x = 72 + (idx % 2) * 570
    y = 324 + (idx // 2) * 135
    d.rounded_rectangle((x, y, x + 510, y + 105), 16, fill=(14, 48, 69))
    d.text((x + 25, y + 18), label, font=font(23), fill=MUTED)
    d.text((x + 338, y + 34), value, font=font(38), fill=TEAL)
line(d, "Internal evidence only; wide group-bootstrap intervals.", 617, 23, WHITE)
line(d, "No calibrated automatic-PASS threshold met the planned safety target.", 653, 21)
slides.append((im, 14))

im, d = base("Reproduce, inspect, and improve")
line(d, "Public code + report + split manifest + evaluation outputs", 185, 29, WHITE)
line(d, "github.com/yy5652-hash/chipqc-guardian-ai4s", 285, 32, TEAL)
line(d, "Third-party raw images are not republished.", 360, 26)
line(d, "Next: prospective review and external-lab evaluation.", 480, 27, WHITE)
line(d, "Current claims remain limited to the frozen internal test.", 547, 24)
slides.append((im, 10))

frames = []
durations = []
for i, (im, seconds) in enumerate(slides, 1):
    im.save(HERE / f"demo-slide-{i:02d}.jpg", quality=92)
    frames.append(im.quantize(colors=128, method=Image.Quantize.FASTOCTREE))
    durations.append(seconds * 1000)

gif = OUT / "ChipQC_Guardian_Demo_Silent.gif"
frames[0].save(gif, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=False)
print(f"Saved {gif} ({sum(durations)/1000:.0f}s, {len(frames)} scenes)")
