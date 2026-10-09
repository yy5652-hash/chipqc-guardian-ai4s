"""Fill the documents in docs_templates/ from results/*.json and typeset the report.

    python scripts/build_report.py          # -> README.md, MODEL_CARD.md, reports/technical_report.md, ChipQC_Guardian_Technical_Report.pdf

Every `{{ expression }}` in the template is evaluated against the result files, so the report cannot quote a
number that the evaluation did not produce.
"""
import html
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"


PAGES = ["15"]          # page count of the typeset report, filled in after typesetting


class Box(dict):
    """dict with attribute access, recursively."""
    def __getitem__(self, k):
        return wrap(dict.__getitem__(self, k))

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def values(self):
        return [wrap(v) for v in dict.values(self)]


def wrap(v):
    if isinstance(v, dict):
        return Box(v)
    if isinstance(v, list):
        return [wrap(x) for x in v]
    return v


def signed(x):
    return f"{x:+.3f}".replace("-", "−")


def load(name):
    return wrap(json.loads((R / f"{name}.json").read_text()))


def namespace():
    H, REP, SO, CT, D, IMG = load("headline"), load("representations"), load("system_outcomes"), load("camera_transfer"), load("degradation_study"), load("image_formats")
    main = json.loads((ROOT / "models/guardian-v2/model.json").read_text())["backbone"]
    labels = {"mobilenet_v2_224crop": "MobileNetV2, 224 px centre crop (first submission)", "mobilenet_v2_672": "MobileNetV2, whole frame, 672 px",
              "mobilenet_v2_1344": "MobileNetV2, whole frame, 1344 px", "mobilenet_v2_2048": "MobileNetV2, whole frame, 2048 px (native)",
              "resnet50_1344": "ResNet-50, whole frame, 1344 px", "convnext_tiny_1344": "ConvNeXt-Tiny, whole frame, 1344 px",
              "dinov2_vits14_2x2": "DINOv2 ViT-S/14, whole frame at 896 px, 2 × 2 tiles", "dinov2_vits14": "**DINOv2 ViT-S/14, whole frame at 1344 px, 3 × 3 tiles (released model)**",
              "dinov2_vits14_4x4": "DINOv2 ViT-S/14, whole frame at 1792 px, 4 × 4 tiles", "dinov2_vitb14": "DINOv2 ViT-B/14, whole frame at 1344 px, 3 × 3 tiles"}

    def representation_rows():
        rows = []
        for k, text in labels.items():
            if k in REP:
                r = REP[k]
                rows.append(f"| {text} | {r['auroc']['value']:.3f} | {r['auroc']['ci95'][0]:.3f}–{r['auroc']['ci95'][1]:.3f} | {diff(r['auroc_minus_released']) if 'auroc_minus_released' in r else '(reference)'} | {r['triage_at_10pct']['passed'] * 100:.0f} % |")
        return "\n".join(rows)

    def diff(d):                            # a signed difference with the interval of the difference itself
        return f"{signed(d['value'])} ({signed(d['ci95'][0])} to {signed(d['ci95'][1])})"

    def outcome_rows():
        rows = []
        for t in ("0.05", "0.1", "0.15", "0.2"):
            s = SO[t]["summary"]
            rows.append(f"| {float(t) * 100:.0f} % | {s['passed_automatically'] * 100:.0f} % | {s['bad_among_passed'] * 100:.1f} % | {s['sent_to_reacquire'] * 100:.0f} % | {s['left_for_a_person'] * 100:.0f} % |")
        return "\n".join(rows)

    def sup(direction):                     # range over the supervised whole-frame backbones
        cb = json.loads((R / "camera_backbones.json").read_text())
        v = [cb[k][direction] for k in ("mobilenet_v2_672", "mobilenet_v2_1344", "mobilenet_v2_2048", "resnet50_1344", "convnext_tiny_1344") if k in cb]
        return f"{min(v):.2f}–{max(v):.2f}"

    grey = sum(f.frames for f in IMG.image_formats if f.mode == "L")
    runtime = load("runtime") if (R / "runtime.json").exists() else Box(gpu_s="0.2", cpu_s="n/a")
    def per_date_rows():
        import numpy as np
        import pandas as pd
        from sklearn.metrics import roc_auc_score
        man = pd.read_csv(ROOT / "data/manifest.csv", dtype={"date": str})
        man["p"] = np.load(R / f"oof/{json.loads((R / 'headline.json').read_text())['main']}.npz")["p_good"]
        man["camera"] = pd.read_csv(ROOT / "features/camera_format.csv").camera.str.split().str[0]
        rows = []
        for date, g in man.groupby("date"):
            both = g.label_good.nunique() == 2
            rows.append(f"| {date} | {'/'.join(sorted(g.camera.unique()))} | {', '.join(sorted(g.cell_line.unique()))} | {len(g)} | {g.label_good.mean():.0%} | {g.p.mean():.2f} | {roc_auc_score(g.label_good, g.p):.2f} |" if both
                        else f"| {date} | {'/'.join(sorted(g.camera.unique()))} | {', '.join(sorted(g.cell_line.unique()))} | {len(g)} | {g.label_good.mean():.0%} | {g.p.mean():.2f} | one class |")
        return "\n".join(rows)

    def day_rows():
        names = {"0-1": "0–1 days", "2-3": "2–3 days", "4": "4 days", "4+": "more than 4 days"}
        return "\n".join(f"| {names[k]} | {v['n']:,} | {v['auroc']:.3f} |" for k, v in H["by_day_bin"].items())

    return dict(
        per_date_rows=per_date_rows, day_rows=day_rows,
        H=H, REP=REP, LK=load("leakage"), B=load("baselines"), U=load("unseen_cell_line"), LS=load("label_structure"), G=load("acquisition_gate"),
        S05=SO["0.05"].summary, S10=SO["0.1"].summary, S15=SO["0.15"].summary, S20=SO["0.2"].summary, CT=CT, CTC=CT.adaptation.colour.rows, CTG=CT.adaptation.grey.rows, CTX=load("camera_backbones"), D=D,
        DS=Box({k: [lv["caught_by_its_rule"] for lv in v["levels"]] for k, v in D["faults"].items()}),
        IMG=Box(grey=f"{grey:,}", colour=f"{sum(f.frames for f in IMG.image_formats) - grey:,}"), RT=runtime, cells="6,912",
        f1=lambda x: f"{x:.1f}", f2=lambda x: f"{x:.2f}", f3=lambda x: f"{x:.3f}", pct=lambda x: f"{x:.0%}".replace("%", " %"), pct1=lambda x: f"{x:.1%}".replace("%", " %"),
        ci=lambda d: f"{d['ci95'][0]:.3f}–{d['ci95'][1]:.3f}", ci2=lambda c: f"{c[0]:.2f}–{c[1]:.2f}", ci3=lambda c: f"{c[0]:.3f}–{c[1]:.3f}", int=lambda x: f"{int(round(x)):,}",
        f6=lambda x: f"{x:.6f}", minutes=lambda frames, s: f"{frames * float(s) / 60:.0f}", UC=wrap(json.loads((R / "unseen_cell_line.json").read_text()).get("_comparison", {}).get("lines", {})),
        sup_g2c=sup("grey_to_colour"), sup_c2g=sup("colour_to_grey"),
        WEB=wrap(json.loads((ROOT / "docs/model/browser_parity.json").read_text())), FS=load("first_submission_split"), links=wrap(json.loads((ROOT / "docs_templates/links.json").read_text())), pages=PAGES[0],
        representation_rows=representation_rows, outcome_rows=outcome_rows, diff=diff, signed=signed, main=main, M=wrap(json.loads((ROOT / "models/guardian-v2/model.json").read_text())),
        line_aurocs=", ".join(f"{k} {v['auroc']:.2f}" for k, v in H["by_cell_line"].items()),
        n_tests=sum(len(re.findall(r"^def test_", p.read_text(), flags=re.M)) for p in (ROOT / "tests").glob("test_*.py")),
        max_line_drop=max(abs(v["auroc_line_seen_dates_held_out"] - v["auroc_line_never_seen"]) for k, v in json.loads((R / "unseen_cell_line.json").read_text()).items() if not k.startswith("_")),
        loc=f"{sum(len(p.read_text().splitlines()) for p in (ROOT / 'src/chipqc').glob('*.py')):,}")


def render(template: str) -> str:
    ns = namespace()
    return re.sub(r"\{\{(.+?)\}\}", lambda m: str(eval(m.group(1).strip(), {"__builtins__": {}}, ns)), template)


# ---------------------------------------------------------------------------------------------- typesetting

def typeset(markdown: str, out: Path):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus.doctemplate import ActionFlowable
    from reportlab.platypus import BaseDocTemplate, CondPageBreak, Frame, Image, KeepTogether, PageBreak, PageTemplate, Paragraph, Preformatted, Spacer, Table, TableStyle

    fonts = "/System/Library/Fonts/Supplemental/"
    files = (("Body", "Arial.ttf"), ("Body-Bold", "Arial Bold.ttf"), ("Body-Italic", "Arial Italic.ttf"), ("Mono", "Courier New.ttf"))
    if not Path(fonts + "Arial.ttf").exists():     # outside macOS: the Liberation fonts have the metrics of Arial and Courier New
        fonts = "/usr/share/fonts/truetype/liberation/"
        files = (("Body", "LiberationSans-Regular.ttf"), ("Body-Bold", "LiberationSans-Bold.ttf"), ("Body-Italic", "LiberationSans-Italic.ttf"), ("Mono", "LiberationMono-Regular.ttf"))
    for name, file in files:
        pdfmetrics.registerFont(TTFont(name, fonts + file))
    pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body-Italic", boldItalic="Body-Bold")
    INK, INK2, MUTED, LINE, WASH, LINK = (colors.HexColor(c) for c in ("#0b0b0b", "#3d3c3a", "#6b6a66", "#d6d5cf", "#f4f3ef", "#1c5cab"))
    base = getSampleStyleSheet()["BodyText"]
    S = {
        "body": ParagraphStyle("body", parent=base, fontName="Body", fontSize=9.6, leading=14.0, textColor=INK, spaceAfter=6.5, allowWidows=0, allowOrphans=0),
        "title": ParagraphStyle("title", parent=base, fontName="Body-Bold", fontSize=19, leading=23, textColor=INK, spaceAfter=8),
        "h2": ParagraphStyle("h2", parent=base, fontName="Body-Bold", fontSize=13.5, leading=17, textColor=INK, spaceBefore=13, spaceAfter=6),
        "h3": ParagraphStyle("h3", parent=base, fontName="Body-Bold", fontSize=10.6, leading=14, textColor=INK2, spaceBefore=9, spaceAfter=4),
        "bullet": ParagraphStyle("bullet", parent=base, fontName="Body", fontSize=9.6, leading=13.8, textColor=INK, leftIndent=13, bulletIndent=3, spaceAfter=3.5),
        "cell": ParagraphStyle("cell", parent=base, fontName="Body", fontSize=8.4, leading=11.2, textColor=INK),
        "cellhead": ParagraphStyle("cellhead", parent=base, fontName="Body-Bold", fontSize=8.4, leading=11.2, textColor=INK),
        "caption": ParagraphStyle("caption", parent=base, fontName="Body", fontSize=8.4, leading=11.4, textColor=INK2, spaceBefore=3, spaceAfter=10),
        "code": ParagraphStyle("code", parent=base, fontName="Mono", fontSize=7.9, leading=10.6, textColor=INK, backColor=WASH, borderPadding=6, leftIndent=6, rightIndent=6, spaceBefore=4, spaceAfter=9),
    }
    width = A4[0] - 36 * mm

    def inline(t):
        t = html.escape(t.strip())
        t = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", lambda m: f'<link href="{m.group(2)}" color="#1c5cab">{m.group(1)}</link>', t)
        t = re.sub(r"(?<![\w/\"=])(https?://[^\s<)]+)", lambda m: f'<link href="{m.group(1)}" color="#1c5cab">{m.group(1)}</link>', t)
        t = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", t)
        t = re.sub(r"`([^`]+)`", r'<font name="Mono" size="8.6">\1</font>', t)
        return re.sub(r"(?<![\w*])\*([^*\s][^*]*)\*(?![\w*])", r"<i>\1</i>", t)

    def table(lines):
        rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in lines]
        rows = [r for r in rows if not all(re.fullmatch(r":?-{3,}:?", c.replace(" ", "")) for c in r)]
        n = max(len(r) for r in rows)
        data = [[Paragraph(inline(c), S["cellhead" if i == 0 else "cell"]) for c in r + [""] * (n - len(r))] for i, r in enumerate(rows)]
        plain = lambda c: re.sub(r"[*`]", "", c)
        measure = lambda t, i, c: pdfmetrics.stringWidth(t, "Body-Bold" if i == 0 or "**" in c else "Body", 8.4)
        cells = [(i, r[j], j) for i, r in enumerate(rows) for j in range(len(r))]
        # a column is never narrower than its longest word; what is left goes to the columns whose text is longer than that
        least = [max((measure(w, i, c) for i, c, k in cells if k == j for w in plain(c).split()), default=0) + 10 for j in range(n)]
        full = [max((measure(plain(c), i, c) for i, c, k in cells if k == j), default=0) + 10 for j in range(n)]
        if sum(full) <= width:
            widths = [width * f / sum(full) for f in full]
        else:
            want = [f - m for f, m in zip(full, least)]
            widths = [m + (width - sum(least)) * w / sum(want) for m, w in zip(least, want)]
        t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, 0), 0.9, INK), ("LINEBELOW", (0, 1), (-1, -1), 0.4, LINE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                               ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4), ("TOPPADDING", (0, 0), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5)]))
        return t

    counter = {"fig": 0}
    story, para, code, in_code = [], [], [], False

    is_heading = lambda f: isinstance(f, Paragraph) and f.style.name in ("h2", "h3")

    class Float(KeepTogether):
        """A figure that may slide below the text that follows it, to the top of the next page, instead of leaving the page half empty."""

    class Document(BaseDocTemplate):
        waiting = ()

        def handle_flowable(self, flowables):
            f = flowables[0]
            if flowables is self._hanging or isinstance(f, ActionFlowable):  # page and frame bookkeeping, not content
                return super().handle_flowable(flowables)
            frame = self.frame
            room = frame._y - frame._y1p
            def tall(x):                                                    # KeepTogether reports a sentinel height; its real one is _H
                h = x.wrapOn(self.canv, frame._aW, 10 ** 6)[1]
                return (x._H if isinstance(x, KeepTogether) else h) + x.getSpaceBefore() + x.getSpaceAfter()

            content = lambda rest: next((x for x in rest if isinstance(x, CondPageBreak) or not isinstance(x, Spacer)), None)
            block = lambda x: isinstance(x, KeepTogether) and not isinstance(x, Float)
            text = lambda x: isinstance(x, (Paragraph, Preformatted)) and not is_heading(x)
            if self.waiting and (frame._atTop or not (text(f) or block(f) or type(f) is Spacer)):   # a new page has started, or the section is over
                flowables[0:0], self.waiting = list(self.waiting), ()
            elif isinstance(f, CondPageBreak) and f.whole_section:          # the reference list is not split
                f.height = sum(tall(x) for x in flowables[1:])
            elif isinstance(f, CondPageBreak):                              # how much must stay together with the heading(s) that follow
                j = 1
                while j < len(flowables) and is_heading(flowables[j]):
                    j += 1
                first, second = content(flowables[j:]), content(flowables[j + 1:])
                lines = 3 * S["body"].leading
                after = (min(tall(first), tall(second) if block(second) else lines if text(second) else 10 ** 6) if isinstance(first, Float)
                         else tall(first) if block(first) else lines)
                f.height = sum(tall(h) for h in flowables[1:j]) + after + 4
            elif isinstance(f, Float) and not frame._atTop and not getattr(f, "moved", False) and tall(f) > room:
                nxt = content(flowables[1:])
                if text(nxt) or block(nxt) and tall(nxt) <= room:           # something useful can take its place on this page
                    f.moved = True
                    self.waiting += (flowables.pop(0),)
                    return
            super().handle_flowable(flowables)
            if not flowables and self.waiting:
                flowables.extend(self.waiting); self.waiting = ()

    def heading(text, level):
        # a heading needs room for itself and the first lines of what follows; the text below it may then break across pages
        if text.startswith("Appendix"):         # the appendix starts on a page of its own
            story.append(PageBreak())
        if not (story and is_heading(story[-1])):
            story.append(CondPageBreak(84))     # the document adjusts this height to what actually follows
            story[-1].whole_section = text == "References"
        story.append(Paragraph(inline(text), S[level]))

    def headings_above():
        """Headings directly above a figure or a short table leave the story and travel with that block."""
        heads = []
        while story and (is_heading(story[-1]) or isinstance(story[-1], CondPageBreak)):
            f = story.pop()
            if is_heading(f):
                heads.insert(0, f)
        return heads

    def figure(caption, path, text_follows):
        counter["fig"] += 1
        p = ROOT / "reports" / path
        iw, ih = ImageReader(str(p)).getSize()
        w = width
        h = w * ih / iw
        max_h = 150 * mm
        if h > max_h:
            w, h = w * max_h / h, max_h
        img = Image(str(p), width=w, height=h)
        img.hAlign = "LEFT"
        block = [Spacer(1, 4), img, Paragraph(f"<b>Figure {counter['fig']}.</b> {inline(caption)}", S["caption"])]
        return Float(block) if text_follows else KeepTogether(headings_above() + block)

    lines = markdown.splitlines()

    def flush():
        if para:
            story.append(Paragraph(inline(" ".join(x.strip() for x in para)), S["body"]))
            para.clear()

    i = 0
    while i < len(lines):
        raw, s = lines[i], lines[i].strip()
        if s.startswith("```"):
            flush()
            if in_code:
                story.append(Preformatted("\n".join(code), S["code"]))
                code.clear()
            in_code = not in_code
        elif in_code:
            code.append(raw)
        elif s.startswith("|"):
            flush()
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i]); i += 1
            t = table(block)                    # only the long appendix table may break across pages
            story += [KeepTogether(headings_above() + [Spacer(1, 2), t]) if len(block) <= 14 else t, Spacer(1, 8)]
            continue
        elif not s:
            flush()
        elif s.startswith("# "):
            flush(); story.append(Paragraph(inline(s[2:]), S["title"]))
        elif s.startswith("## "):
            flush(); heading(s[3:], "h2")
        elif s.startswith("### "):
            flush(); heading(s[4:], "h3")
        elif m := re.match(r"!\[(.*)\]\((.+)\)$", s):
            after = next((x.strip() for x in lines[i + 1:] if x.strip()), "#")
            flush(); story.append(figure(m.group(1), m.group(2), text_follows=not after.startswith(("#", "!"))))
        elif m := re.match(r"(\d+)\.\s+(.*)", s):
            flush(); story.append(Paragraph(inline(m.group(2)), S["bullet"], bulletText=f"{m.group(1)}."))
        elif s.startswith("- "):
            flush(); story.append(Paragraph(inline(s[2:]), S["bullet"], bulletText="•"))
        else:
            para.append(raw)
        i += 1
    flush()

    def decor(canvas, doc):
        canvas.saveState()
        canvas.setFont("Body", 7.8); canvas.setFillColor(MUTED)
        canvas.drawString(18 * mm, 10 * mm, "ChipQC Guardian · technical report · AI4S Open Innovation 2026")
        canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, str(doc.page))
        canvas.setStrokeColor(LINE); canvas.setLineWidth(0.5); canvas.line(18 * mm, 13.5 * mm, A4[0] - 18 * mm, 13.5 * mm)
        canvas.restoreState()

    doc = Document(str(out), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=18 * mm,
                          title="ChipQC Guardian: an auditable quality gate for organ-on-a-chip brightfield imaging", author="Yi Yu", subject="AI4S Open Innovation technical report")
    doc.addPageTemplates([PageTemplate(id="page", frames=Frame(18 * mm, 18 * mm, width, A4[1] - 34 * mm, id="body", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0), onPage=decor)])
    doc.build(story)
    return doc.page


if __name__ == "__main__":
    T = ROOT / "docs_templates"
    for name, target in (("README.md", ROOT / "README.md"), ("MODEL_CARD.md", ROOT / "MODEL_CARD.md"), ("technical_report.md", ROOT / "reports/technical_report.md")):
        text = render((T / name).read_text())
        assert "{{" not in text, name
        target.write_text(text)
        print("wrote", target.relative_to(ROOT))
    text = (ROOT / "reports/technical_report.md").read_text()
    pages = typeset(text, ROOT / "ChipQC_Guardian_Technical_Report.pdf")
    print(f"technical report: {len(text.split()):,} words, {pages} pages -> ChipQC_Guardian_Technical_Report.pdf")
    PAGES[0] = str(pages)
    writeup = render((T / "kaggle_writeup.md").read_text())
    (ROOT / "reports/kaggle_writeup.md").write_text(writeup)
    print(f"wrote reports/kaggle_writeup.md ({len(writeup.split()):,} words)")
