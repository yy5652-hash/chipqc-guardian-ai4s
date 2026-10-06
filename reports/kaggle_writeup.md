# ChipQC Guardian: an auditable quality gate for organ-on-a-chip brightfield imaging

**Submission category: End-to-End System**

## Demo video

- YouTube: https://youtu.be/TDwHyMEzVOM (also the first item of the media gallery above)
- Bilibili: https://www.bilibili.com/video/BV1h1Hd6vESY
- File: `ChipQC_Guardian_Demo_Video.mp4` (63 MB), attached to this writeup under Project Files

A 3½-minute film made from the real system and the real data: every frame of the dataset sorted by the outcome it received on a day the model never saw, an evidence map building up patch by patch, the acquisition gate reacting to controlled faults, the review console, the browser demo and the command line, then the measured results and the limits. Narration is a synthetic voice and the soundtrack is generated.

## Code repository

https://github.com/yy5652-hash/chipqc-guardian-ai4s

Source, released model, embeddings of all 3,072 frames, result files, figures, tests and the scripts that regenerate every number. `python evaluate.py` reproduces the full evaluation in two minutes on a CPU without downloading any images.

## Project summary

Before an organ-on-a-chip (OoC) culture is dosed, stained or measured, someone looks at brightfield frames and decides whether the culture is usable. That decision is made by eye, differs between people and is rarely recorded. **ChipQC Guardian** turns it into a logged, checkable step: every frame gets **PASS** (use it without a person looking), **REACQUIRE** (not a usable observation: image it again) or **REVIEW** (a person decides, most suspicious first), with the evidence behind the outcome.

An acquisition gate measures five physical properties of the image. A culture-quality model reads the whole 2056 × 1542 px frame with a frozen self-supervised vision transformer (DINOv2 ViT-S/14) and a linear classifier, so each of the 6,912 image patches has an exact share of the score, drawn as an evidence map. A frame passes automatically only above a threshold fitted on other acquisition dates to keep the share of bad frames among passed frames under a chosen target.

On the public Organ-on-a-Chip Image Dataset (3,072 frames, six cell lines, 59 acquisition dates) we hold out **whole acquisition dates**, because frames from one day are not independent. The model reaches an AUROC of **0.852** (95 % interval 0.821–0.883); on the frozen test dates of our first version it scores 0.859 against that version's 0.772. At a 10 % target, 18 % of frames pass with 8.6 % bad among them and 15 % are sent back for re-imaging. No cell line loses accuracy when withheld from training; a different camera does, until local labels are added; automatic rejection is not supported by the data. We report all three. The value is a quality decision with a stated error rate, visible evidence and an audit record that travels with the image.

## Technical report

[ChipQC Guardian technical report (PDF, 16 pages)](https://github.com/yy5652-hash/chipqc-guardian-ai4s/blob/main/ChipQC_Guardian_Technical_Report.pdf), also attached to this writeup under Project Files. It covers the problem, data and compliance, system design, experimental protocol, all results, reliability analysis, impact, reproduction and the licences of every external asset. Its main results, the reasons behind the design and the limits are summarised here.

### Results at a glance

All numbers are for acquisition dates the model never saw; intervals resample dates.

| | |
|---|---|
| AUROC | **0.852** (0.821–0.883) |
| Same protocol, 224 px centre crop (the representation of our first version) | 0.738 |
| Accuracy on the dataset authors' own split (published baseline: 0.81) | 0.86 |
| Largest AUROC change when a cell line is withheld from training (six lines) | 0.02 |
| Different camera without local labels | 0.71 / 0.74 (same camera: 0.79 / 0.84) |
| Passed automatically at a 10 % / 20 % target | 18 % / 47 % |
| Bad among passed at a 10 % / 20 % target | 8.6 % / 17.1 % |
| Controlled faults caught by the acquisition gate | defocus 4 px 100 %, 20 px motion 99 %, 40 % occlusion 100 % |
| Time per frame | 0.14 s on a laptop GPU, 0.43 s on a laptop CPU |

### Why this problem

OoC platforms are moving from single experiments to studies with hundreds of chips, and regulators now accept non-animal methods in preclinical testing. Quality control has not kept up: whether a culture is fit to analyse is still a personal judgement that leaves no trace in the data. A wrong "yes" contaminates every downstream measurement and every dataset later used to train models or digital twins; a wrong "no" discards days of culture. A gate that states its own error rate, shows its evidence and writes an audit record makes that judgement reproducible and turns the quality state of each frame into metadata that travels with the image.

### What is technically new

- **The whole frame at resolution.** Keeping the entire field of view at 1344 px instead of a 224 px crop is worth 0.086 AUROC with the same backbone.
- **Self-supervised features for robustness.** DINOv2 patch descriptors transfer to unseen cell lines without loss and to an unseen camera better than any supervised backbone we tested.
- **Exact evidence maps.** The explanation is the model's own arithmetic, not a post-hoc approximation.
- **Validation that matches how the data were produced.** Dates are held out, tuning stays inside training dates, intervals resample dates. The same model scores 0.918 AUROC on a random split of the images and 0.852 on held-out dates.
- **A decision rule that states its own error rate**, with a safety margin, and two documented negative results (no automatic rejection, no 5 % target).

### Limits

One laboratory and one chip family; labels are expert judgements without per-rater votes; 59 dates give wide intervals; a new camera needs local reference frames; the system is not validated prospectively and is not for clinical or regulatory use. The technical report quantifies each of these.

## Interactive demo

https://yy5652-hash.github.io/chipqc-guardian-ai4s/

The released pipeline running entirely in the visitor's browser (ONNX Runtime Web; no login, nothing uploaded): pick one of nine real frames from two acquisition dates the model never saw, or open your own image, and get the outcome, the evidence map and the acquisition descriptors. On the bundled frames the browser result matches the Python reference to within 0.000001 in P(good). The repository also contains a local review console (`streamlit run app.py`) and a command-line tool.

## Team

Yi Yu, New York University (individual entry; no cross-disciplinary bonus claimed). Data: Movčana et al., Organ-on-a-Chip (OOC) Image Dataset, Zenodo, doi:10.5281/zenodo.10203721, CC BY 4.0. Backbone: DINOv2 (Meta AI), Apache-2.0. AI coding assistants were used to write code and edit text under the author's direction.
