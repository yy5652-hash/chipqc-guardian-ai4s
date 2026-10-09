# ChipQC Guardian: an auditable quality gate for organ-on-a-chip brightfield imaging

**Submission category: End-to-End System**

## 中文摘要

器官芯片培养在给药、染色或测量之前，都要有人看明场图像，判断培养是否可用。这个判断靠肉眼，因人而异，而且几乎不留记录。**ChipQC Guardian** 把它变成有记录、可核查的一步：每一帧图像得到三种结果之一，并附上依据。**PASS** 表示无需人工查看、直接使用；**REACQUIRE** 表示这不是有效观测，应趁芯片还在显微镜上重拍；**REVIEW** 表示交给人判断，最可疑的排在最前。

我们在公开的器官芯片图像数据集（3,072 帧，6 种细胞系，59 个采集日期）上按**整个采集日期**留出测试，因为同一天拍摄的图像彼此并不独立。模型 AUROC 为 **{{ f3(H.auroc.value) }}**（95% 区间 {{ ci(H.auroc) }}）。在 10% 的错误目标下，{{ pct(S10.passed_automatically).replace(" ", "") }} 的帧自动放行，其中专家判为不良的占 {{ pct1(S10.bad_among_passed).replace(" ", "") }}；{{ pct(S10.sent_to_reacquire).replace(" ", "") }} 的帧在拍摄当时就被要求重拍。每个判断都写入审计记录（结果、校准概率、采集指标、模型版本、图像哈希），可以作为器官芯片数据资产和数字孪生的数据质量层。以下为英文详述。

## Demo video

- YouTube: {{ links.video }} (also the first item of the media gallery above)
{{ "- Bilibili: " + links.video_bilibili if links.video_bilibili else "" }}
- File: `ChipQC_Guardian_Demo_Video.mp4` (63 MB), attached to this writeup under Project Files

A {{ links.video_length }} film made from the real system and the real data: every frame of the dataset sorted by the outcome it received on a day the model never saw, an evidence map building up patch by patch, the acquisition gate reacting to controlled faults, the review console, the browser demo and the command line, then the measured results and the limits. Narration is a synthetic voice and the soundtrack is generated. The film was recorded with the previous version of the model (3 × 3 tiles, AUROC {{ f3(REP.dinov2_vits14.auroc.value) }}); the system, the workflow and the limits it shows are unchanged, and the numbers in this writeup and in the report are those of the released model.

## Code repository

https://github.com/yy5652-hash/chipqc-guardian-ai4s

Source, released model, embeddings of all 3,072 frames, result files, figures, tests and the scripts that regenerate every number. `python evaluate.py` reproduces the full evaluation without downloading any images: about three minutes on an Apple-silicon desktop, longer on a small cloud machine (`--only headline` is the quick check there).

## Project summary

Before an organ-on-a-chip (OoC) culture is dosed, stained or measured, someone looks at brightfield frames and decides whether the culture is usable. That decision is made by eye, differs between people and is rarely recorded. **ChipQC Guardian** turns it into a logged, checkable step: every frame gets **PASS** (use it without a person looking), **REACQUIRE** (not a usable observation: image it again) or **REVIEW** (a person decides, most suspicious first), with the evidence behind the outcome.

An acquisition gate measures five physical properties of the image. A culture-quality model reads the whole 2056 × 1542 px frame with a frozen self-supervised vision transformer (DINOv2 ViT-S/14), read as 4 × 4 tiles at 1792 px, and a linear classifier on the last four blocks, so each of the {{ cells }} image patches has an exact share of the score, drawn as an evidence map. A frame passes automatically only above a threshold fitted on other acquisition dates to keep the share of bad frames among passed frames under a chosen target.

On the public Organ-on-a-Chip Image Dataset (3,072 frames, six cell lines, 59 acquisition dates) we hold out **whole acquisition dates**, because frames from one day are not independent. The model reaches an AUROC of **{{ f3(H.auroc.value) }}** (95 % interval {{ ci(H.auroc) }}); our previous version scored {{ f3(REP.dinov2_vits14.auroc.value) }} under the same protocol, and on the frozen test dates of our first version the model scores {{ f3(FS.dinov2_vits14_4x4_l4.auroc) }} against that version's {{ f3(FS.first_submission.auroc) }}. At a 10 % target, {{ pct(S10.passed_automatically) }} of frames pass with {{ pct1(S10.bad_among_passed) }} bad among them and {{ pct(S10.sent_to_reacquire) }} are sent back for re-imaging. A cell line withheld from training loses at most {{ f2(max_line_drop) }} AUROC; a different camera loses about ten points, until local labels are added; automatic rejection is not supported by the data. We report all three. The value is a quality decision with a stated error rate, visible evidence and an audit record that travels with the image.

## Technical report

[ChipQC Guardian technical report (PDF, {{ pages }} pages)](https://github.com/yy5652-hash/chipqc-guardian-ai4s/blob/main/ChipQC_Guardian_Technical_Report.pdf), also attached to this writeup under Project Files. It covers the problem, data and compliance, system design, experimental protocol, all results, reliability analysis, impact, reproduction and the licences of every external asset. Its main results, the reasons behind the design and the limits are summarised here.

### Results at a glance

All numbers are for acquisition dates the model never saw; intervals resample dates.

| | |
|---|---|
| AUROC | **{{ f3(H.auroc.value) }}** ({{ ci(H.auroc) }}) |
| Same protocol, our previous version (3 × 3 tiles, last block only) | {{ f3(REP.dinov2_vits14.auroc.value) }} |
| Same protocol, 224 px centre crop (the representation of our first version) | {{ f3(REP.mobilenet_v2_224crop.auroc.value) }} |
| Accuracy on the dataset authors' own split (published baseline: 0.81) | {{ f2(LK.authors_split.accuracy) }} |
| Largest AUROC change when a cell line is withheld from training (six lines) | {{ f2(max_line_drop) }} |
| Different camera without local labels | {{ f2(CTX.main.grey_to_colour) }} / {{ f2(CTX.main.colour_to_grey) }} (same camera: {{ f2(CT.within_camera_dates_held_out.colour.auroc) }} / {{ f2(CT.within_camera_dates_held_out.grey.auroc) }}) |
| Passed automatically at a 10 % / 20 % target | {{ pct(S10.passed_automatically) }} / {{ pct(S20.passed_automatically) }} |
| Bad among passed at a 10 % / 20 % target | {{ pct1(S10.bad_among_passed) }} / {{ pct1(S20.bad_among_passed) }} |
| Controlled faults caught by the acquisition gate | defocus 4 px {{ pct(DS.defocus[2]) }}, 20 px motion {{ pct(DS.motion_streak[2]) }}, 40 % occlusion {{ pct(DS.occlusion[2]) }} |
| Time per frame | {{ RT.gpu_s }} s on a laptop GPU, {{ RT.cpu_s }} s on a laptop CPU |

### Why this problem

OoC platforms are moving from single experiments to studies with hundreds of chips, and regulators now accept non-animal methods in preclinical testing. Quality control has not kept up: whether a culture is fit to analyse is still a personal judgement that leaves no trace in the data. A wrong "yes" contaminates every downstream measurement and every dataset later used to train models or digital twins; a wrong "no" discards days of culture. A gate that states its own error rate, shows its evidence and writes an audit record makes that judgement reproducible and turns the quality state of each frame into metadata that travels with the image. At a 10 % target about {{ int(1000 * S10.passed_automatically) }} frames per 1,000 need no look; at an assumed 10 to 30 seconds per look that is {{ minutes(1000 * S10.passed_automatically, 10) }} to {{ minutes(1000 * S10.passed_automatically, 30) }} minutes of looking avoided per 1,000 frames, and {{ minutes(1000 * S20.passed_automatically, 10) }} to {{ minutes(1000 * S20.passed_automatically, 30) }} at a 20 % target (report, section 7.1). The audit record is also the quality layer that an organ-on-a-chip data asset, and a digital twin trained on it, needs first (section 7.2).

### What is technically new

- **The whole frame at resolution.** Keeping the entire field of view at 1344 px instead of a 224 px crop is worth {{ f3(REP.mobilenet_v2_1344.auroc.value - REP.mobilenet_v2_224crop.auroc.value) }} AUROC with the same backbone.
- **Self-supervised features for robustness.** DINOv2 patch descriptors transfer to unseen cell lines with at most {{ f2(max_line_drop) }} AUROC lost and to an unseen camera better than any supervised backbone we tested.
- **A deeper read-out of the same frozen backbone.** Describing each patch by the last four transformer blocks instead of the last one, on 4 × 4 tiles, adds {{ gain(REP.dinov2_vits14.auroc_minus_released) }} AUROC over our previous version at no extra parameters, and the model still runs in a browser.
- **Exact evidence maps.** The explanation is the model's own arithmetic, not a post-hoc approximation.
- **Two stages for two kinds of "bad".** A frame that is not a usable observation goes back to the microscope; a culture that looks poor goes to a person. The experts' "bad" label mixes both, and they call for different actions.
- **Validation that matches how the data were produced.** Dates are held out, tuning stays inside training dates, intervals resample dates. The same model scores {{ f3(LK.random_image_folds.auroc) }} AUROC on a random split of the images and {{ f3(H.auroc.value) }} on held-out dates.
- **A decision rule that states its own error rate**, with a safety margin, and two documented negative results (no automatic rejection, no 5 % target).

### Limits

One laboratory and one chip family; labels are expert judgements without per-rater votes; 59 dates give wide intervals; a new camera needs local reference frames; the system is not validated prospectively and is not for clinical or regulatory use. The technical report quantifies each of these.

## Interactive demo

{{ links.demo }}

The released pipeline running entirely in the visitor's browser (ONNX Runtime Web; no login, nothing uploaded): pick one of nine real frames from two acquisition dates the model never saw, or open your own image, and get the outcome, the evidence map and the acquisition descriptors. On the bundled frames the browser result matches the Python reference to within {{ f6(WEB.largest_p_good_difference) }} in P(good). The repository also contains a local review console (`streamlit run app.py`) and a command-line tool.

## Team

Yi Yu, New York University (individual entry; no cross-disciplinary bonus claimed). Data: Movčana et al., Organ-on-a-Chip (OOC) Image Dataset, Zenodo, doi:10.5281/zenodo.10203721, CC BY 4.0. Backbone: DINOv2 (Meta AI), Apache-2.0. AI coding assistants were used to write code and edit text under the author's direction.
