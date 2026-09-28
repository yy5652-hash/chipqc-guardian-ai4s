# ChipQC Guardian

ChipQC Guardian is a research prototype for human-reviewed quality control of brightfield images from organ-on-a-chip (OoC) experiments. The current Streamlit page shows image acquisition descriptors and an **unvalidated rule preview** labeled PASS, REVIEW, or REACQUIRE; if a compatible trained artifact exists, it shows the model's quality estimate separately. These suggestions do not replace a biologist's judgment. In particular, the source dataset's `good`/`bad` labels describe expert-assessed sample quality; a `bad` sample is not automatically an image that can be fixed by taking another photograph.

The project targets the [AI4S Open Innovation: AI for Life Science](https://www.kaggle.com/competitions/ai-4-s-open-innovation-artificial-intelligence-for-life-scien/overview/description) challenge in the **End-to-End System** category. Kaggle rules are accepted, the extra organizer registration form has confirmed receipt, and a Writeup draft exists. The final Writeup has not been submitted.

The [public browser demo](https://chipqc-guardian-ai4s.hudsonyuy.chatgpt.site) is a transparent, browser-only acquisition-rule preview. It does not execute the trained research model or upload visitor images to a server; the model-backed Streamlit application currently runs locally.

## What the system is designed to show

1. An operator uploads one or more brightfield images. The current page displays the original image, simple clarity/brightness/contrast/clipping descriptors, and explicit reasons for its rule preview; source metadata stay in the audit files.
2. If a compatible trained artifact is present, the page separately displays its `good`/`bad` estimate and probability. No model score is shown when no artifact is available.
3. The target operating policy would present PASS only at a validated acceptance threshold, REVIEW for ambiguity or unsupported conditions, and REACQUIRE for a separately supported acquisition-quality issue or a reviewer-confirmed need to re-image. The current rule thresholds and model confidence cutoff are demonstration settings without that validation.
4. A researcher reviews the suggestion. The current page exports a machine-generated CSV/JSON manifest; persistent logging of a human final decision is future work. No clinical, drug-efficacy, or tissue-viability conclusion is inferred from the image alone.

The decision thresholds are operational choices, not biological ground truth. The selected classifier threshold is `0.62`, chosen on calibration groups for balanced accuracy. No calibration threshold met the planned automatic-PASS target of at least 20 accepted images with ≤10% observed error, so operational acceptance remains human-controlled.

## Data and evidence status

The source is the public [Organ-on-a-Chip (OOC) Image Dataset](https://zenodo.org/records/10203721), described by [Movčana et al. (2024)](https://doi.org/10.3390/data9020028). Its spreadsheet has **3,072 non-empty image IDs**, six cell lines, and numeric quality labels `1` (1,727 rows) and `2` (1,345 rows). The 6.7 GB archive matched upstream MD5 `8f7e058996203d48eb03b2d86c0a2e4d`, and all 3,072 rows matched one decodable image with a consistent `1 = good`, `2 = bad` path label.

On the frozen 12-group, 670-image test set, the selected MobileNetV2-embedding + ExtraTrees model achieved **0.718 accuracy, 0.712 balanced accuracy, 0.712 macro-F1, 0.772 AUROC, and 0.204 Brier score**. Complete-group bootstrap intervals are wide, so these are internal evidence—not external or clinical validation. The paper's figures use a different protocol and are not project results.

## Reproduce the workflow

Use Python 3 in a fresh virtual environment and install the pinned dependencies when `requirements.txt` is available. Keep the upstream raw files under `data/raw/`; do not commit images or copy them into a public demo without checking the dataset record and contest usage terms.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/audit_metadata.py
python scripts/prepare_dataset.py
python scripts/train_baseline.py
python scripts/evaluate.py
python -m pip install -r requirements-vision.txt
python scripts/extract_embeddings.py
python scripts/train_vision_model.py
streamlit run app.py
```

The scripts should record their input paths, seed, split IDs, label mapping evidence, model artifact, and output location. Check `data/processed/matching_summary.json` before training or publishing a result; the current preparation code can export a matched subset if the image directory is incomplete. A result on such a subset must be named as a subset experiment, never as the full 3,072-image dataset. Run `python <script> --help` for script-specific options. The exact frozen run command, software versions, dataset checksums, and measured outputs belong in [the experiment protocol](reports/experiment_protocol.md) and [technical report](reports/technical_report.md).

## Evaluation design

The filename prefix before `_` is a six-digit acquisition-date-like group key. There are 59 distinct prefixes in the source sheet. All images with one prefix stay in one of train, validation, or test, reducing leakage from near-related acquisitions. The source archive's own `train`/`val`/`test` directories are not automatically an acceptable independence test for this project. Freeze a split manifest and confirm no image ID, exact duplicate, or date prefix crosses partitions before fitting preprocessing, calibration, or thresholds.

Report both discrimination and workflow behavior: per-class precision/recall, balanced accuracy and AUROC where defined, Brier score and reliability plot, PASS false-accept rate, REVIEW coverage, and performance by cell line. Include uncertainty intervals based on grouped resampling. Compare against simple metadata-only and image-only baselines under the same split. [MODEL_CARD.md](MODEL_CARD.md) and [reports/experiment_protocol.md](reports/experiment_protocol.md) define the intended tests.

## Project materials

| File | Purpose |
| --- | --- |
| [DATA_CARD.md](DATA_CARD.md) | Source, fields, missingness, rights, leakage risks |
| [MODEL_CARD.md](MODEL_CARD.md) | Intended use, decision logic, risks, and performance status |
| [reports/technical_report.md](reports/technical_report.md) | Detailed report manuscript and result slots |
| [reports/kaggle_writeup.md](reports/kaggle_writeup.md) | Kaggle Writeup copy and required links |
| [reports/demo_video_script_zh.md](reports/demo_video_script_zh.md) | Up-to-five-minute Chinese video storyboard |
| [reports/experiment_protocol.md](reports/experiment_protocol.md) | Frozen evaluation and reproducibility checklist |
| [reports/frozen_split_groups.json](reports/frozen_split_groups.json) | Exact 35/12/12 held-out prefix partition |
| [reports/rubric_self_score.md](reports/rubric_self_score.md) | Evidence-based rubric audit, not an official score |

## Rights and citation

Project-authored source code and documentation are covered by [LICENSE](LICENSE). The original OoC images and spreadsheet are third-party material and are **not** relicensed by this repository. Cite the [dataset DOI](https://doi.org/10.5281/zenodo.10203721) and the [data descriptor DOI](https://doi.org/10.3390/data9020028) when using them. Follow the current [competition participation guidelines](https://www.aicompetition-pz.com/guidelines) and the source record's rights information before distributing raw data, thumbnails, or trained weights. The confirmed external registration does not by itself submit the Kaggle Writeup.
