# ChipQC Guardian: Human-reviewed quality control for OoC brightfield images

> **Evidence status:** The external registration form confirmed receipt, and this Kaggle Writeup is still a draft. The reported metrics are from a frozen internal test split, not an official competition score or external validation.

**Submission category: End-to-End System**

## Demo video

The 83-second [demonstration video](https://www.youtube.com/watch?v=QYr0GIJ8JRs) shows the research workflow, synthetic interface examples, and internally measured held-out results. It is uploaded but remains private while YouTube's copyright check is ongoing; this Writeup will not be submitted until public playback is verified.

## Public code repository

[GitHub: ChipQC Guardian AI4S](https://github.com/yy5652-hash/chipqc-guardian-ai4s). The repository offers browsable source code, evaluation records, the [exact frozen group split](https://github.com/yy5652-hash/chipqc-guardian-ai4s/blob/main/reports/frozen_split_groups.json), and a downloadable source package; third-party raw images are excluded.

## Public browser demo

[Open the browser-only quality-rule preview](https://chipqc-guardian-ai4s.hudsonyuy.chatgpt.site). It runs four transparent image descriptors in the visitor's browser and labels its PASS/REVIEW/REACQUIRE output as an **unvalidated workflow preview**. The trained research model is not served by this page; its measurements come from the separate frozen internal evaluation and local Streamlit application. The preview sends no uploaded image to a server.

## Project summary (approximately 230 words)

Organ-on-a-chip (OoC) experiments generate brightfield images that researchers inspect to judge whether a culture is suitable for further analysis. ChipQC Guardian explores a more transparent quality-control workflow: it presents the original image, an interpretable quality estimate, and an explicit suggestion to PASS, REVIEW, or REACQUIRE. Researchers remain responsible for the final choice. The source's expert `good`/`bad` labels describe sample quality; a poor sample is not automatically fixed by taking another photograph, so a re-acquisition suggestion requires separate evidence of an unusable image or a human decision.

We use the public OoC image dataset of Movčana and colleagues. Its spreadsheet contains 3,072 unique image IDs across six cell lines. The official 6.7 GB archive passed its MD5 check; all 3,072 rows matched a decodable image, with `1 = good` and `2 = bad` consistent with the archive paths. Rather than randomly dividing images, we group all images sharing one of 59 six-digit date-like filename prefixes into the same partition. This guards against related acquisitions appearing on both sides of a test split.

The stronger model uses public ImageNet MobileNetV2 embeddings, an ExtraTrees classifier selected only on calibration groups, and logistic probability calibration. On 670 images from 12 held-out prefix groups, it achieved 0.718 accuracy, 0.712 balanced accuracy, 0.712 macro-F1, 0.772 AUROC, and 0.204 Brier score. Group-bootstrap intervals are wide, and no calibration threshold met our planned ≤10% accepted-error objective with at least 20 accepted images. We therefore show the research score but do not claim a safe automated PASS policy.

## Problem and application value

Routine brightfield inspection is a bottleneck when many OoC samples and time points are imaged. A binary model output alone is hard to act on: the cost of mistakenly accepting a poor-quality sample differs from the cost of asking a researcher to review an uncertain image. ChipQC Guardian separates the model's estimate from an operational decision and shows the reason for each suggestion. Its intended value is to prioritize human attention and make acceptance rules auditable. Time saved, error reduction, and generalization to another laboratory remain hypotheses until measured prospectively.

## Data and safeguards

The [OoC dataset](https://doi.org/10.5281/zenodo.10203721) contains brightfield images from six cell lines, described in the [dataset paper](https://doi.org/10.3390/data9020028). The local spreadsheet audit found 3,072 records, 59 acquisition-date-like prefixes, and missing values in seeding density (344), elapsed hours (2,244), and flow rate (859). The full image-to-label join passed with no conflicts. Raw image rights are separate from this repository's code license.

All images with the same prefix remain in one of train, validation, or test. We inspect class and cell-line balance, exact/near duplicates, and missingness patterns. The prefix is a conservative proxy, not a confirmed chip or donor ID. This makes our planned held-out evaluation more meaningful than a random image split, but it does not prove external generalization.

## Method and interface

The reproducible baseline extracts pixel descriptors for brightness, contrast, entropy, gradients, edges, and focus-related structure. The selected model instead uses a frozen 1,280-dimensional MobileNetV2 representation and an ExtraTrees classifier; three capacity settings were compared on calibration groups only. Logistic calibration and a 0.62 classification threshold were fitted on those groups before one test evaluation. The interface labels acquisition-rule PASS/REVIEW/REACQUIRE outputs as unvalidated previews and shows the research classifier separately. It keeps the original image visible and exports a manifest. No output is a clinical, drug-response, or tissue-viability claim.

## Results and validation

The official 6.7 GB archive matched MD5 `8f7e058996203d48eb03b2d86c0a2e4d`; all 3,072 rows matched and decoded without label-path conflict. The frozen split contains 35/12/12 train/calibration/test groups and 1,804/598/670 images. On test, the selected model achieved **0.718 accuracy, 0.712 balanced accuracy, 0.712 macro-F1, 0.772 AUROC, and 0.204 Brier score**. Its confusion matrix (rows actual bad/good) was `[[193,120],[69,288]]`. Complete-group bootstrap 95% intervals were 0.510–0.815 for balanced accuracy and 0.552–0.875 for AUROC, showing substantial uncertainty. A549 test balanced accuracy was 0.551 versus 0.848 for CACO; HSAEC and NHBE slices are too sparse for stable conclusions. The handwritten-feature baseline reached only 0.508 balanced accuracy and 0.536 AUROC. The paper's approximately **0.81 accuracy** uses a different protocol and is not our score.

No calibration threshold accepted at least 20 images with ≤10% observed error. Accordingly, the current prototype does not automate PASS: the final operational action remains a human decision. This negative result is part of the evidence, not hidden by the interface.

## Reproduce and inspect

Browse the [public source repository](https://github.com/yy5652-hash/chipqc-guardian-ai4s) or download its source package, install `requirements.txt` and `requirements-vision.txt`, obtain the source data from Zenodo, and run `audit_metadata.py`, `prepare_dataset.py`, `train_baseline.py`, `extract_embeddings.py`, `train_vision_model.py`, and `evaluate.py`. Start the interface with `streamlit run app.py`. The package contains the image manifest and evaluation outputs; the repository also includes the [frozen split groups](https://github.com/yy5652-hash/chipqc-guardian-ai4s/blob/main/reports/frozen_split_groups.json), protocols, and model/data cards. Neither distribution includes third-party raw images. The [final technical report PDF](https://github.com/yy5652-hash/chipqc-guardian-ai4s/blob/main/ChipQC_Guardian_Technical_Report.pdf) explains limits and measurements. This is not yet a verified fresh-install reproduction.

## Links and references

- Interactive demo: [public browser-only rule preview](https://chipqc-guardian-ai4s.hudsonyuy.chatgpt.site); the trained model runs only in the local Streamlit prototype, so no public model-backed deployment is claimed.
- Dataset: [Movčana et al., Zenodo 10.5281/zenodo.10203721](https://doi.org/10.5281/zenodo.10203721)
- Dataset paper: [Movčana et al., *Data* 9(2), 28 (2024)](https://doi.org/10.3390/data9020028)
- Challenge: [AI4S Open Innovation: AI for Life Science](https://www.kaggle.com/competitions/ai-4-s-open-innovation-artificial-intelligence-for-life-scien/overview/description)
