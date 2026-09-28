# ChipQC Guardian: Group-aware, human-reviewed quality control for organ-on-a-chip brightfield images

**Submission category:** End-to-End System  
**Report status:** Working manuscript with verified internal grouped-test results. Public repository, public video, final competition form, and external validation remain pending.  
**Team:** Yi Yu, New York University  
**Date:** 2026-09-28 (draft)

## Abstract

Organ-on-a-chip (OoC) experiments can produce many brightfield images across cell lines and cultivation times. Researchers inspect these images to assess sample quality, yet binary classification alone does not tell an operator when to trust a model or request expert review. ChipQC Guardian is a research prototype that combines an auditable image-quality classifier with human review and a three-action interface: PASS, REVIEW, and REACQUIRE. The source is the public OoC image dataset of Movčana and colleagues, with 3,072 metadata records across six cell lines. We use the six-digit prefix of each image ID as a conservative acquisition-date-like group key so related images do not cross training, calibration, and test partitions. The transparent baseline extracts deterministic image statistics; the stronger model uses frozen ImageNet MobileNetV2 embeddings, an ExtraTrees classifier selected on calibration groups, and logistic probability calibration. On 670 images from 12 held-out prefix groups, the selected model reached 0.718 accuracy, 0.712 balanced accuracy, 0.772 AUROC, and 0.204 Brier score. Group-bootstrap 95% intervals were wide (balanced accuracy 0.510–0.815; AUROC 0.552–0.875), and no calibration threshold met the planned ≤10% accepted-error objective with at least 20 accepted images. The system therefore does not claim a safe automatic PASS policy or practical time saving; ambiguous and unsupported cases remain human-reviewed.

**Keywords:** organ-on-a-chip; brightfield microscopy; image quality control; group leakage; probability calibration; human review; selective prediction.

## 1. Problem and application setting

OoC systems place living cells in engineered microfluidic environments. Brightfield imaging permits repeated visual monitoring without necessarily consuming the sample. A routine experiment can generate multiple fields or time points for a chip, and a researcher may need to decide whether an image shows a usable culture, whether it needs closer review, and whether the image acquisition itself should be repeated. These are related but different questions. An expert's judgment of a biologically poor sample cannot be converted directly into an instruction to photograph it again.

The [AI4S Open Innovation challenge](https://www.kaggle.com/competitions/ai-4-s-open-innovation-artificial-intelligence-for-life-scien/overview/description) encourages runnable, explainable AI systems for life science, including OoC use cases. This project selects a narrow task with a real public dataset: quality screening of OoC brightfield images. The proposed user is a researcher or laboratory operator reviewing experimental images. The goal is to prioritize expert attention and make the model's uncertainty and provenance visible at the moment of review. The system is not intended to decide whether a culture is functionally equivalent to a human organ, whether a drug works, or whether a patient has a disease.

The source [data descriptor](https://doi.org/10.3390/data9020028) reports a MobileNetV3 test result of 0.81 accuracy, 0.79 precision, and 0.78 recall under its own protocol. Those are **published literature results**. They are neither our model's results nor a valid target to beat without reproducing the same data split and task. Our central methodological concern is that an image-level split may place similar acquisitions across train and test, inflating apparent generalization. We therefore make acquisition-date-like grouping, calibration, and abstention explicit in both implementation and reporting.

### 1.1 Research questions

1. Does a transparent image-feature baseline retain useful discrimination on image groups whose six-digit filename prefix was not used for training?
2. Do its predicted probabilities remain calibrated on held-out groups, including under the six cell-line subpopulations?
3. Can a PASS band provide useful coverage while keeping the fraction of truly poor-quality samples suggested PASS within a declared operating bound?
4. Which samples need REVIEW, and can an independent acquisition-quality check identify images that genuinely merit REACQUIRE?
5. How much of any predictive signal comes from image content rather than cell line, acquisition context, or missing metadata?

These questions are deliberately measurable. Answers require the final grouped-test outputs and, for time savings, a future prospective workflow study.

## 2. Source dataset and provenance

The [Organ-on-a-Chip (OOC) Image Dataset](https://doi.org/10.5281/zenodo.10203721) contains automated brightfield images of cells grown in OoC devices. Its associated spreadsheet supplies image IDs, cell type, seeding density, elapsed hours after seeding when present, a day field, an expert quality decision, and flow rate when present. The data descriptor identifies A549, Caco-2, HPMEC, HUVEC, NHBE, and HSAEC cell lines and states that a cell-biology expert assigned `good` or `bad` quality labels. This is a sample-quality task with possible acquisition artifacts, not a directly labeled image-retake task.

The local sheet `OOC_datasheet.xlsx` contains 3,072 rows with non-empty, unique image IDs. Its source column `Decision 1/2 (good/bad)` encodes 1,727 rows as `1` and 1,345 as `2`. The heading alone does not specify which number has which meaning. We cross-checked two official archive-preview paths against the sheet: `221010_82.png` under `test/good/A549/0-1_days` has decision `1`; `230529_207.png` under `test/bad/A549/4+_days` has decision `2`. Accordingly, the working source mapping is `1 = good`, `2 = bad`. This is a sample-verified mapping. A full one-to-one image-path join must confirm all rows before any trained release.

### 2.1 Dataset profile

| Cell-line code | Total images | `1 = good` | `2 = bad` |
| --- | ---: | ---: | ---: |
| HPMEC | 1,462 | 798 | 664 |
| A549 | 775 | 537 | 238 |
| CACO | 346 | 109 | 237 |
| HSAEC | 244 | 163 | 81 |
| NHBE | 138 | 105 | 33 |
| HUVEC | 107 | 15 | 92 |
| **Total** | **3,072** | **1,727** | **1,345** |

The sheet spells the second line `CACO`; the source description uses Caco-2. We preserve the raw value and only normalize display text explicitly. Class balance differs strongly by line. A model can appear adequate overall while performing poorly on HUVEC or NHBE. Grouped analysis is essential because there are only 59 distinct six-digit prefixes, and their sizes range from 6 to 229 images. Some prefixes are almost all one label; the metadata audit found, for example, prefix `230405` with 138 label-2 images and `230119` with 70 label-1 images. This is a concrete reason to suspect collection-context confounding.

The three most important missing fields are seeding density (344 missing), time after seeding in hours (2,244 missing), and flow rate (859 missing). A missing value is not a zero dose, zero elapsed time, or zero flow. The first baseline uses image pixels and cell line only for slice analysis, reducing the temptation to learn collection shortcuts from these incomplete fields. A separate metadata-only experiment will test how much predictive signal the context carries.

### 2.2 Archive integrity and rights

The 6.7 GB image archive matched the upstream MD5 checksum `8f7e058996203d48eb03b2d86c0a2e4d`. Python's ZIP64 reader enumerated 3,237 archive members and `testzip()` returned no corrupt member. The preparation pipeline matched and decoded all 3,072 spreadsheet image IDs, with zero missing, ambiguous, path-label-conflicting, or decode-failed rows. The system `unzip` binary could not parse this ZIP64 archive and reported an extra-byte/central-directory error; that tool limitation is recorded rather than misreported as source corruption because the upstream checksum matched and Python's ZIP64 validation and full decode succeeded.

The [paper](https://doi.org/10.3390/data9020028) says `Dataset License: CC-BY-SA`; the exact version and current Zenodo rights text must be checked before redistribution. The project MIT license covers project-authored code and documentation, not raw images. Competition use and public demo publication are subject to the current [organizer guidelines](https://www.aicompetition-pz.com/guidelines) and source rights. We intend to link reviewers to the source rather than put a multi-gigabyte archive in the repository.

## 3. System design

ChipQC Guardian separates four layers: (i) source and identity audit, (ii) deterministic pixel feature extraction and supervised model, (iii) probability calibration and decision policy, and (iv) a human-facing review interface. This separation matters because a good-looking interface, a probability score, and a biologically valid action are different forms of evidence.

```text
Zenodo image + spreadsheet
  -> checksum, ID/path/label audit
  -> group-frozen manifest and pixel features
  -> training-group classifier
  -> independent-group probability calibration
  -> image-usability check + score policy
  -> PASS / REVIEW / REACQUIRE suggestion
  -> human judgment; persistent reviewer-reason logging is future work
```

The current interface accepts uploaded images, shows the original image and acquisition-quality descriptors, marks model availability, displays a model score if a compatible artifact is loaded, and explains the reason for its rule preview. Source context is available in the data-audit files but is not yet displayed on this page. If the model artifact is absent or incompatible, the page says so. The illustrative quality heuristic demonstrates what low clarity, darkness, or clipping looks like, but until its thresholds have been evaluated against relevant labels it is a **demo rule**, not an accuracy result.

### 3.1 Image representation and models

The current baseline computes 24 reproducible pixel-derived descriptors after EXIF orientation handling, RGB conversion, and bounded resizing to a maximum side of 512 pixels for statistics. They include original width/height and aspect ratio; grayscale brightness mean and percentiles; dark and bright pixel fractions; contrast; grayscale entropy; color means and variation; Laplacian variance; gradient magnitude and edge fraction; and spatial illumination variation. These are inexpensive to compute and can help an operator inspect acquisition conditions. They are not a full morphology representation and may miss biologically important patterns.

The baseline imputes missing feature values by the training median, standardizes features, fits class-weighted logistic regression, and calibrates its decision score on independent groups. It performed poorly on held-out groups (0.508 balanced accuracy and 0.536 AUROC), so it is retained as a falsifiable lower reference rather than promoted as the product model.

The selected model applies the public torchvision MobileNetV2 `IMAGENET1K_V2` preprocessing and frozen backbone to obtain a 1,280-dimensional image representation. ExtraTrees candidates with three `max_features` settings were fit only on 35 training groups and ranked only on 12 calibration groups by balanced accuracy plus AUROC. The selected `max_features=0.5` model was then calibrated with logistic regression on the calibration groups. A 0.62 classification threshold maximized calibration balanced accuracy; after selection, the 12-group test partition was evaluated once. This is transfer learning with a frozen representation, not end-to-end biological pretraining, and ImageNet features may encode acquisition style rather than transferable cell morphology.

### 3.2 Operational decisions

Let `p_good` be the calibrated estimate for the source expert's `good` class. A demonstration rule may classify `p_good >= t` as likely good, `p_good <= 1-t` as likely bad, and the interval between them as REVIEW. The current baseline's default `t = 0.8` is an implementation parameter, **not** a validated safety threshold. A release policy must choose and freeze the threshold on calibration data against a stated false PASS objective, then report its held-out coverage and error.

PASS means the image is readable and the model is sufficiently confident under its evaluated conditions to present a positive suggestion for human confirmation. REVIEW covers uncertain scores, data outside validation support, or incompatible model/data states. REACQUIRE must be tied to a separate image-usability condition or a human-confirmed need for repeat imaging. Low `p_good` from a biologically poor sample should prompt REVIEW or a sample-quality warning; it does not prove that another photograph will improve the sample. This distinction is a core limit of the source labels and should appear on screen and in the video.

### 3.3 Human override and auditability

The desired workflow keeps the original image and a short reason next to every suggested action. A trained reviewer can accept, reject, or amend the suggestion. The prototype should preserve the distinction between a model score, a heuristic acquisition check, and a human final decision. When reviewer event logging is implemented, the log should store image ID, timestamp, artifact version, suggested action, reviewer choice, and reason while respecting laboratory data governance. A demonstration without persistent event logging must be described as a review interface prototype, not an audited production workflow.

## 4. Leakage-aware experiment design

The primary test of generalization uses the first six characters of `imageID` as a date-like group key. All images sharing a prefix must stay in the same train, calibration, or test set. The current implementation searches deterministic GroupShuffleSplit candidates to approximate a 60/20/20 image allocation while maintaining both labels and reasonable cell-line balance; the actual group and image counts are recorded after the complete archive is prepared. This is a stronger independence barrier than random images, but the prefix does not prove separate chips, donors, or experiments. If stronger provenance becomes available, the analysis should group by the strongest shared source.

Preprocessing fit only on training groups. Calibration fit only on calibration groups. The test set remains closed during model and policy selection. A source directory named `train`, `val`, or `test` is recorded but does not override this primary grouping. The split-search algorithm uses labels and cell-line composition for partition balance, not image features or model outcomes; the report should disclose this design choice. Every released experiment must include a split manifest and hash, and assert that image IDs and prefixes do not cross partitions. Exact and perceptual duplicate checks add another leakage barrier.

### 4.1 Comparators and ablations

The first reference is a majority-class predictor calculated only from training data. The second is the image-statistic logistic baseline. Planned ablations include dropping dimensions, brightness, or color features to test acquisition shortcuts; metadata-only and missingness-only predictors to reveal context confounding; and a conditional fusion model if it adds value without leaking source identity. Each comparator must use the same group split and model-selection discipline. A random image split may be shown as a secondary demonstration of leakage risk, clearly labeled non-primary, rather than as the headline score.

The six cell lines also support a leave-one-cell-line-out stress check, but group and label availability may make some folds unstable. These analyses should remain exploratory. A high score on the six source lines cannot establish performance on a different microscope, chip geometry, stain, or laboratory.

## 5. Evaluation measures and interpretation

Accuracy alone is insufficient for this task. The 3,072 spreadsheet labels are moderately imbalanced overall, and individual cell lines are much more skewed. For the classifier, we report class-specific precision, recall, F1, macro-F1, balanced accuracy, AUROC and average precision where defined, Brier score, a reliability diagram, and a confusion matrix. Every metric is tied to one manifest, one checkpoint, and one fixed class convention: `good` is the positive class. Results by cell line must display image count **and** independent prefix-group count.

For the workflow, define **false PASS exposure** as `bad images suggested PASS / all bad images evaluated`. This measures how often an actually poor-quality source sample escapes review. Also define **error among PASS** as `bad images suggested PASS / all images suggested PASS`, which answers a different operator question. **PASS coverage** is `PASS suggestions / all evaluated images`. REVIEW and REACQUIRE proportions use the same total-image denominator and are reported separately. A zero denominator produces `not estimable`, not an apparently perfect 0% error.

Resample complete held-out groups to form uncertainty intervals, or use an appropriate cluster-aware method; ordinary image-level bootstrap would understate correlation among images from one acquisition group. Report the number of groups and how many resamples lack one class. Calibration uncertainty and per-cell-line estimates may be broad because the independent group count is limited. Selective policy curves should show coverage versus false PASS exposure across thresholds, with the chosen threshold fixed before test evaluation.

### 5.1 Frozen internal results

| Result | MobileNetV2 + ExtraTrees | Handcrafted logistic | Majority | Evidence |
| --- | ---: | ---: | ---: | --- |
| Train / calibration / test groups | 35 / 12 / 12 | Same | Same | Frozen split artifact |
| Train / calibration / test images | 1,804 / 598 / 670 | Same | Same | `vision_evaluation.json` |
| Test images: good / bad | 357 / 313 | Same | Same | Matching report |
| Accuracy | **0.718** | 0.540 | 0.533 | Test predictions |
| Balanced accuracy | **0.712** | 0.508 | 0.500 | Test predictions |
| Macro-F1 | **0.712** | Not promoted | 0.348 | Confusion matrix |
| AUROC | **0.772** | 0.536 | 0.500 constant score | Test probabilities |
| Brier score | **0.204** | 0.255 | Not separately fitted | Calibrated probabilities |
| Confusion matrix, rows bad/good | `[[193,120],[69,288]]` | `[[5,308],[0,357]]` | All good | Evaluation artifacts |
| 95% group-bootstrap balanced accuracy | **0.510–0.815** | Not computed | 0.500 | 2,000 group resamples |
| 95% group-bootstrap AUROC | **0.552–0.875** | Not computed | 0.500 | 2,000 group resamples |

The selected model's point estimates improve materially over both lower references, but the intervals are wide because only 12 independent prefix groups are in the test partition. Cell-line results are heterogeneous: test balanced accuracy ranged from 0.551 for A549 to 0.848 for CACO; HSAEC had only one bad test example, and NHBE had only one class in test, so their balanced metrics are unstable or undefined. The published MobileNetV3 result of 0.81 accuracy remains absent from the model columns because its evaluation protocol differs.

The planned automated-PASS condition was not met: no calibration threshold accepted at least 20 images while keeping accepted error at or below 10%. Consequently, the current release has no validated automatic PASS coverage or false-PASS guarantee. The interface displays the research score and keeps the final decision with a human; the acquisition-quality REACQUIRE heuristic remains separate and unvalidated.

### 5.2 Figures and case evidence to insert

1. **Data audit:** class and cell-line distributions with group counts and missingness. Caption source and distinguish rows from decoded images.
2. **Split audit:** 59-prefix size distribution and a train/calibration/test balance table. Caption the grouping limitation.
3. **Discrimination and calibration:** confusion matrix, ROC/precision-recall curves if estimable, and reliability diagram; cite checkpoint and manifest hash.
4. **Selective policy:** PASS coverage versus false PASS exposure, with validation-selected threshold and held-out estimate marked separately.
5. **Failure cases:** consent/right-checked examples of false PASS, false REVIEW, and acquisition-quality issues, with image ID, source label, suggestion, and human interpretation.
6. **Interface:** one full screenshot showing image, score provenance, action reason, and model-unavailable behavior. Label any unvalidated heuristic.

## 6. What this system can and cannot establish

The model can learn patterns correlated with one expert's labels in one public data collection. It cannot prove that those labels capture all biological quality dimensions. Expert criteria and inter-rater variation were not present in the spreadsheet audit. A model's probability can be miscalibrated when the acquisition hardware, culture protocol, cell line, or prevalence changes. The current design has no independent prospective laboratory cohort and no measured operator time savings. It cannot distinguish all biologically poor samples from capture errors using the binary source label alone.

The name REACQUIRE is therefore operationally narrow. A corrupt or unreadable file can justify a prompt to get another capture. Blur, underexposure, or clipped highlights may justify review of imaging settings, but only after thresholds are validated against a suitable acquisition-quality reference. A predicted `bad` tissue sample may require a different laboratory action, and the system should route it to a human rather than prescribing re-imaging. Out-of-distribution inputs from other microscopes or cell lines default to review until validated.

The subgroup table may have high variance for HUVEC and NHBE. If a held-out split contains few independent prefixes for one line, that line's numerical score should be accompanied by an instability warning, not a confident claim of fairness or generality. The project does not have patient health information in the source sheet, but future clinical or donor-linked data would require separate privacy and ethics governance.

## 7. Reproducibility and implementation

The repository contains project-authored source, dependency specifications, the scripts `audit_metadata.py`, `prepare_dataset.py`, `train_baseline.py`, `extract_embeddings.py`, `train_vision_model.py`, and `evaluate.py`, a Streamlit demo entry point, data/model cards, and exact commands. The raw data are obtained from Zenodo. Preparation creates an image-to-sheet manifest and feature table; training saves classifiers, calibrators, label mappings, seeds, and frozen group IDs. `vision_evaluation.json` records candidate selection, split counts, held-out metrics, subgroup slices, and complete-group bootstrap intervals.

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

Before release, a fresh environment must reproduce the same group assignments and metrics from the same complete source files. Record the local source SHA-256 values, upstream MD5 checks, Git commit, dependency versions, platform, run date, manifest/checkpoint hashes, and exact evaluation command. Models or assets that require redistribution rights should be downloaded separately or omitted with clear instructions.

## 8. Potential impact and next evidence

If a future model can achieve a useful PASS coverage at a suitably low false PASS exposure, the system could reduce routine review load while preserving attention for ambiguous images. This is a hypothesis, not a measured impact claim. A prospective pilot should randomize or alternate review sessions, record time per image, disagreements with expert consensus, repeat-capture rates, and downstream tissue-model consequences. It should include new acquisition dates and, ideally, an independent microscope or laboratory. Success would mean measurable operator benefit without increasing the rate of poor-quality samples that proceed unreviewed.

The present contribution is a reproducible and falsifiable workflow: explicit source mapping, group-isolated evaluation, calibrated-score intent, action definitions, and honest abstention. A result should only be promoted from prototype evidence to a scientific claim when the complete archive, label join, threshold validation, subgroup uncertainty, and external checks support it.

## 9. Submission evidence ledger

| Claim | Present evidence | Release condition |
| --- | --- | --- |
| 3,072 metadata rows, six cell lines, 59 prefixes | Local spreadsheet audit | Retain audit output and source checksum |
| `1 = good`, `2 = bad` | Two preview examples plus full 3,072-row path consistency check | Preserve matching manifest |
| Trained model performance | Frozen 12-group test: 0.712 balanced accuracy, 0.772 AUROC | External dataset and independent rerun still needed |
| Calibrated safe PASS threshold | Independent calibration design; current demo default is not safety-validated | Validation-set threshold selection and held-out false PASS interval |
| REACQUIRE is appropriate | UI acquisition-quality rule is illustrative | Dedicated acquisition-quality labels or expert review study |
| Public reproducibility | Commands and material list | Public repository plus fresh-install test |
| Competition participation | Kaggle team joined and rules accepted; extra organizer form confirmed received; Writeup saved as draft | Final Writeup submission still required |

## References

1. Movčana, V.; Strods, A.; Narbute, K.; Rūmnieks, F.; Rimša, R.; Mozoļevskis, G.; Ivanovs, M.; Kadiķis, R.; Zviedris, K. G.; Leja, L.; et al. *Organ-On-A-Chip (OOC) Image Dataset for Machine Learning and Tissue Model Evaluation*. **Data** 2024, 9(2), 28. [https://doi.org/10.3390/data9020028](https://doi.org/10.3390/data9020028).
2. Movčana, V. et al. *Organ-on-a-Chip (OOC) Image Dataset*. Zenodo, 2023. [https://doi.org/10.5281/zenodo.10203721](https://doi.org/10.5281/zenodo.10203721).
3. 5th Pazhou Algorithm Competition Organizing Committee. *AI4S Open Innovation: AI for Life Science*. Kaggle, 2026. [Competition description](https://www.kaggle.com/competitions/ai-4-s-open-innovation-artificial-intelligence-for-life-scien/overview/description).
4. Pazhou Algorithm Competition. *Participation Guidelines*. [Official rules](https://www.aicompetition-pz.com/guidelines), accessed 2026-09-28.

## Report production notes (remove from submitted PDF)

This manuscript is structured for a self-contained **15–20-page PDF** after figures and verified results are inserted: title/abstract (1 page); problem and source (2–3); data audit and rights (2); system and method (3–4); grouped protocol (2–3); results and case studies (3–4); limitations, impact, reproducibility, and references (2–3). Keep every figure legible and include source, partition, and model version in captions. The final PDF should remove all bracketed placeholders and this production note. The Kaggle Writeup must link to the final public PDF or include its full technical content directly.
