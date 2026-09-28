# AI4S rubric evidence audit

This is a preparation checklist against the published [AI4S evaluation criteria](https://www.kaggle.com/competitions/ai-4-s-open-innovation-artificial-intelligence-for-life-scien/overview/description), **not** a judge's score or evidence that a submission was made. No numerical self-score is assigned while the core grouped-test results, public repository, and video are unfinished. The official weights total 100%.

| Criterion | Weight | Evidence already available | Evidence required for a defensible claim |
| --- | ---: | --- | --- |
| Problem Importance & Potential Impact | 30% | Real brightfield OoC dataset; explicit QC workflow and human review; source paper describes expert quality labels | Quantify the time/error burden in an actual OoC workflow or clearly label it as a proposed benefit; have a domain expert review action semantics |
| Technical Approach & Innovation | 30% | Date-prefix grouping, image-statistic baseline design, calibration and abstention policy, separate acquisition-quality checks | Working implementation, frozen architecture diagram, ablations and evidence that the three actions add value beyond ordinary binary classification |
| Results & Validation | 20% | Metadata audit of 3,072 rows, six cell lines, 59 groups; evaluation protocol | Complete archive integrity, full label-path join, group-isolated test metrics with intervals, calibration and per-cell-line error analysis |
| Reproducibility & Implementation Quality | 10% | Documented source and intended CLI workflow | Public repository, environment lock, reproducible commands, release artifact hashes, source-rights check, tested demo from fresh setup |
| Presentation Quality | 10% | Report manuscript, Writeup draft, Chinese video script | Recorded ≤5-minute public video showing actual execution and verified results; accessible links and polished screenshots |

## Required package and status

The competition's preliminary submission is a **Kaggle Writeup** with category declaration, public demo video, public code repository, and a self-contained technical report (ideally 15–20 pages). A public interactive demo is optional. The contest page states an additional registration form must be completed before submission; this document does not establish registration, acceptance of terms, or Writeup submission. The current deadline observed by the project audit is **2026-10-10 15:59:59 UTC**; recheck the live Kaggle page before final upload because organizers may change schedules.

| Gate | Current evidence | Release condition |
| --- | --- | --- |
| Source data | Complete ZIP MD5 matched; 3,072/3,072 unique image/path matches decoded | Preserve checksums and matching manifest |
| Model | Architecture and metrics specified | Trained checkpoint and saved validation/test outputs |
| Results | None claimed here | Independently rerun frozen grouped test |
| Video | Script drafted | Actual screen recording, public playback without login |
| Repository | Local working files | Public URL, clean install and run on fresh environment |
| Technical report | Manuscript with result slots | Exact measured values, figures, limitations, citations |
| Registration and Kaggle Writeup | No completion evidence in this repository | User completes identity/terms steps and verifies final posted Writeup |

## Claim-to-evidence discipline

- `3,072 images`, class counts, cell-line counts, and 59 prefixes are supported by both the spreadsheet audit and the complete image/path join; this still does not establish external-lab generalization.
- `1 = good`, `2 = bad` is supported by **two official archive-preview matches** and remains subject to the full join.
- Any **0.81 accuracy** reference is a **published literature figure**, not our accuracy.
- An illustrative PASS/REVIEW/REACQUIRE UI proves that the interface can display a workflow; it does not prove calibrated decisions or reduced laboratory work.
- A recorded video or filled Writeup editor does not prove a public, eligible submission. Use the final public page and required registration confirmation as the evidence.
