# Model card: ChipQC Guardian

## Model status

This card describes a working internal research model, not a production or clinical model. The selected frozen ImageNet MobileNetV2 representation plus calibrated ExtraTrees classifier has group-held-out internal results, but no external laboratory validation and no validated automatic PASS policy.

## Task and inputs

The supervised task is to estimate the source expert's binary **OoC sample-quality label** from a brightfield image. Cell line, day, elapsed hours, seeding density, and flow rate are available for context and slice analysis; they are not default predictor inputs because they may encode acquisition batch and missingness shortcuts. The system must handle missing metadata explicitly. Two official archive-preview examples and the complete 3,072-row image-ID-to-folder audit support `1 = good`, `2 = bad`.

The input population is the six cell lines in the source dataset: A549, CACO/Caco-2, HPMEC, HUVEC, NHBE, and HSAEC. The source setup is automated brightfield microscopy. A user-supplied image from a different microscope or cell line is out of scope unless external validation supports it.

## Output and human decision policy

The model's primary output is a probability for the verified acceptable-quality class, with a calibrated confidence statement and provenance of the calibration set. The interface translates this to an operational suggestion:

| Suggestion | Required evidence and human action |
| --- | --- |
| PASS | Image is readable and the validated acceptance threshold is met. Show score and allow the reviewer to override. |
| REVIEW | Prediction is near the decision boundary, metadata/context is outside validated support, or calibration is unreliable. Route to a biologist or trained reviewer. |
| REACQUIRE | A separately defined file/image-usability check fails, or a reviewer confirms that new imaging is appropriate. Explain the trigger. A low biological quality score alone does not show that re-imaging will help. |

The thresholds should be selected on grouped validation data to control false PASS decisions, then frozen for final grouped-test evaluation. Report what fraction of cases receive each suggestion and the error rate within each action. If the test evidence does not support a safe PASS band, the system can route all cases to REVIEW rather than manufacture confidence.

## Intended users and prohibited interpretations

Intended users are OoC researchers reviewing images as part of a quality-control workflow. The system is a triage aid, not an autonomous acceptance gate. It does not diagnose disease, assess patient outcomes, infer drug efficacy/toxicity, or establish tissue viability. A PASS means consistency with the source expert's image-quality label under evaluated conditions, not guaranteed biological validity. A REACQUIRE suggestion is not a directive to discard a culture or repeat an experiment without human assessment.

## Training and evaluation requirements

Use `imageID`'s six-digit prefix as the minimum group boundary: all images in a prefix remain in one of train, validation, or test. Freeze the split, fitted transforms, model-selection rule, calibration method, and thresholds before opening held-out results. Training can compare a class-prior baseline, metadata-only baseline, image-only baseline, and any fusion model using identical groups. Preprocessing and imputation fit on train only. Calibration fits on validation predictions and is evaluated on test; if validation data are used for threshold selection, say so.

Required report fields include group and image counts per split; per-class precision/recall; balanced accuracy; AUROC/AUPRC if both classes are present; Brier score and calibration plot; confusion matrix; PASS false-accept rate; REVIEW coverage; subgroup results by cell line; and confidence intervals that resample whole groups. Any cell line with too few independent groups should be shown as unstable rather than assigned a strong conclusion. Compare against the same metrics on simple baselines and include a failure-case gallery with image-use rights checked.

## Known risks and mitigations

- **Label semantics:** The spreadsheet does not specify which numeric code is `good`. A wrong mapping reverses decisions. Resolve against archive paths before semantic outputs.
- **Group leakage:** Related images can be visually similar. Partition by acquisition-date-like prefix, then inspect exact and near duplicates and stronger provenance if available.
- **Missingness and context shortcuts:** Metadata completeness may identify collection conditions rather than morphology. Audit missingness by class/cell line and include image-only and missingness-only comparisons.
- **Calibration shift:** A probability calibrated on one microscope, experiment, or cell line may fail elsewhere. Route unsupported inputs to REVIEW and test by subgroup.
- **Action mismatch:** The dataset labels sample quality, not capture failure or treatment choice. Keep the REACQUIRE trigger separate and provide human override.
- **Hidden stratification:** Six cell lines have unequal counts. Overall metrics can conceal poor performance on the smaller NHBE and HUVEC groups.

## Release record to complete

| Item | Required value before performance claims |
| --- | --- |
| Code revision and environment | Local source package and dependency files prepared; public Git revision pending |
| Source archive and sheet checksums | Archive MD5 `8f7e058996203d48eb03b2d86c0a2e4d`; sheet audit saved |
| Numeric-to-semantic label mapping evidence | Two examples plus full 3,072-row path audit |
| Frozen split manifest and group counts | 35 / 12 / 12 train/calibration/test groups |
| Training seed and model checkpoint hash | Seed 42; public checkpoint hash pending release packaging |
| Calibration method and fitted parameters | Logistic calibration on 12 groups; decision threshold 0.62 |
| PASS/REVIEW thresholds and REACQUIRE rule | No threshold met planned safe-PASS target; human decision retained |
| Held-out metrics with uncertainty | Balanced accuracy 0.712 (group-bootstrap 95% 0.510–0.815); AUROC 0.772 (0.552–0.875) |
| External laboratory or microscope validation | Not available |

Source data: [Zenodo 10.5281/zenodo.10203721](https://doi.org/10.5281/zenodo.10203721). Dataset paper: [Movčana et al., Data 2024](https://doi.org/10.3390/data9020028). See [DATA_CARD.md](DATA_CARD.md) and [reports/experiment_protocol.md](reports/experiment_protocol.md).
