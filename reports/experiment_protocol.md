# ChipQC Guardian experiment protocol

This protocol defines how project performance may be measured. Planned analyses below are not observed results. Publish a metric only after the complete source archive, exact run configuration, saved split manifest, and model artifact can be checked together.

## 1. Freeze the question and data

**Primary question:** On acquisition-date-like groups not used for training, how well can pixel-derived features predict the source expert's `good` versus `bad` OoC sample-quality label, and how much work can a calibrated, human-reviewed triage policy safely defer?

**Source:** [Zenodo 10.5281/zenodo.10203721](https://doi.org/10.5281/zenodo.10203721), described in [Movčana et al., 2024](https://doi.org/10.3390/data9020028). Use the complete original archive and spreadsheet. Record download date, source URL, byte counts, SHA-256 of both local files, and upstream MD5 values. The Zenodo-listed MD5 values are `a3d4875e3da4da83bac45c0ce727cb40` for the sheet and `8f7e058996203d48eb03b2d86c0a2e4d` for the image archive. A ZIP central-directory failure or checksum mismatch blocks training.

The spreadsheet audit found 3,072 unique image IDs, six cell lines, 59 six-digit prefixes, and label counts `1 = 1,727` and `2 = 1,345`. Official archive-preview examples map `1 = good` and `2 = bad`: `221010_82.png` appears under `test/good/...` and `230529_207.png` under `test/bad/...`. Treat this as a sample-verified mapping until the full join proves every matched path agrees.

## 2. Data audit before splitting

The preparation script should emit an immutable row-level manifest with `imageID`, source path inside the ZIP, raw numeric label, folder-derived word label, cell line, prefix, source metadata, missingness flags, image dimensions, decode status, and exclusion reason. Assert that every included ID matches exactly one image, every included path matches exactly one metadata row, and every label matches the path. Report all failures rather than discarding them silently. Keep the source sheet values unmodified in the audit output.

Check exact image SHA-256 duplicates. For near duplicates, specify the perceptual hash and distance threshold before looking at test labels; review cross-split candidates manually. If image paths encode the upstream `train`, `val`, or `test` split, retain that as provenance only. It is not a predictor feature or the primary split.

The grouping prefix is the six digits before the underscore in `imageID`. Verify all IDs fit `^[0-9]{6}_[0-9]+$`; the current metadata audit found no exceptions. It is a proxy for acquisition date, not proof of unique experiment or chip identity. If chip/run identifiers become available, use the strongest shared-identity key and document the change.

## 3. Frozen partitions and leakage barriers

Build one train, one validation, and one held-out test partition using **whole prefix groups**. Choose the seed and group-allocation algorithm before any image-model tuning. Record the ordered group IDs and image IDs for all partitions, class counts, cell-line counts, and a hash of the manifest. Check three disjointness conditions: no image ID overlap, no group overlap, and no exact/declared near-duplicate cross-over. Because 59 groups vary from 6 to 229 images and can have very different class balance, report both group and image counts; nominal percentages alone are misleading.

Fit every preprocessing parameter, imputer, scaler, feature selector, and model using the training partition only. Use validation for model choice, probability calibration, and action thresholds. Open held-out test results once for final evaluation. If a change is made after seeing test performance, identify it as exploratory and evaluate the final frozen method on a new untouched split or external data before making a confirmatory claim.

## 4. Models and ablations

The first reproducible model is a lightweight baseline using deterministic pixel statistics such as brightness, contrast, entropy, gradient, edge, and focus-related descriptors. Record feature names and exact extraction code version. Do not call hand-designed image features a deep neural network. The following comparisons should use the same frozen group split:

| Comparator | Purpose | Status field |
| --- | --- | --- |
| Majority-class and/or prior-only predictor | Establish a trivial reference | Pending run |
| Image-feature baseline | Test the actual image signal | Pending run |
| Metadata-only predictor | Detect collection-context shortcuts | Pending run |
| Missingness-only predictor | Test whether absent fields encode labels | Pending run |
| Image plus metadata | Measure any incremental benefit, if implemented | Optional, pending |
| Label permutation within the training protocol | Negative control for leakage | Optional, pending |

Cell line is a required **evaluation slice**, not a default model input. If a later fusion model uses it, say so clearly and compare it to image-only results. Run leave-one-cell-line-out experiments as exploratory domain-shift checks if each training/test configuration has enough groups and both classes. Do not average undefined metrics from single-class test slices.

## 5. Calibration and three-action policy

The classifier estimates `P(good | image, validated context)`. Record whether its output is an uncalibrated score or a probability calibrated on validation data. Compare reliability plots, Brier score, and expected calibration error with a declared binning rule. Calibration and action thresholds must never be fit on held-out test labels.

The **PASS** rule requires a readable image, in-scope conditions, and `P(good)` at or above a validation-selected threshold. **REVIEW** covers uncertain scores, unsupported cell line or acquisition context, or insufficient calibration evidence. **REACQUIRE** is reserved for a separate image-usability failure (for example, unreadable file or a documented acquisition-quality criterion) or a human-confirmed request to capture again. The dataset's biological `bad` label does not itself establish that re-imaging is useful. If an illustrative heuristic is shown, label it as an unvalidated demo rule and exclude it from claims about model sensitivity.

For each frozen policy, report:

- PASS coverage = number suggested PASS / number evaluated;
- false PASS exposure = number of truly `bad` images suggested PASS / number of truly `bad` images;
- error among PASS = number of truly `bad` images suggested PASS / number suggested PASS;
- REVIEW rate and REACQUIRE rate, each with explicit denominators;
- class-conditional coverage by cell line, with group counts.

If a denominator is zero, show `not estimable`, never zero. State the operating objective used to set the threshold (for example, a predeclared upper bound on false PASS exposure) and whether validation group counts were adequate to support it. If not adequate, restrict outputs to REVIEW and avoid an unsupported safety claim.

## 6. Primary and secondary metrics

Primary classifier reporting: confusion matrix with `good` as positive class; class-specific precision, recall, and F1; macro-F1; balanced accuracy; AUROC and AUPRC when both classes are represented; and Brier score. Also report ordinary accuracy to compare with prior literature, but do not make it the sole metric because class mix differs across cell lines and groups. Every table should give numerator/denominator or sample count plus the split manifest hash.

Primary workflow reporting: PASS coverage and false PASS exposure, plus error among PASS. To estimate uncertainty, resample whole held-out prefix groups with replacement, not individual images. Declare the bootstrap replicate count and confidence-interval method; if a replicate lacks a class, omit undefined class-specific metrics from that interval and report how often this occurs. Show performance by cell line with independent group counts and note where intervals are too wide to interpret.

Secondary stress checks: compare performance when metadata are missing, on high/low brightness or focus-related feature strata, by day category, and under simple image perturbations that mimic plausible acquisition variability. Do not use perturbation tests to claim real-world external validation.

## 7. Literature comparison

The [Movčana et al. data descriptor](https://doi.org/10.3390/data9020028) reports a MobileNetV3 test result of **0.81 accuracy, 0.79 precision, and 0.78 recall** on its own evaluation protocol, versus a stated naive-majority accuracy of 0.56. Those figures belong to the published paper, not this project. A direct score comparison requires matching image sets, labels, and split protocols; a grouped date-prefix test is deliberately different from the paper's split. Label any side-by-side table `different protocol / descriptive only` unless a method is reproduced on identical partitions.

## 8. Reproducibility record and release gate

Archive the following with each experiment: exact command, UTC run time, code commit, dependency lock or environment export, OS/CPU/GPU, source file checksums, image manifest hash, split manifest, seed, feature list, hyperparameters, serialized model hash, calibration parameters, thresholds, predictions for every evaluated image, metrics JSON, and plots. Keep raw images out of public source control unless redistribution rights are confirmed; link to Zenodo with download instructions.

Canonical command sequence after environment installation:

```bash
python scripts/audit_metadata.py
python scripts/prepare_dataset.py
python scripts/train_baseline.py
python scripts/evaluate.py
```

Before copying numbers into the technical report or video, independently rerun `evaluate.py` against the frozen checkpoint and manifest, reconcile its predictions with the metrics table, and inspect at least one error case per supported cell line. The report must distinguish `not run`, `run but unverified`, `verified locally`, and `externally validated`. Only the last two can support a numerical project result, and even then the scope of validation must be stated.
