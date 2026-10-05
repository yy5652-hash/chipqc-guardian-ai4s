# Model card: ChipQC Guardian v2

## What it is

A quality gate for brightfield frames of organ-on-a-chip cultures. Input: one frame. Output: `PASS`, `REVIEW` or `REACQUIRE`, a calibrated probability that experts would call the culture good, an evidence map and five acquisition descriptors.

| Part | Implementation |
|---|---|
| Acquisition gate | five descriptors on the grey frame at 1024 × 768 px; limits stored in `model.json` |
| Representation | frame resized to 1344 × 1008 px, 3 × 3 tiles of 448 × 336 px, frozen DINOv2 ViT-S/14 with registers; mean of 6,912 patch descriptors (384 numbers) |
| Classifier | L2-regularised logistic regression (C = {{ M.training.C }}) on the standardised mean descriptor |
| Calibration | two-parameter logistic map fitted on out-of-fold log-odds |
| Decision | pass threshold per target: the one-sided 90 % Wilson bound of the share of bad frames among passed training-date frames must not exceed the target |

Fitted on {{ int(M.training.frames) }} frames ({{ int(M.training.good) }} good, {{ int(M.training.bad) }} bad) from {{ M.training.dates }} acquisition dates of the Organ-on-a-Chip Image Dataset (Movčana et al., Zenodo 10.5281/zenodo.10203721, CC BY 4.0).

## Intended use

Triage of brightfield frames in a laboratory that images OoC cultures: pass clear frames, send unusable frames back to the microscope, and order the rest for a person. The person keeps the decision on every frame that is not passed.

Not intended for: clinical or regulatory decisions; judging viability, barrier function, toxicity or drug response; discarding cultures automatically; use on a new instrument, chip design or tissue without local reference frames.

## Performance on acquisition dates the model never saw

| Measure | Value |
|---|---|
| AUROC | {{ f3(H.auroc.value) }} (95 % interval {{ ci(H.auroc) }}) |
| Balanced accuracy at 0.5 | {{ f3(H.balanced_accuracy.value) }} |
| Expected calibration error | {{ f3(H.ece_calibrated) }} |
| By cell line (AUROC) | {{ line_aurocs }} |
| By camera (AUROC) | grey {{ f2(CT.within_camera_dates_held_out.grey.auroc) }}, colour {{ f2(CT.within_camera_dates_held_out.colour.auroc) }} |

| Target | Passed automatically | Bad among passed | Re-acquire | Left for a person |
|---|---|---|---|---|
{{ outcome_rows() }}

The 5 % row is shown for completeness; that target is not met and is not offered in the console.

## Known failure modes

- **A different camera.** Trained on one camera and applied to the other without local labels: AUROC {{ f2(CTX.main.grey_to_colour) }} (grey to colour) and {{ f2(CTX.main.colour_to_grey) }} (colour to grey).
- **Low scores are not reliable enough to reject.** At a 10 % target an automatic FAIL would have discarded a set in which {{ pct(S10.automatic_fail_if_it_existed.good_among_failed) }} of frames were good.
- **A549** is the hardest cell line (AUROC {{ f2(U.A549.auroc_line_seen_dates_held_out) }}).
- **Defocus rule.** It catches controlled defocus but also fires on sparse, correctly focused cultures.
- **Evidence outside the culture.** Channel walls and the chip body receive non-zero evidence in some frames, and tile borders are visible in the map.

## Files

`model.json`: classifier weights per descriptor dimension (`cell_weights`, `cell_bias`), calibrator (`platt`), thresholds per target, acquisition limits, standardisation constants, training summary and the evaluation summary above. `reference.npz`: standardised embeddings, labels and cell lines of the reference frames. `atlas/`: 144 px thumbnails of the reference frames. No pickles.

## Licences

Code MIT. Backbone weights Apache-2.0 (Meta AI, via `timm`). Reference data CC BY 4.0.
