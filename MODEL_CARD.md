# Model card: ChipQC Guardian

## What it is

A quality gate for brightfield frames of organ-on-a-chip cultures. Input: one frame. Output: `PASS`, `REVIEW` or `REACQUIRE`, a calibrated probability that experts would call the culture good, an evidence map and five acquisition descriptors.

| Part | Implementation |
|---|---|
| Acquisition gate | five descriptors on the grey frame at 1024 × 768 px; limits stored in `model.json` |
| Representation | frame resized to 1792 × 1344 px, 4 × 4 tiles of 448 × 336 px, frozen DINOv2 ViT-S/14 with registers; each patch described by its tokens after the last four blocks; mean of 12,288 patch descriptors (1,536 numbers) |
| Classifier | L2-regularised logistic regression (C = 0.01) on the standardised mean descriptor |
| Calibration | two-parameter logistic map fitted on out-of-fold log-odds |
| Decision | pass threshold per target: the one-sided 90 % Wilson bound of the share of bad frames among passed training-date frames must not exceed the target |

Fitted on 3,072 frames (1,727 good, 1,345 bad) from 59 acquisition dates of the Organ-on-a-Chip Image Dataset (Movčana et al., Zenodo 10.5281/zenodo.10203721, CC BY 4.0).

## Intended use

Triage of brightfield frames in a laboratory that images OoC cultures: pass clear frames, send unusable frames back to the microscope, and order the rest for a person. The person keeps the decision on every frame that is not passed.

Not intended for: clinical or regulatory decisions; judging viability, barrier function, toxicity or drug response; discarding cultures automatically; use on a new instrument, chip design or tissue without local reference frames.

## Performance on acquisition dates the model never saw

| Measure | Value |
|---|---|
| AUROC | 0.872 (95 % interval 0.842–0.900) |
| Balanced accuracy at 0.5 | 0.793 |
| Expected calibration error | 0.024 |
| By cell line (AUROC) | A549 0.79, CACO 0.91, HPMEC 0.87, HSAEC 0.87, HUVEC 0.99, NHBE 0.98 |
| By camera (AUROC) | grey 0.87, colour 0.82 |

| Target | Passed automatically | Bad among passed | Re-acquire | Left for a person |
|---|---|---|---|---|
| 5 % | 8 % | 5.2 % | 15 % | 77 % |
| 10 % | 26 % | 9.6 % | 15 % | 59 % |
| 15 % | 39 % | 13.4 % | 15 % | 45 % |
| 20 % | 51 % | 17.5 % | 15 % | 34 % |

The 5 % row is shown for completeness; that target is not met and is not offered in the console.

## Known failure modes

- **A different camera.** Trained on one camera and applied to the other without local labels: AUROC 0.72 (grey to colour) and 0.75 (colour to grey).
- **Low scores are not reliable enough to reject.** At a 10 % target an automatic FAIL would have discarded a set in which 13 % of frames were good.
- **A549** is the hardest cell line (AUROC 0.79).
- **Defocus rule.** It catches controlled defocus but also fires on sparse, correctly focused cultures.
- **Evidence outside the culture.** Channel walls and the chip body receive non-zero evidence in some frames, and tile borders are visible in the map.

## Files

`model.json`: classifier weights per descriptor dimension (`cell_weights`, `cell_bias`), calibrator (`platt`), thresholds per target, acquisition limits, standardisation constants, training summary and the evaluation summary above. `reference.npz`: standardised embeddings, labels and cell lines of the reference frames. `atlas/`: 144 px thumbnails of the reference frames. No pickles.

## Licences

Code MIT. Backbone weights Apache-2.0 (Meta AI, via `timm`). Reference data CC BY 4.0.
