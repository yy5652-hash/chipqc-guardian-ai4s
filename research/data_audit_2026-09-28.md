# OOC metadata audit — 2026-09-28

Source: `data/raw/OOC_datasheet.xlsx`, sheet `Main`.

## Observed facts

- DataFrame shape after reading the sheet: 3,072 rows × 7 columns.
- Non-empty `imageID`: 3,072.
- Duplicate `imageID`: 0.
- Exact duplicate rows: 0.
- Label values in `Decision 1/2 (good/bad)`: `1 = good` (1,727 rows); `2 = bad` (1,345 rows).
- Mapping evidence: the official Zenodo archive preview places `221010_82.png` under `test/good/...`; the spreadsheet records label `1` for that image. It places `230529_207.png` under `test/bad/...`; the spreadsheet records label `2`. This mapping was therefore cross-checked against the official archive structure rather than inferred only from the spreadsheet header.
- Six cell-line values:

| Cell line | Label 1 | Label 2 | Total |
|---|---:|---:|---:|
| A549 | 537 | 238 | 775 |
| CACO | 109 | 237 | 346 |
| HPMEC | 798 | 664 | 1,462 |
| HSAEC | 163 | 81 | 244 |
| HUVEC | 15 | 92 | 107 |
| NHBE | 105 | 33 | 138 |

- Missing values: seeding density 344; time after seeding 2,244; flow rate 859.
- `imageID` has 59 distinct six-digit date-like prefixes.
- Several acquisition prefixes are almost or completely label-homogeneous, indicating material batch/label confounding risk. Examples observed in the metadata include prefix `230405` (138 rows, all label 2), `230119` (70 rows, all label 1), `230411` (1 label 1, 78 label 2), and `230419` (8 label 1, 116 label 2).
- Flow-rate and density fields contain formatting variants (spacing, capitalization, scientific notation, commas); normalization must retain the raw source values.

## Consequences for evaluation

1. The primary train/validation/test protocol must group by the six-digit acquisition prefix; a random image split is not sufficient evidence of cross-experiment generalization.
2. Cell line is used for slice analysis and leave-one-cell-line-out evaluation, not as a default predictor input.
3. Missing metadata are reported, not silently imputed into biological meaning.
4. Metrics must include macro-F1 and balanced accuracy alongside accuracy because both overall and cell-line-specific class balance vary.
5. The original archive split can be reported only as a literature-comparable secondary protocol.
