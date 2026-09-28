# Data card: OoC brightfield image quality

## Source and scope

The project uses the [Organ-on-a-Chip (OOC) Image Dataset](https://doi.org/10.5281/zenodo.10203721) by Movčana and colleagues. Its [data descriptor](https://doi.org/10.3390/data9020028) describes automated brightfield microscopy of OoC samples and quality judgments assigned by a cell-biology expert. The Zenodo record describes `good` and `bad` image folders under supplied train, validation, and test directories. This project audits the spreadsheet and constructs an independent grouped split for evaluating a workflow, rather than assuming that the upstream directory split prevents related acquisitions from crossing partitions.

Local audit basis: `data/raw/OOC_datasheet.xlsx`, sheet `Main`, seven columns, 3,118 rows after the header of which 3,072 have a non-empty `imageID`. The complete 6.7 GB source archive matched upstream MD5 `8f7e058996203d48eb03b2d86c0a2e4d`; Python ZIP64 validation found no corrupt member, and all 3,072 non-empty spreadsheet IDs matched one decodable image with a consistent path label. Raw images remain excluded from the project repository.

## Observed spreadsheet profile

| Field | Interpretation | Observed completeness |
| --- | --- | ---: |
| `imageID` | Unique string such as `220429_01` | 3,072 / 3,072 |
| `cell type` | Cell-line code | 3,072 / 3,072 |
| `seeding density, cells/ml` | Recorded seeding density; spreadsheet values are formatted strings | 2,728 / 3,072 |
| `time after seeding, h` | Recorded elapsed hours when present | 828 / 3,072 |
| `day` | Day category in source sheet; meaning must be preserved as supplied | 3,072 / 3,072 |
| `Decision 1/2 (good/bad)` | Numeric expert decision; archive-preview examples support `1 = good`, `2 = bad` | 3,072 / 3,072 |
| `flow rate` | Recorded flow rate; text with units | 2,213 / 3,072 |

The missing counts are 344 for density, 2,244 for elapsed hours, and 859 for flow rate. Do not fill them with zero. Missingness may depend on acquisition protocol or cell line and can itself leak collection context; compare image-only, metadata-only, and missingness-aware models on the same held-out groups.

| Cell-line code in sheet | Rows |
| --- | ---: |
| HPMEC | 1,462 |
| A549 | 775 |
| CACO | 346 |
| HSAEC | 244 |
| NHBE | 138 |
| HUVEC | 107 |

The `CACO` code is the spreadsheet spelling; the Zenodo description calls the cell line Caco-2. Preserve the raw code and document any display-name normalization. The numeric labels are `1`: 1,727 rows and `2`: 1,345 rows. Their direction was cross-checked against the official Zenodo archive preview: `test/good/A549/0-1_days/221010_82.png` has spreadsheet label `1`, while `test/bad/A549/4+_days/230529_207.png` has label `2`. The subsequent complete row-to-path audit matched all 3,072 rows with no label conflict, confirming the working mapping `1 = good`, `2 = bad` for this archive.

All 3,072 image IDs are unique and match a six-digit prefix, underscore, and sequence number. There are 59 unique six-digit prefixes, with group sizes from 6 to 229 rows; 45 prefixes contain both numeric labels. We treat the prefix as a conservative acquisition-date-like grouping key. It is an inferred key from filenames, not independently verified chip, donor, experiment, or biological replicate identity.

## Label and image integrity gate

Before model training or a semantic demo:

1. Finish and verify the upstream archive with its Zenodo checksum (`md5:8f7e058996203d48eb03b2d86c0a2e4d`) and verify the spreadsheet checksum (`md5:a3d4875e3da4da83bac45c0ce727cb40`).
2. Create a one-to-one join from every spreadsheet `imageID` to an archive image path. Log unmatched, duplicate, unreadable, and non-image files. Do not resolve ambiguity by taking the first match.
3. Use the path's explicit `good`/`bad` folder to confirm the preview-derived mapping `1 = good`, `2 = bad` across every matched row. Independently inspect representative samples of each class and cell line. Record any disagreement; if unresolved, stop semantic training and report neutral numeric labels only.
4. Detect exact duplicates by cryptographic hash and near duplicates by a declared perceptual-hash or image-similarity procedure. Keep every connected duplicate cluster in one partition, or exclude cross-partition copies and document the count.
5. Preserve the raw manifest and generate a derived manifest with source path, raw label, verified semantic label, prefix group, cell line, missingness indicators, and split. Keep all transformations and exclusions auditable.

## Intended use and limits

Appropriate research use is testing how image models, metadata, calibration, and human review can support quality screening of the source's OoC samples. The data are not a clinical diagnostic cohort, drug response dataset, or proof of tissue function. A classifier trained on expert sample-quality labels cannot separately identify focus errors, lighting artifacts, contamination, or whether taking another photograph will repair the issue. A REACQUIRE suggestion therefore needs an independently defined file/image-usability trigger or a human-confirmed reason.

The six cell lines and acquisition setup limit transfer to other chips, microscopes, media, laboratories, and cell types. No patient-level demographics or private health records are supplied in this spreadsheet. The source paper says the labels were expert judgments; their criteria, disagreement rate, and inter-rater reliability are not available here. Class counts alone do not reveal the prevalence of unacceptable quality in deployment.

## Split and evaluation risks

The upstream folders were reportedly proportionally split by labels, cell lines, and time after seeding. Those properties do not establish independence across near-related images. For this project, assign all images with the same six-digit prefix to one partition and save the manifest before feature fitting. Check class and cell-line coverage in each partition; some strata may be too small for stable metrics. Report group counts alongside image counts. A date-prefix split is conservative but still may leave shared chips or experiments across dates; any stronger provenance found later should replace or augment it.

## Rights, provenance, and citation

The [dataset paper](https://doi.org/10.3390/data9020028) states **“Dataset License: CC-BY-SA”**; the version and the live Zenodo record's full rights text should still be checked before redistributing images or derivatives. The paper's open-access license and this repository's MIT license do not relicense the raw dataset. Check the live source rights and the [competition guidelines](https://www.aicompetition-pz.com/guidelines) before publishing images, image thumbnails, or model weights. The organizer's guidelines limit use of organizer-provided data for contest training and bar commercial use; whether that clause applies to this independently sourced Zenodo copy should be resolved from the applicable terms, not assumed. Link to the source instead of bundling images in a public repository.

Recommended source citation: Movčana, V. et al. *Organ-On-A-Chip (OOC) Image Dataset for Machine Learning and Tissue Model Evaluation*. Data 2024, 9(2), 28. [https://doi.org/10.3390/data9020028](https://doi.org/10.3390/data9020028). Dataset: [https://doi.org/10.5281/zenodo.10203721](https://doi.org/10.5281/zenodo.10203721).
