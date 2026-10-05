# Data card

## Source

**Organ-on-a-Chip (OOC) Image Dataset**, Movčana V., Strods A., Narbute K., Rūmnieks F., Rimša R., Mozolevskis G., Kadiķis R., Ivanovs M. Zenodo, 2023, [doi:10.5281/zenodo.10203721](https://doi.org/10.5281/zenodo.10203721), licence **CC BY 4.0**. Described in *Data* 9(2), 28 (2024), [doi:10.3390/data9020028](https://doi.org/10.3390/data9020028).

Brightfield images from an automated microscope on an organ-on-a-chip setup, six human cell lines, with a class label ("good" or "bad" sample quality as assessed by a biology expert), cell type, time after seeding and, for some images, seeding density and flow rate. The images show cultured cell lines; there is no personal, patient or clinical information.

## What we checked

- The 6.7 GB archive matches its published MD5 (`8f7e058996203d48eb03b2d86c0a2e4d`).
- All 3,072 spreadsheet rows match exactly one decodable image; no image ID occurs twice.
- The folder label of every image (`good` / `bad`) agrees with the spreadsheet label (`1` / `2`).

`scripts/download_data.py` and `scripts/build_manifest.py` repeat these checks and write `data/manifest.csv`.

## Composition

| Cell line | Frames | Good | Bad | Acquisition dates |
|---|---|---|---|---|
| HPMEC | 1,462 | 798 | 664 | 29 |
| A549 | 775 | 537 | 238 | 24 |
| Caco-2 | 346 | 109 | 237 | 17 |
| HSAEC | 244 | 163 | 81 | 21 |
| NHBE | 138 | 105 | 33 | 6 |
| HUVEC | 107 | 15 | 92 | 4 |
| Total | 3,072 | 1,727 | 1,345 | 59 |

Formats: 2,048 frames at 2056 × 1542 px, greyscale; 934 at 2048 × 1536 px, colour; 87 at 640 × 480 px, colour; 3 at 1536 × 2048 px, colour. Metadata coverage: time after seeding in hours is missing for 2,244 frames (the day of culture is always present), seeding density for 344, flow rate for 859.

## Structure that matters for evaluation

- **Acquisition date.** The first six characters of an image ID (`YYMMDD`). Frames of one date are neighbouring fields of the same chips: neighbouring frames share a label 85 % of the time, and 14 of the 59 dates contain a single class. All our evaluations hold out whole dates. The date is a proxy for an acquisition session, not a verified chip or donor identifier.
- **The authors' folders.** The archive's `train` / `val` / `test` folders split images at random within dates: all 57 dates in `test` also occur in training. We use that split only to compare with the published baseline.
- **Two cameras.** Grey and colour frames come from different cameras and have different label proportions (grey 46 % good, colour 76 % good). We report performance per camera and across cameras.
- **What "bad" contains.** Both poor cultures and unusable images (motion smear, blacked-out fields) carry the label "bad".

## What this repository redistributes

Derived material only, with attribution: `data/manifest.csv` (one row per frame, from the authors' spreadsheet), embeddings of every frame (`features/`), 144 px grey thumbnails of the reference frames (`models/guardian-v2/atlas/`) and nine example frames resized to 1344 × 1008 px (`examples/`). The full-resolution images are not redistributed; download them from Zenodo.
