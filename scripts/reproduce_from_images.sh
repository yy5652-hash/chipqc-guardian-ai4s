#!/usr/bin/env bash
# Rebuild everything from the raw Zenodo archive: manifest, features, descriptors, model, results, figures, report.
# About 15 minutes on a laptop GPU after the 6.7 GB download. Ablation features other than the released
# representation are shipped in features/ and are not recomputed here.
set -euo pipefail
cd "$(dirname "$0")/.."

python scripts/download_data.py
python scripts/build_manifest.py
python scripts/camera_formats.py
python scripts/extract_features.py --backbone dinov2_vits14
python scripts/acquisition_descriptors.py
# python scripts/extract_ablation_features.py      # optional: the other representations of section 5.3 (about one hour)
python evaluate.py
python scripts/train_final.py
python scripts/train_final.py --exclude-dates 230524 230425 --out models/guardian-v2-demo --atlas ../guardian-v2/atlas
python scripts/build_atlas.py
python scripts/build_examples.py
python scripts/degradation_study.py
python scripts/camera_transfer.py
python scripts/benchmark.py
python evaluate.py
python scripts/make_figures.py --images data/raw/OOC_image_dataset
python scripts/build_web_demo.py
python scripts/check_web_demo.py
python scripts/build_report.py
