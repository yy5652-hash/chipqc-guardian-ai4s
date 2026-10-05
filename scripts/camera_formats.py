"""Which camera took each frame? -> features/camera_format.csv and results/image_formats.json

The dataset mixes a grey 2056 x 1542 camera with a colour camera (2048 x 1536, a few frames at 640 x 480).
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from chipqc.data import load_manifest  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--images", type=Path, default=ROOT / "data/raw/OOC_image_dataset")
a = ap.parse_args()
man = load_manifest(ROOT / "data/manifest.csv")
camera, formats = [], collections.Counter()
for path in man.path:
    with Image.open(a.images / path) as im:
        formats[(im.size, im.mode)] += 1
        camera.append("grey 2056x1542" if im.mode == "L" else "colour 640x480" if max(im.size) == 640 else "colour 2048x1536")
pd.DataFrame({"image_id": man.image_id, "camera": camera}).to_csv(ROOT / "features/camera_format.csv", index=False)
(ROOT / "results/image_formats.json").write_text(json.dumps({"image_formats": [{"size": list(k[0]), "mode": k[1], "frames": v} for k, v in formats.most_common()]}, indent=1))
print(dict(collections.Counter(camera)))
