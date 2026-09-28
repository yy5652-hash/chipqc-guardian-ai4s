"""Extract deterministic ImageNet MobileNetV2 embeddings for matched images."""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chipqc.dataset import ImageIndex  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=ROOT / "data/raw/OOC_image_dataset.zip")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/processed/manifest.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/mobilenet_v2_embeddings.npz")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    import torch
    from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    manifest = pd.read_csv(
        args.manifest, dtype={"image_id": str, "group_id": str, "cell_type": str}
    )
    manifest = manifest.loc[manifest["status"] == "matched"].reset_index(drop=True)
    if manifest.empty or manifest["image_id"].duplicated().any():
        parser.error("Manifest must contain unique matched images")

    weights = MobileNet_V2_Weights.IMAGENET1K_V2
    transform = weights.transforms()
    backbone = mobilenet_v2(weights=weights)
    backbone.classifier = torch.nn.Identity()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    backbone.eval().to(device)
    image_index = ImageIndex(args.images)

    batches: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(manifest), args.batch_size):
            part = manifest.iloc[start:start + args.batch_size]
            tensors = []
            for row in part.itertuples(index=False):
                with Image.open(BytesIO(image_index.read(row.member))) as source:
                    tensors.append(transform(source.convert("RGB")))
            matrix = torch.stack(tensors).to(device)
            batches.append(backbone(matrix).detach().cpu().numpy().astype(np.float32))
            done = min(start + args.batch_size, len(manifest))
            if done == len(manifest) or done % (args.batch_size * 10) == 0:
                print(f"Embedded {done}/{len(manifest)} images", flush=True)

    embeddings = np.concatenate(batches, axis=0)
    if embeddings.shape != (len(manifest), 1280) or not np.isfinite(embeddings).all():
        raise RuntimeError(f"Invalid embedding matrix: {embeddings.shape}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        embeddings=embeddings,
        image_id=manifest["image_id"].to_numpy(dtype=str),
        group_id=manifest["group_id"].to_numpy(dtype=str),
        cell_type=manifest["cell_type"].to_numpy(dtype=str),
        label_good=manifest["label_good"].to_numpy(dtype=np.int8),
        member=manifest["member"].to_numpy(dtype=str),
        backbone=np.array("torchvision MobileNetV2 IMAGENET1K_V2"),
    )
    print(f"Saved {embeddings.shape} embeddings to {args.output}")


if __name__ == "__main__":
    main()
