"""Fine-tune MobileNetV2 on frozen group splits and evaluate the selected epoch once."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import random

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, confusion_matrix, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]


def metrics(labels: np.ndarray, probability: np.ndarray, threshold: float) -> dict:
    prediction = (probability >= threshold).astype(int)
    return {
        "n_images": int(len(labels)),
        "accuracy": float(accuracy_score(labels, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "roc_auc": float(roc_auc_score(labels, probability)),
        "brier_score": float(brier_score_loss(labels, probability)),
        "confusion_matrix_rows_bad_good": confusion_matrix(labels, prediction, labels=[0, 1]).tolist(),
        "decision_threshold": float(threshold),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/processed/manifest.csv")
    parser.add_argument("--image-root", type=Path, default=ROOT / "data/raw/images")
    parser.add_argument("--split-artifact", type=Path, default=ROOT / "artifacts/baseline.joblib")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/finetuned_mobilenet_v2.pt")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/finetuned_evaluation.json")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    manifest = pd.read_csv(args.manifest, dtype={"image_id": str, "group_id": str, "cell_type": str})
    manifest = manifest.loc[manifest["status"] == "matched"].reset_index(drop=True)
    split_ids = joblib.load(args.split_artifact)["split_image_ids"]
    split_frames = {
        name: manifest.loc[manifest["image_id"].isin(ids)].reset_index(drop=True)
        for name, ids in split_ids.items()
    }

    weights = MobileNet_V2_Weights.IMAGENET1K_V2
    mean, std = weights.transforms().mean, weights.transforms().std
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.82, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(12),
        transforms.ColorJitter(brightness=0.12, contrast=0.12),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    eval_transform = weights.transforms()

    class OoCDataset(Dataset):
        def __init__(self, frame: pd.DataFrame, transform: object):
            self.frame = frame
            self.transform = transform

        def __len__(self) -> int:
            return len(self.frame)

        def __getitem__(self, index: int):
            row = self.frame.iloc[index]
            with Image.open(args.image_root / row["member"]) as source:
                image = self.transform(source.convert("RGB"))
            return image, int(row["label_good"]), str(row["image_id"])

    loaders = {
        name: DataLoader(
            OoCDataset(frame, train_transform if name == "train" else eval_transform),
            batch_size=args.batch_size,
            shuffle=(name == "train"),
            num_workers=0,
        )
        for name, frame in split_frames.items()
    }
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = mobilenet_v2(weights=weights)
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    for block in model.features[14:]:
        for parameter in block.parameters():
            parameter.requires_grad = True
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    model.to(device)
    train_labels = split_frames["train"]["label_good"].to_numpy(dtype=int)
    counts = np.bincount(train_labels, minlength=2)
    class_weights = torch.tensor(len(train_labels) / (2 * counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=1e-4,
    )

    def predict(loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        model.eval()
        labels, probabilities = [], []
        with torch.inference_mode():
            for images, batch_labels, _ in loader:
                logits = model(images.to(device))
                probabilities.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy().tolist())
                labels.extend(batch_labels.numpy().tolist())
        return np.asarray(labels, dtype=int), np.asarray(probabilities, dtype=float)

    history: list[dict] = []
    best: dict | None = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for images, labels, _ in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        labels, probability = predict(loaders["calibration"])
        thresholds = np.linspace(0.1, 0.9, 161)
        balanced = [balanced_accuracy_score(labels, probability >= threshold) for threshold in thresholds]
        threshold = float(thresholds[int(np.argmax(balanced))])
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            **metrics(labels, probability, threshold),
        }
        row["selection_score"] = row["balanced_accuracy"] + row["roc_auc"]
        history.append(row)
        print(json.dumps(row), flush=True)
        if best is None or row["selection_score"] > best["selection_score"]:
            best = {**row, "state_dict": deepcopy(model.state_dict())}

    assert best is not None
    model.load_state_dict(best.pop("state_dict"))
    test_labels, test_probability = predict(loaders["test"])
    test = metrics(test_labels, test_probability, float(best["decision_threshold"]))
    payload = {
        "kind": "finetuned_mobilenet_v2",
        "state_dict": model.cpu().state_dict(),
        "decision_threshold": float(best["decision_threshold"]),
        "label_mapping": {"0": "bad", "1": "good"},
        "seed": args.seed,
        "weights": "MobileNet_V2_Weights.IMAGENET1K_V2",
    }
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "internal group-held-out evaluation; not external or clinical validation",
        "device": str(device),
        "split_counts": {
            name: {"images": len(frame), "groups": int(frame["group_id"].nunique())}
            for name, frame in split_frames.items()
        },
        "history": history,
        "selected_epoch": best,
        "test": test,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("TEST " + json.dumps(test), flush=True)


if __name__ == "__main__":
    main()
