"""Optional, explicitly labeled integration for a local baseline artifact."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .quality import QualityMetrics


@dataclass(frozen=True)
class ModelAvailability:
    estimator: Any | None
    feature_names: tuple[str, ...]
    label_map: dict[str, str]
    message: str
    kind: str = "none"
    confidence_threshold: float | None = None

    @property
    def available(self) -> bool:
        return self.estimator is not None


@dataclass(frozen=True)
class ModelPrediction:
    label: str
    confidence: float | None
    probability_good: float | None = None


def load_baseline_model(project_root: Path) -> ModelAvailability:
    vision_artifact = project_root / "artifacts" / "vision_model.joblib"
    vision_report = project_root / "reports" / "vision_evaluation.json"
    if vision_artifact.is_file() and vision_report.is_file():
        try:
            import joblib

            loaded = joblib.load(vision_artifact)
            required = {"kind", "model", "calibrator", "decision_threshold", "label_mapping"}
            if not isinstance(loaded, dict) or not required <= loaded.keys():
                raise ValueError("视觉模型 artifact 缺少必需字段")
            if loaded["kind"] != "mobilenet_v2_embedding_extratrees":
                raise ValueError(f"不支持的视觉模型类型：{loaded['kind']}")
            report = json.loads(vision_report.read_text(encoding="utf-8"))
            test = report.get("test", {})
            message = (
                "已加载 MobileNetV2 视觉表征模型。内部按采集前缀分组的独立测试："
                f"balanced accuracy {float(test['balanced_accuracy']):.3f}，"
                f"AUROC {float(test['roc_auc']):.3f}；这不是外部或临床验证。"
            )
            return ModelAvailability(
                loaded,
                (),
                {str(key): str(value) for key, value in loaded["label_mapping"].items()},
                message,
                "mobilenet_v2_embedding",
                None,
            )
        except Exception as exc:
            return ModelAvailability(None, (), {}, f"视觉模型 artifact 无法加载：{exc}")

    artifact = project_root / "artifacts" / "baseline.joblib"
    metadata_path = project_root / "artifacts" / "metadata.json"
    if not artifact.is_file():
        return ModelAvailability(None, (), {}, "未发现 artifacts/baseline.joblib；模型未训练或尚未接入。")
    if not metadata_path.is_file():
        return ModelAvailability(None, (), {}, "发现模型文件，但缺少 artifacts/metadata.json 特征契约。")

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        names = (
            metadata.get("feature_names")
            or metadata.get("feature_columns")
            or metadata.get("features")
        )
        if not isinstance(names, list) or not names or not all(isinstance(name, str) for name in names):
            raise ValueError("metadata.json 缺少非空 feature_names 数组")
        mapping = (
            metadata.get("label_mapping")
            or metadata.get("label_map")
            or metadata.get("class_mapping")
            or {}
        )
        if not isinstance(mapping, dict):
            raise ValueError("label_map 必须是对象")
        threshold_value = metadata.get("confidence_threshold")
        confidence_threshold = float(threshold_value) if threshold_value is not None else None
        if confidence_threshold is not None and not 0.5 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold 超出 0.5–1.0 范围")

        import joblib

        loaded = joblib.load(artifact)
        if isinstance(loaded, dict) and {"base_model", "calibrator", "feature_columns"} <= loaded.keys():
            if tuple(loaded["feature_columns"]) != tuple(names):
                raise ValueError("模型与元数据的特征顺序不一致")
            if not callable(getattr(loaded["base_model"], "decision_function", None)):
                raise ValueError("基线模型缺少 decision_function")
            if not callable(getattr(loaded["calibrator"], "predict_proba", None)):
                raise ValueError("校准器缺少 predict_proba")
            estimator = loaded
            kind = "calibrated_good_probability"
        else:
            estimator = loaded.get("model", loaded.get("estimator")) if isinstance(loaded, dict) else loaded
            if estimator is None or not callable(getattr(estimator, "predict", None)):
                raise ValueError("模型文件未提供 predict 方法")
            kind = "classifier"
        return ModelAvailability(
            estimator,
            tuple(names),
            {str(key): str(value) for key, value in mapping.items()},
            "已加载本地基线 artifact；本页不声明或验证模型准确率。",
            kind,
            confidence_threshold,
        )
    except Exception as exc:
        return ModelAvailability(None, (), {}, f"模型 artifact 无法加载：{exc}")


def model_feature_values(image: Image.Image, metrics: QualityMetrics) -> dict[str, float]:
    """Use the core extractor when available, with UI metrics as explicit aliases."""
    values = metrics.feature_values()
    try:
        from chipqc.features import extract_quality_features
    except ImportError:
        return values

    extracted = extract_quality_features(image)
    if not isinstance(extracted, Mapping):
        raise ValueError("核心特征提取器未返回字典")
    for name, value in extracted.items():
        if isinstance(name, str) and isinstance(value, (int, float, np.number)):
            numeric = float(value)
            if np.isfinite(numeric):
                values[name] = numeric
    return values


def predict_baseline(
    model: ModelAvailability,
    image: Image.Image,
    metrics: QualityMetrics,
) -> ModelPrediction:
    if not model.available:
        raise ValueError("模型尚未加载")

    if model.kind == "mobilenet_v2_embedding":
        matrix = _mobilenet_v2_embedding(image)
        estimator = model.estimator
        raw_probability = np.clip(estimator["model"].predict_proba(matrix)[:, 1], 1e-6, 1 - 1e-6)
        score = np.log(raw_probability / (1 - raw_probability)).reshape(-1, 1)
        probability_good = float(estimator["calibrator"].predict_proba(score)[0, 1])
        threshold = float(estimator["decision_threshold"])
        raw_label = "1" if probability_good >= threshold else "0"
        label = model.label_map.get(raw_label, "good" if raw_label == "1" else "bad")
        confidence = probability_good if raw_label == "1" else 1.0 - probability_good
        return ModelPrediction(label, confidence, probability_good)

    values = model_feature_values(image, metrics)
    missing = [name for name in model.feature_names if name not in values]
    if missing:
        raise ValueError(f"缺少模型特征：{', '.join(missing)}")
    feature_values = [values[name] for name in model.feature_names]
    if not all(np.isfinite(value) for value in feature_values):
        raise ValueError("模型特征包含非有限数值")

    try:
        import pandas as pd

        matrix: Any = pd.DataFrame([feature_values], columns=model.feature_names)
    except ImportError:
        matrix = np.asarray([feature_values], dtype=float)

    if model.kind == "calibrated_good_probability":
        # Mirrors chipqc.modeling.predict_good_probability for a single upload.
        scores = model.estimator["base_model"].decision_function(matrix).reshape(-1, 1)
        probability_good = float(model.estimator["calibrator"].predict_proba(scores)[0, 1])
        if not np.isfinite(probability_good) or not 0.0 <= probability_good <= 1.0:
            raise ValueError("校准概率超出 0–1 范围")
        raw_label = "1" if probability_good >= 0.5 else "0"
        label = model.label_map.get(raw_label, "good" if raw_label == "1" else "bad")
        confidence = probability_good if raw_label == "1" else 1.0 - probability_good
        return ModelPrediction(label, confidence, probability_good)

    raw = model.estimator.predict(matrix)[0]
    if hasattr(raw, "item"):
        raw = raw.item()
    label = model.label_map.get(str(raw), str(raw))
    confidence: float | None = None
    if callable(getattr(model.estimator, "predict_proba", None)):
        probabilities = np.asarray(model.estimator.predict_proba(matrix)[0], dtype=float)
        classes = getattr(model.estimator, "classes_", None)
        if classes is not None:
            matching = [index for index, value in enumerate(classes) if str(value) == str(raw)]
            if matching:
                confidence = float(probabilities[matching[0]])
        if confidence is None and probabilities.size:
            confidence = float(probabilities.max())
        if confidence is not None and not np.isfinite(confidence):
            confidence = None
    return ModelPrediction(label=label, confidence=confidence)


@lru_cache(maxsize=1)
def _mobilenet_v2_runtime() -> tuple[Any, Any, Any]:
    """Load the frozen public backbone once per app process."""
    import torch
    from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

    weights = MobileNet_V2_Weights.IMAGENET1K_V2
    backbone = mobilenet_v2(weights=weights)
    backbone.classifier = torch.nn.Identity()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    backbone.eval().to(device)
    return backbone, weights.transforms(), device


def _mobilenet_v2_embedding(image: Image.Image) -> np.ndarray:
    import torch

    backbone, transform, device = _mobilenet_v2_runtime()
    normalized = Image.open(BytesIO(_image_png_bytes(image))).convert("RGB")
    with torch.inference_mode():
        embedding = backbone(transform(normalized).unsqueeze(0).to(device)).detach().cpu().numpy()
    if embedding.shape != (1, 1280) or not np.isfinite(embedding).all():
        raise ValueError(f"视觉表征无效：{embedding.shape}")
    return embedding.astype(np.float32)


def _image_png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
