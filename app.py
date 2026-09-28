"""ChipQC Guardian: a non-clinical, pre-model image QC demonstration."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chipqc_ui.manifest import manifest_csv, manifest_json, manifest_row  # noqa: E402
from chipqc_ui.model_adapter import load_baseline_model, predict_baseline  # noqa: E402
from chipqc_ui.quality import classify_rule, extract_display_metrics  # noqa: E402


STATUS_COPY = {
    "PASS": "规则预览：未触发明显采集质量提醒",
    "REVIEW": "规则预览：建议人工查看图像采集质量",
    "REACQUIRE": "规则预览：明显采集缺陷，建议考虑复拍或实验复核",
}


def open_uploaded_image(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> Image.Image:
    """Decode only the selected image frame and normalize its orientation."""
    with Image.open(io.BytesIO(uploaded_file.getvalue())) as opened:
        opened.load()
        return ImageOps.exif_transpose(opened).convert("RGB")


def show_status(status: str, reasons: tuple[str, ...]) -> None:
    message = f"{status} · {STATUS_COPY[status]}。依据：{'；'.join(reasons)}。"
    if status == "PASS":
        st.success(message)
    elif status == "REVIEW":
        st.warning(message)
    else:
        st.error(message)


st.set_page_config(page_title="ChipQC Guardian", page_icon="🔬", layout="wide")

st.markdown(
    """
    <style>
    .block-container { max-width: 1200px; padding-top: 2rem; }
    h1 { letter-spacing: -0.035em; }
    [data-testid="stMetric"] { background: #f5f8fa; border: 1px solid #e1e9ed;
        border-radius: 12px; padding: 14px 18px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("ChipQC Guardian")
st.write("芯片/显微图像采集质量预览 · 非临床研究演示")

model = load_baseline_model(PROJECT_ROOT)
if model.available:
    st.info(f"**模型状态：本地研究模型已加载。** {model.message} 模型输出与规则预览分开展示。")
    if model.confidence_threshold is not None:
        st.warning(
            f"模型选择阈值 {model.confidence_threshold:.2f} 只是训练脚本中的预设；"
            "本界面没有证明它经过校准集选优或适用于真实操作，因此不据此给出复拍结论。"
        )
else:
    st.error(f"**模型状态：未训练 / 未接入。** {model.message} 下方 PASS / REVIEW / REACQUIRE 仅来自未校准的图像规则。")

st.warning(
    "**用途边界**：本页只演示图像采集质量检查流程。它不诊断疾病、不判断芯片是否合格，"
    "也不替代专家对样本的质量评定。数据集的 good/bad 是专家对样本质量的标签；"
    "bad 不等于必须重拍，本页的 REACQUIRE 只表示明显图像采集缺陷下的复拍或实验复核建议。"
)

with st.sidebar:
    st.subheader("判定说明")
    st.write("规则阈值是演示值，没有按设备、图像类型或专家标签校准。")
    st.write("PASS：未触发规则；REVIEW：建议人工查看；REACQUIRE：建议考虑复拍或实验复核。")
    st.caption("上传图像仅在当前应用会话中处理；导出的清单只含文件名、指标与预览结论。")
    st.subheader("合成演示图")
    st.caption("这些像素图不含生物样本，用于观察规则行为；规则 PASS 也不表示真实样本有效。")
    for sample_name, label in (
        ("synthetic_checker.pgm", "下载棋盘图 · 规则 PASS"),
        ("synthetic_flat.pgm", "下载平坦图 · 规则 REACQUIRE"),
    ):
        st.download_button(
            label,
            data=(PROJECT_ROOT / "demo_assets" / sample_name).read_bytes(),
            file_name=sample_name,
            mime="image/x-portable-graymap",
            use_container_width=True,
        )

mode = st.radio("处理方式", ("单张", "批量"), horizontal=True)
uploaded = st.file_uploader(
    "上传 JPG、PNG、TIFF、BMP、WebP 图像或合成 PGM 示例",
    type=["jpg", "jpeg", "png", "tif", "tiff", "bmp", "webp", "pgm"],
    accept_multiple_files=(mode == "批量"),
    key=f"upload_{mode}",
)
uploads = uploaded if isinstance(uploaded, list) else ([uploaded] if uploaded is not None else [])

if uploads:
    rows = []
    previews = []
    errors = []
    with st.spinner(f"正在分析 {len(uploads)} 张图像…"):
        for file in uploads:
            try:
                image = open_uploaded_image(file)
                metrics = extract_display_metrics(image)
                decision = classify_rule(metrics)
                model_output = ""
                model_confidence = None
                model_probability_good = None
                if model.available:
                    try:
                        prediction = predict_baseline(model, image, metrics)
                        model_output = prediction.label
                        model_confidence = prediction.confidence
                        model_probability_good = prediction.probability_good
                    except Exception as exc:
                        errors.append(f"{file.name}：模型推理不可用（{exc}）")
                rows.append(
                    manifest_row(
                        file.name,
                        metrics,
                        decision,
                        model_output,
                        model_confidence,
                        model_probability_good,
                    )
                )
                if mode == "单张" or len(previews) < 6:
                    previews.append((file.name, image, metrics, decision))
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                errors.append(f"{file.name}：无法解析图像（{exc}）")

    if errors:
        st.warning("部分文件未完成处理：\n\n" + "\n\n".join(f"- {error}" for error in errors))

    if rows:
        st.subheader(f"分析结果 · {len(rows)} 张")
        if mode == "单张":
            filename, image, metrics, decision = previews[0]
            left, right = st.columns([1, 1.4], gap="large")
            with left:
                st.image(image, caption=f"{filename} · {metrics.width} × {metrics.height} px", use_container_width=True)
            with right:
                show_status(decision.status, decision.reasons)
                metric_cols = st.columns(2)
                metric_cols[0].metric("清晰度", f"{metrics.clarity:.1f}")
                metric_cols[1].metric("亮度", f"{metrics.brightness:.3f}")
                metric_cols[0].metric("对比度", f"{metrics.contrast:.3f}")
                metric_cols[1].metric("饱和/剪切比例", f"{metrics.clipped_fraction:.1%}")
                if model.available:
                    if rows[0]["model_output"]:
                        probability_good = rows[0]["model_probability_good"]
                        confidence = rows[0]["model_confidence"]
                        if isinstance(probability_good, float):
                            suffix = f" · P(good) {probability_good:.1%}"
                        elif isinstance(confidence, float):
                            suffix = f" · 输出类别概率 {confidence:.1%}"
                        else:
                            suffix = ""
                        st.info(f"模型输出（研究性）：{rows[0]['model_output']}{suffix}")
                    else:
                        st.warning("该图像没有可用的模型输出；规则预览仍可查看。")
        else:
            st.dataframe(rows, hide_index=True, use_container_width=True)
            with st.expander("查看前 6 张图像及规则依据"):
                for filename, image, metrics, decision in previews[:6]:
                    st.image(image, caption=f"{filename} · {metrics.width} × {metrics.height} px", width=280)
                    show_status(decision.status, decision.reasons)
            if len(rows) > 6:
                st.caption("其余图像的指标与结论可在表格和导出清单中查看。")

        st.caption("模型概率若显示，仅表示该模型对自身输出的分数；它不是准确率，也不是临床可信度。")
        download_csv, download_json = st.columns(2)
        download_csv.download_button(
            "下载 qc_manifest.csv",
            data=manifest_csv(rows),
            file_name="qc_manifest.csv",
            mime="text/csv",
            use_container_width=True,
        )
        download_json.download_button(
            "下载 qc_manifest.json",
            data=manifest_json(rows),
            file_name="qc_manifest.json",
            mime="application/json",
            use_container_width=True,
        )
else:
    st.info("上传一张图像开始预览；批量模式可一次选择多张。")

with st.expander("四项指标和演示规则如何计算"):
    st.markdown(
        """
        指标在最长边缩至 1024 像素的灰度视图上计算，图像尺寸仍显示原图尺寸。

        | 指标 | 计算方式 | REVIEW 区间 | REACQUIRE 区间 |
        |---|---|---|---|
        | 清晰度 | 拉普拉斯响应的方差，越大通常边缘越明显 | < 70 | < 25 |
        | 亮度 | 灰度均值 ÷ 255 | < 0.30 或 > 0.80 | < 0.15 或 > 0.90 |
        | 对比度 | 灰度标准差 ÷ 255 | < 0.10 | < 0.045 |
        | 饱和/剪切比例 | 灰度 ≤ 5 或 ≥ 250 的像素占比 | > 30% | > 65% |

        多项规则同时触发时，显示最严重的建议及全部原因。这里的“饱和”是黑白端像素剪切，
        不是颜色饱和度。阈值未用真实标签训练或验证，也不对应数据集的 good/bad 标签。
        """
    )
