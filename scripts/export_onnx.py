"""Export the culture-quality model as one ONNX graph for the browser demo, and check it against PyTorch.

    python scripts/export_onnx.py --model models/guardian-v2-demo --out web/model

Input  `tiles`:      N x 3 x 336 x 448, ImageNet-normalised tiles of the 1792 x 1344 frame (row-major, 4 x 4).
Output `projection`: N x 24 x 32, each patch's projection v . f_i on the classifier weights.
Output `tile_mean`:  N x C, the mean descriptor of each tile.
The page assembles the frame: the frame embedding is the mean of the tile means; its normalisation factor (1 / norm for
the released head) scales every projection; the bias and the calibrator are added; the frame's log-odds is the mean
over all patches of all tiles, and P(good) its sigmoid. Classifier weights are baked into the graph.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from chipqc.backbone import BACKBONES, patch_tokens  # noqa: E402
from chipqc.frames import open_frame, to_model_input  # noqa: E402


class TileHead(torch.nn.Module):
    def __init__(self, vit, v, blocks=1):
        super().__init__()
        self.vit, self.blocks = vit, blocks
        self.register_buffer("v", torch.tensor(v, dtype=torch.float32))

    def forward(self, tiles):
        tokens = patch_tokens(self.vit, tiles, self.blocks)
        return (tokens @ self.v).reshape(-1, 24, 32), tokens.mean(1)


def tiles_of(frame: np.ndarray) -> np.ndarray:
    """3 x H x W -> tiles x 3 x 336 x 448, row-major."""
    return np.stack([frame[:, r * 336:(r + 1) * 336, c * 448:(c + 1) * 448] for r in range(frame.shape[1] // 336) for c in range(frame.shape[2] // 448)])


def assemble(projection: np.ndarray, tile_mean: np.ndarray, spec: dict) -> tuple[np.ndarray, float]:
    """Tile outputs of one frame -> (evidence map per tile, P(good)); mirrors `assemble` in docs/core.js."""
    x = tile_mean.mean(0)
    scale = 1.0 / np.linalg.norm(x) if spec.get("embedding_norm") == "l2" else 1.0
    a, c = spec["platt"]
    evidence = a * (scale * projection + spec["cell_bias"]) + c
    return evidence, float(1 / (1 + np.exp(-evidence.mean())))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=ROOT / "models/guardian-v2-demo")
    ap.add_argument("--out", type=Path, default=ROOT / "web/model")
    ap.add_argument("--quantize", action="store_true", help="also write an 8-bit weight-quantised copy")
    a = ap.parse_args()
    import timm
    from timm.layers import set_fused_attn

    set_fused_attn(False)                                   # plain matmul attention exports cleanly
    spec = json.loads((a.model / "model.json").read_text())
    b = BACKBONES[spec["backbone"]]
    vit = timm.create_model(b["timm"], pretrained=True, num_classes=0, img_size=(336, 448)).eval()
    net = TileHead(vit, spec["cell_weights"], b.get("blocks", 1)).eval()
    size = tuple(spec["frame_size"])
    a.out.mkdir(parents=True, exist_ok=True)
    path = a.out / "guardian_tiles.onnx"
    torch.onnx.export(net, torch.zeros(1, 3, 336, 448), str(path), input_names=["tiles"], output_names=["projection", "tile_mean"],
                      dynamic_axes={"tiles": {0: "n"}, "projection": {0: "n"}, "tile_mean": {0: "n"}}, opset_version=17, dynamo=False)
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")

    import onnxruntime as ort
    candidates = [path]
    if a.quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        q = a.out / "guardian_tiles.q8.onnx"
        quantize_dynamic(str(path), str(q), weight_type=QuantType.QUInt8)
        print(f"wrote {q} ({q.stat().st_size / 1e6:.1f} MB)")
        candidates.append(q)
    reference = {}
    frames = sorted((ROOT / "examples").glob("*.png"))
    for f in frames:
        t = tiles_of(to_model_input(open_frame(f), size))
        with torch.inference_mode():
            proj, tm = net(torch.from_numpy(t))
        reference[f.stem] = assemble(proj.numpy(), tm.numpy(), spec)
    report = {}
    for cand in candidates:
        sess = ort.InferenceSession(str(cand), providers=["CPUExecutionProvider"])
        worst_map, worst_p = 0.0, 0.0
        for f in frames:
            proj, tm = sess.run(None, {"tiles": tiles_of(to_model_input(open_frame(f), size))})
            e, p = assemble(proj, tm, spec)
            e_ref, p_ref = reference[f.stem]
            worst_map = max(worst_map, float(np.abs(e - e_ref).max()))
            worst_p = max(worst_p, float(abs(p - p_ref)))
        report[cand.name] = {"max_abs_evidence_difference": worst_map, "max_abs_probability_difference": worst_p, "megabytes": round(cand.stat().st_size / 1e6, 1)}
        print(cand.name, report[cand.name])
    reference_p = {k: p for k, (_, p) in reference.items()}
    (a.out / "parity.json").write_text(json.dumps({"frames": len(frames), "pytorch_p_good": reference_p, "onnx_vs_pytorch": report}, indent=1))
