"""Whole-frame encoders.

A backbone turns a frame into a grid of cell descriptors. The frame embedding is their mean. Because the
classifier is linear in that mean, every grid cell has an exact additive share of the frame's score: the
evidence map is the model's own arithmetic, not a post-hoc approximation.

The released model uses DINOv2 ViT-S/14 (self-supervised, Apache-2.0). The 1792 x 1344 frame is cut into 4 x 4
non-overlapping tiles of 448 x 336 px; each tile is one forward pass; every 14 px patch is one grid cell
(128 x 96 cells per frame), described by its tokens after the last four transformer blocks (4 x 384 values).
The 3 x 3 tilings and the supervised ImageNet CNNs (one cell per 32 px) are kept for the ablations.
"""
from __future__ import annotations

import numpy as np

BACKBONES = {
    "dinov2_vits14": {"kind": "vit", "timm": "vit_small_patch14_reg4_dinov2.lvd142m", "dim": 384, "tile": (448, 336), "grid": (3, 3)},
    "dinov2_vits14_l4": {"kind": "vit", "timm": "vit_small_patch14_reg4_dinov2.lvd142m", "dim": 1536, "tile": (448, 336), "grid": (3, 3), "blocks": 4},
    "dinov2_vits14_4x4_l4": {"kind": "vit", "timm": "vit_small_patch14_reg4_dinov2.lvd142m", "dim": 1536, "tile": (448, 336), "grid": (4, 4), "blocks": 4},
    "dinov2_vitb14": {"kind": "vit", "timm": "vit_base_patch14_reg4_dinov2.lvd142m", "dim": 768, "tile": (448, 336), "grid": (3, 3)},
    "convnext_tiny": {"kind": "cnn", "torchvision": ("convnext_tiny", "ConvNeXt_Tiny_Weights", "IMAGENET1K_V1"), "dim": 768},
    "mobilenet_v2": {"kind": "cnn", "torchvision": ("mobilenet_v2", "MobileNet_V2_Weights", "IMAGENET1K_V2"), "dim": 1280},
    "resnet50": {"kind": "cnn", "torchvision": ("resnet50", "ResNet50_Weights", "IMAGENET1K_V2"), "dim": 2048},
}


def load_backbone(name: str):
    import torch

    spec = BACKBONES[name]
    if spec["kind"] == "vit":
        import timm

        tw, th = spec["tile"]
        return timm.create_model(spec["timm"], pretrained=True, num_classes=0, img_size=(th, tw)).eval()
    import torchvision.models as tvm

    ctor, enum_name, member = spec["torchvision"]
    net = getattr(tvm, ctor)(weights=getattr(getattr(tvm, enum_name), member))
    if name == "resnet50":
        return torch.nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1, net.layer2, net.layer3, net.layer4).eval()
    return net.features.eval()


def patch_tokens(net, tiles, blocks: int = 1):
    """Patch tokens of a ViT: tiles x patches x C. With blocks > 1, the tokens after each of the last `blocks`
    transformer blocks (final norm applied) are concatenated along C, earliest block first."""
    if blocks == 1:
        return net.forward_features(tiles)[:, net.num_prefix_tokens:]
    import torch

    return torch.cat(net.get_intermediate_layers(tiles, n=blocks, norm=True), dim=-1)


class FrameEncoder:
    def __init__(self, name: str = "dinov2_vits14_4x4_l4", device: str | None = None, grid: tuple[int, int] | None = None):
        """`grid` (tiles across, tiles down) overrides the 3 x 3 tiling of a ViT backbone; used only by the ablations."""
        import torch

        self.name, self.spec, self.dim = name, dict(BACKBONES[name]), BACKBONES[name]["dim"]
        if grid is not None:
            self.spec["grid"] = tuple(grid)
        self.device = torch.device(device or ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"))
        self.net = load_backbone(name).to(self.device)

    @property
    def frame_size(self) -> tuple[int, int]:
        """Width and height of the frame this encoder reads (frames.to_model_input)."""
        if self.spec["kind"] == "cnn":
            return (1344, 1008)
        (tw, th), (nx, ny) = self.spec["tile"], self.spec["grid"]
        return (nx * tw, ny * th)

    def cell_descriptors(self, batch: np.ndarray) -> np.ndarray:
        """batch: B x 3 x H x W (frames.to_model_input) -> B x h x w x C float32, one descriptor per grid cell."""
        import torch

        x = torch.from_numpy(batch).to(self.device)
        with torch.inference_mode():
            if self.spec["kind"] == "cnn":
                return self.net(x).permute(0, 2, 3, 1).float().cpu().numpy()
            (tw, th), (nx, ny) = self.spec["tile"], self.spec["grid"]
            if x.shape[2:] != (ny * th, nx * tw):
                raise ValueError(f"{self.name} expects {nx * tw} x {ny * th} px frames, got {x.shape[3]} x {x.shape[2]}")
            b = x.shape[0]
            tiles = x.unfold(2, th, th).unfold(3, tw, tw).permute(0, 2, 3, 1, 4, 5).reshape(b * ny * nx, 3, th, tw)
            tokens = patch_tokens(self.net, tiles, self.spec.get("blocks", 1))                   # tiles x (gh * gw) x C
            gh, gw = th // 14, tw // 14
            grid = tokens.reshape(b, ny, nx, gh, gw, -1).permute(0, 1, 3, 2, 4, 5).reshape(b, ny * gh, nx * gw, -1)
            return grid.float().cpu().numpy()

    def embed(self, batch: np.ndarray) -> np.ndarray:
        return self.cell_descriptors(batch).mean(axis=(1, 2))
