"""Frozen VideoMAEv2 backbone: geometry, frame preprocessing, pooled slot tokens.

Inference subset of the training repo's src/data/videomae_features.py, cut to the
one backbone family this repo runs. Numerics (crop box, resize, normalisation,
pooling, window padding, bf16 autocast) are unchanged from it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

# x1, y1, x2, y2 fractions of the eye: the centre-bottom crop.
CROP_BOX = (1 / 6, 1 / 3, 5 / 6, 1.0)
FULL_FRAME_CROP_BOX = (0.0, 0.0, 1.0, 1.0)

# Each layout is a pyramid of grid resolutions k, contributing k*k mean-pooled
# tokens, concatenated in order. Level 1 first: token 0 is always the full-grid
# mean. The head's projector width pins which layout a checkpoint needs.
POOLING_LAYOUTS: dict[str, tuple[int, ...]] = {
    "quadrants": (1, 2),     # 5 tokens
    "pyramid3": (1, 2, 3),   # 14 tokens
    "pyramid7": (1, 2, 7),   # 54 tokens
}
DEFAULT_POOLING = "quadrants"


def pooling_num_tokens(pooling: str = DEFAULT_POOLING) -> int:
    return sum(k * k for k in resolve_pooling(pooling))


def pooling_from_num_tokens(n_pool_tokens: int) -> str:
    for name in POOLING_LAYOUTS:
        if pooling_num_tokens(name) == n_pool_tokens:
            return name
    raise ValueError(
        f"no pooling layout produces {n_pool_tokens} tokens/slot; known layouts: "
        + ", ".join(f"{k}={pooling_num_tokens(k)}" for k in sorted(POOLING_LAYOUTS))
    )


def resolve_pooling(pooling: str | None) -> tuple[int, ...]:
    name = pooling or DEFAULT_POOLING
    if name not in POOLING_LAYOUTS:
        raise ValueError(f"unknown pooling layout {name!r}; expected one of {sorted(POOLING_LAYOUTS)}")
    return POOLING_LAYOUTS[name]


@dataclass
class BackboneGeometry:
    backbone_id: str
    backbone_revision: str | None
    family: str
    hidden_dim: int
    tubelet_size: int
    patch_size: int
    window: int  # native clip length in frames
    spatial_grid: tuple[int, int]  # (Hg, Wg) patches
    resize: tuple[int, int]  # (H, W) fed to the backbone
    norm_mean: tuple[float, float, float]
    norm_std: tuple[float, float, float]
    slug: str

    @property
    def frames_per_slot(self) -> int:
        return self.tubelet_size

    @property
    def slots_per_window(self) -> int:
        return self.window // self.tubelet_size


def _assert_token_grid(model: Any, geometry: BackboneGeometry, device: torch.device) -> None:
    dummy = torch.randn(1, geometry.window, 3, *geometry.resize, device=device, dtype=torch.float32)
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        hidden = _run_backbone(model, geometry, dummy)
    expected_patches = geometry.slots_per_window * geometry.spatial_grid[0] * geometry.spatial_grid[1]
    if hidden.shape[1] != expected_patches or hidden.shape[2] != geometry.hidden_dim:
        raise RuntimeError(
            f"Token grid assertion failed for {geometry.backbone_id}: got hidden shape "
            f"{tuple(hidden.shape)}, expected [B, {expected_patches}, {geometry.hidden_dim}] "
            f"(slots_per_window={geometry.slots_per_window} x grid={geometry.spatial_grid})"
        )


def _run_backbone(model: Any, geometry: BackboneGeometry, pixel_values: torch.Tensor) -> torch.Tensor:
    """[B, num_frames, 3, H, W] -> [B, num_patches, D]. The ViT's Conv3d tubelet
    embed wants channels first, hence the permute."""
    return model.forward_tokens(pixel_values.permute(0, 2, 1, 3, 4))


def load_backbone(
    checkpoint_id: str, device: str | torch.device = "cuda", slug_override: str | None = None,
) -> tuple[Any, BackboneGeometry]:
    """Load the frozen VideoMAEv2 ViT from a local dir or an HF repo id.

    All geometry comes from the checkpoint's own config.json /
    preprocessor_config.json, then the token-grid reshape is asserted once.
    """
    from src.videomaev2_backbone import (
        build_videomaev2_vit,
        load_videomaev2_weights,
        resolve_videomaev2_source,
    )

    device = torch.device(device)
    source_dir, revision = resolve_videomaev2_source(checkpoint_id)

    model, mc = build_videomaev2_vit(source_dir)
    load_videomaev2_weights(model, source_dir)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    with open(Path(source_dir) / "preprocessor_config.json") as f:
        pp = json.load(f)
    size = pp.get("crop_size", pp.get("size", mc["img_size"]))
    if isinstance(size, dict):
        edge = size.get("shortest_edge")
        crop = (int(size.get("height", edge)), int(size.get("width", edge)))
    else:
        crop = (int(size), int(size))

    hidden = int(mc["embed_dim"])
    patch = int(mc["patch_size"])
    # Size-derived so different VideoMAEv2 variants never collide in the token
    # cache; every OpenGVLab config shares model_type "VideoMAEv2_Base".
    size_code = {384: "s", 768: "b", 1024: "l", 1280: "h", 1408: "g"}.get(hidden, str(hidden))
    geometry = BackboneGeometry(
        backbone_id=str(checkpoint_id),
        backbone_revision=revision,
        family="videomaev2",
        hidden_dim=hidden,
        tubelet_size=int(mc["tubelet_size"]),
        patch_size=patch,
        window=int(mc["num_frames"]),
        spatial_grid=(crop[0] // patch, crop[1] // patch),
        resize=crop,
        norm_mean=tuple(float(x) for x in pp["image_mean"]),
        norm_std=tuple(float(x) for x in pp["image_std"]),
        slug=slug_override or f"videomaev2-{size_code}",
    )
    _assert_token_grid(model, geometry, device)
    return model, geometry


def default_batch_windows(geometry: BackboneGeometry, budget_frames: int = 128) -> int:
    h, w = geometry.resize
    scaled = budget_frames / geometry.window * (224 * 224) / (h * w)
    return max(1, int(round(scaled)))


def crop_resize_normalize(
    frames_nhwc_uint8: torch.Tensor,
    geometry: BackboneGeometry,
    device: torch.device,
    crop_box: tuple[float, float, float, float] | None = None,
) -> torch.Tensor:
    """[N, H, W, C] uint8 -> crop -> resize to the backbone resolution -> normalise.

    `crop_box` None is CROP_BOX (the centre-bottom crop). Returns [N, 3, H, W] float32.
    """
    frames = frames_nhwc_uint8.to(device)
    h, w = frames.shape[1], frames.shape[2]
    x1, y1, x2, y2 = crop_box if crop_box is not None else CROP_BOX
    x1_px, x2_px = int(round(x1 * w)), int(round(x2 * w))
    y1_px, y2_px = int(round(y1 * h)), h if y2 >= 1.0 else int(round(y2 * h))
    cropped = frames[:, y1_px:y2_px, x1_px:x2_px, :]
    nchw = cropped.permute(0, 3, 1, 2).float()
    if nchw.shape[-2:] != geometry.resize:
        nchw = F.interpolate(nchw, size=geometry.resize, mode="bilinear", align_corners=False)
    resized = nchw / 255.0
    mean = torch.tensor(geometry.norm_mean, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(geometry.norm_std, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    return (resized - mean) / std


def _pool_slots(patch_grid: torch.Tensor, pooling: str | None = DEFAULT_POOLING) -> torch.Tensor:
    """[..., Hg, Wg, D] -> [..., n_tokens, D] for the named layout.

    `quadrants` keeps its hand-written implementation: it agrees with the generic
    pyramid path only to float tolerance (different reduction order), and it is
    what produced every existing cache.
    """
    levels = resolve_pooling(pooling)
    if levels == POOLING_LAYOUTS[DEFAULT_POOLING]:
        hg, wg = patch_grid.shape[-3], patch_grid.shape[-2]
        h2, w2 = hg // 2, wg // 2
        full = patch_grid.mean(dim=(-3, -2))
        tl = patch_grid[..., :h2, :w2, :].mean(dim=(-3, -2))
        tr = patch_grid[..., :h2, w2:, :].mean(dim=(-3, -2))
        bl = patch_grid[..., h2:, :w2, :].mean(dim=(-3, -2))
        br = patch_grid[..., h2:, w2:, :].mean(dim=(-3, -2))
        return torch.stack([full, tl, tr, bl, br], dim=-2)

    lead = patch_grid.shape[:-3]
    hg, wg, d = patch_grid.shape[-3], patch_grid.shape[-2], patch_grid.shape[-1]
    x = patch_grid.reshape(-1, hg, wg, d).permute(0, 3, 1, 2)
    # flatten(2) is row-major over (h, w), so a k=2 level yields TL/TR/BL/BR in
    # the same order the legacy path stacks them.
    parts = [F.adaptive_avg_pool2d(x, k).flatten(2).transpose(1, 2) for k in levels]
    return torch.cat(parts, dim=1).reshape(*lead, -1, d)


def clip_tokens_from_frames(
    model: Any,
    geometry: BackboneGeometry,
    normalized_frames: torch.Tensor,
    device: torch.device,
    batch_windows: int | None = None,
    pooling: str | None = DEFAULT_POOLING,
) -> np.ndarray:
    """Run the backbone over non-overlapping windows and pool to slot token packs.

    normalized_frames: [N, 3, H, W] float32 (no padding). Returns [S, P, D]
    float16, S = ceil(N / tubelet_size); the trailing window is padded by
    repeating the last frame, and slots that are entirely padding are dropped.
    """
    n_pool = pooling_num_tokens(pooling)
    if batch_windows is None:
        batch_windows = default_batch_windows(geometry)
    n_real = normalized_frames.shape[0]
    window = geometry.window
    n_windows = -(-n_real // window)
    pad_needed = n_windows * window - n_real
    if pad_needed > 0:
        pad = normalized_frames[-1:].expand(pad_needed, -1, -1, -1)
        padded = torch.cat([normalized_frames, pad], dim=0)
    else:
        padded = normalized_frames
    h, w = geometry.resize
    padded = padded.reshape(n_windows, window, 3, h, w)

    hg, wg = geometry.spatial_grid
    slots_per_window = geometry.slots_per_window
    d = geometry.hidden_dim

    pooled_chunks = []
    for start in range(0, n_windows, batch_windows):
        batch = padded[start:start + batch_windows].to(device)
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            hidden = _run_backbone(model, geometry, batch)
        hidden = hidden.float()
        b = hidden.shape[0]
        grid = hidden.reshape(b, slots_per_window, hg, wg, d)
        pooled = _pool_slots(grid, pooling)
        pooled_chunks.append(pooled.reshape(b * slots_per_window, n_pool, d).cpu())
        del hidden, grid, pooled, batch

    tokens_all = torch.cat(pooled_chunks, dim=0)
    n_real_slots = -(-n_real // geometry.frames_per_slot)
    return tokens_all[:n_real_slots].to(torch.float16).numpy()
