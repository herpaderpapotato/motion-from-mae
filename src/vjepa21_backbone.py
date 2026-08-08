"""Self-contained V-JEPA 2.1 ViT encoder (video inference path).

Inference subset of the training repo's src/models/vjepa21_backbone.py, cut to
the forward pass. Numerics (RoPE, block structure, the final norm, the strict
weight load) are unchanged from it.

Vendored rather than imported from a checkout of facebookresearch/vjepa2 for one
hard reason and one soft one:

1. Import collision. The reference model code imports its own top-level `src`
   package (`src.masks.utils`, `src.utils.tensors`), and THIS repo's package is
   also called `src` and is already first on sys.path. Putting the checkout on
   sys.path cannot work -- one of the two `src` packages always shadows the
   other. Vendoring sidesteps it entirely.
2. It also drags in timm and einops for pieces this path never touches.

Scope: the video, unmasked forward. Dropped from the reference
(`app/vjepa_2_1/models/vision_transformer.py` + `models/utils/modules.py`), and
exact for the released 2.1 checkpoints:

  - masks / token dropping -- extraction never masks.
  - the image tokenizer branch (`patch_embed_img`, tubelet 1) and `img_mod_embed`.
    A 64-frame clip always takes the video branch; the reference only reaches the
    image branch when the input's temporal dim equals `img_temporal_dim_size`.
  - cls token and registers (`n_registers=0`, `has_cls_first=False` for these
    checkpoints), which reduce the reference's rotate-in-three-parts to a
    rotation over every token.
  - `pos_embed` interpolation -- 2.1 is RoPE-only and stores no position table.
  - weight init and `_rescale_blocks()`, both overwritten by the checkpoint load.
  - the deep-supervision return. `forward(training=True)` concatenates 4
    intermediate `norms_block` outputs into a 4*D vector for the distillation
    loss; plain inference returns `norms_block[-1](x)`, which is what the
    reference evals probe (`evals/video_classification_frozen`). All 4 norms are
    still built so the checkpoint loads strictly.

Token order is Conv3d -> `flatten(2).transpose(1, 2)`, i.e. (t, h, w) row-major,
the same order every other DNX backbone emits.

Run the encoder as fp32 weights under bf16 autocast, never `.to(bfloat16)`: the
RoPE tables are built at `pos`'s float32 dtype, so casting the module outright
leaves q/k float32 while v is bf16 and SDPA rejects the mismatch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# ImageNet stats: the reference eval/train transforms' DEFAULT_NORMALIZATION,
# which no eval_2_1 config overrides.
VJEPA21_NORM_MEAN = (0.485, 0.456, 0.406)
VJEPA21_NORM_STD = (0.229, 0.224, 0.225)

# A 2.1 release checkpoint is one .pt holding the trained encoder, its EMA, the
# predictor and optimizer state. The reference evals probe `ema_encoder`
# (`configs/eval_2_1/*/*.yaml: checkpoint_key`), so that is the default here.
DEFAULT_CHECKPOINT_KEY = "ema_encoder"

# The resolution the released 2.1 distilled checkpoints were trained at
# (`configs/eval_2_1/*/`: `resolution: 384`). RoPE takes positions from the input
# grid, so the encoder runs at other resolutions -- but the position indices then
# span a different range than during distillation, so treat any other value as an
# experiment to validate, not a free speedup.
DEFAULT_IMG_SIZE = 384

# Head count is the one geometry the checkpoint does not pin (qkv is fused), so
# it is looked up by width from the reference's vit_* constructors.
HEADS_BY_EMBED_DIM = {768: 12, 1024: 16, 1280: 16, 1408: 22, 1664: 26}

# Loaded by the reference but unused on the video path (see module docstring).
UNUSED_CHECKPOINT_KEYS = ("img_mod_embed", "patch_embed_img.proj.weight", "patch_embed_img.proj.bias")


def _rotate(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Rotary embedding along one axis: `x` is [B, heads, N, d] (d even), `pos`
    the [N] coordinate of each token on that axis."""
    d = x.shape[-1]
    omega = torch.arange(d // 2, dtype=x.dtype, device=x.device)
    omega /= d / 2.0
    omega = 1.0 / 10000**omega
    freq = torch.einsum("..., f -> ... f", pos, omega)
    emb_sin = freq.sin().repeat_interleave(2, dim=-1)
    emb_cos = freq.cos().repeat_interleave(2, dim=-1)
    y = x.unflatten(-1, (-1, 2))
    y1, y2 = y.unbind(dim=-1)
    y = torch.stack((-y2, y1), dim=-1).flatten(-2)
    return x * emb_cos + y * emb_sin


class RoPEAttention(nn.Module):
    """3D-RoPE self-attention: the head dim is split into equal depth/height/width
    thirds (rounded down to an even size), each rotated by its own axis position,
    with any remainder left unrotated. At D=768/12 heads that is 20+20+20 of 64.

    The reference's `with torch.backends.cuda.sdp_kernel():` wrapper is dropped --
    its no-arg form enables exactly the backends SDPA already selects by default,
    and the context manager is deprecated.
    """

    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True, grid_size: int = 24):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.axis_dim = int(2 * ((head_dim // 3) // 2))
        self.grid_size = grid_size

    def forward(self, x: torch.Tensor, t_patches: int, h_patches: int, w_patches: int) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        ids = torch.arange(t_patches * h_patches * w_patches, device=x.device)
        per_frame, per_row = h_patches * w_patches, w_patches
        d_pos = ids // per_frame
        h_pos = (ids - per_frame * d_pos) // per_row
        w_pos = (ids - per_frame * d_pos) - per_row * h_pos
        positions = (1.0 * d_pos, 1.0 * h_pos, 1.0 * w_pos)

        a = self.axis_dim
        q_parts, k_parts = [], []
        for axis, pos in enumerate(positions):
            sl = slice(axis * a, (axis + 1) * a)
            q_parts.append(_rotate(q[..., sl], pos))
            k_parts.append(_rotate(k[..., sl], pos))
        if 3 * a < self.head_dim:
            q_parts.append(q[..., 3 * a:])
            k_parts.append(k[..., 3 * a:])
        q = torch.cat(q_parts, dim=-1)
        k = torch.cat(k_parts, dim=-1)

        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(b, n, c))


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, qkv_bias: bool,
                 grid_size: int, layer_norm_eps: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.attn = RoPEAttention(dim, num_heads, qkv_bias=qkv_bias, grid_size=grid_size)
        self.norm2 = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor, t_patches: int, h_patches: int, w_patches: int) -> torch.Tensor:
        # drop_path_rate is 0 in these checkpoints -> the reference's DropPath is
        # Identity on both residuals.
        x = x + self.attn(self.norm1(x), t_patches, h_patches, w_patches)
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed3D(nn.Module):
    """Tubelet embedding via a strided Conv3d; input [B, C, T, H, W]."""

    def __init__(self, patch_size: int, tubelet_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class VJepa21ViT(nn.Module):
    """V-JEPA 2.1 vision transformer, video path only.

    `forward_tokens(pixel_values)` takes [B, C, T, H, W] and returns
    [B, num_patches, D].
    """

    N_DISTILLATION_NORMS = 4

    def __init__(
        self, img_size: int = 384, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768,
        depth: int = 12, num_heads: int = 12, mlp_ratio: float = 4.0, qkv_bias: bool = True,
        tubelet_size: int = 2, layer_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size

        self.patch_embed = PatchEmbed3D(patch_size, tubelet_size, in_chans, embed_dim)
        self.video_mod_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias, img_size // patch_size, layer_norm_eps)
            for _ in range(depth)
        ])
        # Only [-1] runs on this path; the rest exist so the load stays strict.
        self.norms_block = nn.ModuleList([
            nn.LayerNorm(embed_dim, eps=layer_norm_eps) for _ in range(self.N_DISTILLATION_NORMS)
        ])

    def forward_tokens(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """pixel_values: [B, C, T, H, W] -> tokens [B, num_patches, D], (t,h,w) row-major."""
        _, _, t, h, w = pixel_values.shape
        t_patches = t // self.tubelet_size
        h_patches, w_patches = h // self.patch_size, w // self.patch_size
        x = self.patch_embed(pixel_values) + self.video_mod_embed
        for blk in self.blocks:
            x = blk(x, t_patches, h_patches, w_patches)
        return self.norms_block[-1](x)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.forward_tokens(pixel_values)


def is_vjepa21_checkpoint(path: str | Path) -> bool:
    """Cheap dispatch test: a single `.pt`/`.pth` file, which is how the 2.1
    release ships (every other backbone here is a hub id or a directory). The
    real validation is in `load_vjepa21_vit`, which names what it expected."""
    p = Path(path)
    return p.is_file() and p.suffix.lower() in (".pt", ".pth")


def _encoder_state(checkpoint: Any, checkpoint_key: str, path: str | Path) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict) or checkpoint_key not in checkpoint:
        available = list(checkpoint)[:8] if isinstance(checkpoint, dict) else type(checkpoint).__name__
        raise ValueError(
            f"{path} is not a V-JEPA 2.1 checkpoint: no '{checkpoint_key}' entry (found {available}). "
            "Expected the released single-file .pt with encoder/ema_encoder/predictor entries."
        )
    state = checkpoint[checkpoint_key]
    # DDP + the training wrapper prefix every key; the reference eval loader
    # strips the same two.
    return {k.replace("module.", "").replace("backbone.", ""): v for k, v in state.items()}


def load_vjepa21_vit(
    checkpoint_path: str | Path,
    img_size: int | None = None,
    checkpoint_key: str = DEFAULT_CHECKPOINT_KEY,
) -> tuple[VJepa21ViT, dict[str, Any]]:
    """Build the ViT from the checkpoint's own tensor shapes and load it strictly.

    Resolution is not recoverable from the weights (RoPE, so no position table).
    Precedence: an explicit `img_size` wins (the head's
    `data_config['backbone_img_size']`), then one recorded in the checkpoint by
    the training repo's LoRA merge, then `DEFAULT_IMG_SIZE` -- what the released
    2.1 distilled models were trained at. Returns (model, geometry dict).
    """
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    state = _encoder_state(checkpoint, checkpoint_key, checkpoint_path)
    if img_size is None:
        stored = checkpoint.get("img_size") if isinstance(checkpoint, dict) else None
        img_size = int(stored) if stored is not None else DEFAULT_IMG_SIZE
    proj = state.get("patch_embed.proj.weight")
    if proj is None or proj.ndim != 5:
        raise ValueError(
            f"{checkpoint_path}['{checkpoint_key}'] has no 5-D 'patch_embed.proj.weight'; "
            "this does not look like a V-JEPA 2.1 video encoder."
        )
    embed_dim, _, tubelet_size, patch_size = proj.shape[0], proj.shape[1], proj.shape[2], proj.shape[-1]
    if img_size % int(patch_size) != 0:
        raise ValueError(
            f"img_size={img_size} is not a multiple of this checkpoint's patch size {int(patch_size)}; "
            "a partial patch would silently drop the edge of every frame."
        )
    depth = 1 + max(int(k.split(".")[1]) for k in state if k.startswith("blocks."))
    mlp_ratio = state["blocks.0.mlp.fc1.weight"].shape[0] / embed_dim
    if embed_dim not in HEADS_BY_EMBED_DIM:
        raise ValueError(
            f"{checkpoint_path}: embed_dim={embed_dim} is not a known V-JEPA 2.1 ViT width "
            f"(expected one of {sorted(HEADS_BY_EMBED_DIM)}); head count cannot be inferred from fused qkv."
        )

    model = VJepa21ViT(
        img_size=img_size, patch_size=int(patch_size), embed_dim=int(embed_dim), depth=int(depth),
        num_heads=HEADS_BY_EMBED_DIM[embed_dim], mlp_ratio=float(mlp_ratio),
        qkv_bias="blocks.0.attn.qkv.bias" in state, tubelet_size=int(tubelet_size),
    )
    result = model.load_state_dict(state, strict=False)
    unexpected = [k for k in result.unexpected_keys if k not in UNUSED_CHECKPOINT_KEYS]
    if result.missing_keys or unexpected:
        raise RuntimeError(
            f"V-JEPA 2.1 weight load mismatch for {checkpoint_path}['{checkpoint_key}']: "
            f"missing={list(result.missing_keys)}, unexpected={unexpected}"
        )

    geometry = {
        "embed_dim": int(embed_dim), "depth": int(depth), "num_heads": HEADS_BY_EMBED_DIM[embed_dim],
        "patch_size": int(patch_size), "tubelet_size": int(tubelet_size), "img_size": int(img_size),
        "checkpoint_key": checkpoint_key, "epoch": checkpoint.get("epoch"),
        "derived": bool(checkpoint.get("derived", False)) if isinstance(checkpoint, dict) else False,
    }
    return model, geometry


def find_vjepa21_checkpoint(source: str | Path) -> Path:
    """The weight file for an already-resolved source: `source` itself if it is
    the `.pt`, otherwise the single `.pt`/`.pth` inside the directory.

    A directory holding several is ambiguous and says so rather than guessing
    (hub resolution happens in `backbone._resolve_backbone_source`)."""
    path = Path(source)
    if path.is_file():
        return path
    candidates = sorted(p for p in path.iterdir() if p.suffix.lower() in (".pt", ".pth"))
    if not candidates:
        raise SystemExit(f"{source} contains no .pt/.pth V-JEPA 2.1 checkpoint")
    if len(candidates) > 1:
        raise SystemExit(
            f"{source} contains {len(candidates)} .pt files "
            f"({', '.join(p.name for p in candidates)}); it must hold exactly one"
        )
    return candidates[0]
