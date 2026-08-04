"""Self-contained VideoMAEv2-Base backbone (OpenGVLab/VideoMAEv2-Base).

Vendored rather than loaded via `trust_remote_code` for two reasons:

1. Dependency isolation. OpenGVLab's hub `modeling_videomaev2.py` imports
   `easydict` (not installed here) and `timm.layers` (a path that only exists
   in newer timm than the pinned 0.6.13). Vendoring the (bog-standard ViT)
   architecture removes both dependencies and the `trust_remote_code` failure
   surface flagged in docs/disposition_next/02_architecture.md.

2. LoRA correctness. The hub `Attention.forward` computes q/k/v with
   `F.linear(x, self.qkv.weight, ...)`, reading the weight tensor DIRECTLY
   instead of calling the `qkv` submodule. A peft LoRA adapter wraps the
   submodule's `__call__`, so an adapter on `qkv` there would be a SILENT
   no-op (its delta never runs) -- the exact class of quietly-wrong-features
   bug that `videomae_features._fix_videomae_qkv_bias` documents for the other
   backbone. Here `Attention.forward` routes through `self.qkv(x)`, so a LoRA
   adapter on `qkv` is active.

Numerical parity with the reference (no adapter attached): `self.qkv(x)` for a
bias-free Linear equals `F.linear(x, self.qkv.weight)`, and the separate
q_bias/0/v_bias is concatenated and added exactly as the reference does; the
sinusoidal position table, patch embed, block structure (init_values=0 -> no
LayerScale), and pre-`fc_norm` token pathway all match. `fc_norm` is loaded but
NOT applied to the token sequence -- the DNX pipeline consumes per-patch tokens
analogous to the other backbone's `last_hidden_state` (raw encoder output, no
final norm; `fc_norm` there is a pooling head, applied after mean over tokens).

Weights load from a local checkpoint dir's `model.safetensors`. The hub
checkpoint stores the ViT under a `model.` prefix (it wraps the ViT as
`self.model`); a LoRA-merged backbone saved by `videomae_features.save_videomaev2`
stores it unprefixed. The loader accepts either.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp


def get_sinusoid_encoding_table(n_position: int, d_hid: int) -> torch.Tensor:
    """Sinusoid position encoding table, identical to the OpenGVLab reference."""

    def get_position_angle_vec(position: int) -> list[float]:
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    table[:, 0::2] = np.sin(table[:, 0::2])  # even dims
    table[:, 1::2] = np.cos(table[:, 1::2])  # odd dims
    return torch.tensor(table, dtype=torch.float, requires_grad=False).unsqueeze(0)


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, act_layer=nn.GELU, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        # (reference intentionally drops after fc2 only, mirroring the BERT impl)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """Standard ViT self-attention with BEiT-style split q_bias/v_bias.

    Routes q/k/v through the `qkv` submodule call so a peft LoRA adapter on it
    is active (see module docstring). Uses SDPA for the attention matmul, which
    for scale=head_dim**-0.5 is numerically equivalent to the reference's
    `q = q*scale; softmax(q @ k^T)` while matching the memory profile of the
    other SDPA backbones.
    """

    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x)  # goes through the (LoRA-wrappable) submodule
        if self.q_bias is not None:
            bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias), self.v_bias))
            qkv = qkv + bias
        qkv = qkv.reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each [b, heads, n, head_dim]
        x = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop if self.training else 0.0, scale=self.scale,
        )
        x = x.transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self, dim: int, num_heads: int, mlp_ratio: float = 4.0, qkv_bias: bool = True,
        drop: float = 0.0, attn_drop: float = 0.0, norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # init_values=0 in this checkpoint -> no LayerScale (gamma) terms; and
        # drop_path_rate=0 -> the residual drop-path is identity.
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    """3D tubelet patch embedding via a single strided Conv3d (input B,C,T,H,W)."""

    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3,
                 embed_dim: int = 768, num_frames: int = 16, tubelet_size: int = 2):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.tubelet_size = tubelet_size
        num_spatial = (img_size // patch_size) * (img_size // patch_size)
        self.num_patches = num_spatial * (num_frames // tubelet_size)
        self.proj = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2], x.shape[-1]
        assert h == self.img_size[0] and w == self.img_size[1], (
            f"Input size ({h}x{w}) != model ({self.img_size[0]}x{self.img_size[1]})"
        )
        # [B, C, T, H, W] -> [B, D, T', H', W'] -> [B, T'*H'*W', D] (t,h,w row-major)
        return self.proj(x).flatten(2).transpose(1, 2)


class VideoMAEv2ViT(nn.Module):
    """VideoMAEv2 vision transformer exposing a per-patch token forward.

    `forward_tokens(pixel_values)` takes [B, C, T, H, W] and returns the
    post-block, pre-`fc_norm` token sequence [B, num_patches, D]. Set
    `with_cp = True` to gradient-checkpoint the blocks during training.
    """

    def __init__(
        self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768,
        depth: int = 12, num_heads: int = 12, mlp_ratio: float = 4.0, qkv_bias: bool = True,
        drop_rate: float = 0.0, attn_drop_rate: float = 0.0, layer_norm_eps: float = 1e-6,
        num_frames: int = 16, tubelet_size: int = 2, use_mean_pooling: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.tubelet_size = tubelet_size
        self.with_cp = False

        def norm_layer(dim: int) -> nn.Module:
            return nn.LayerNorm(dim, eps=layer_norm_eps)

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim, num_frames, tubelet_size)
        num_patches = self.patch_embed.num_patches
        # Non-persistent: not in the state_dict (matches the checkpoint, which
        # stores no pos_embed), but still moves/casts with `.to(...)`.
        self.register_buffer(
            "pos_embed", get_sinusoid_encoding_table(num_patches, embed_dim), persistent=False,
        )
        self.pos_drop = nn.Dropout(drop_rate)
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate, norm_layer)
            for _ in range(depth)
        ])
        # use_mean_pooling=True in this checkpoint: `norm` is identity, `fc_norm`
        # holds the (pooling-head) LayerNorm. Kept so the checkpoint loads
        # strictly, but NOT applied in forward_tokens (see module docstring).
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None

    def forward_tokens(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """pixel_values: [B, C, T, H, W] -> tokens [B, num_patches, D]."""
        x = self.patch_embed(pixel_values)
        x = x + self.pos_embed.type_as(x)
        x = self.pos_drop(x)
        for blk in self.blocks:
            if self.with_cp and self.training:
                x = cp.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        return x

    # Alias so callers that think in HF terms still reach the token pathway.
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.forward_tokens(pixel_values)


def _read_model_config(checkpoint_dir: str | Path) -> dict:
    cfg_path = Path(checkpoint_dir) / "config.json"
    with open(cfg_path) as f:
        cfg = json.load(f)
    # Original hub config nests the ViT hyperparameters under "model_config";
    # a merged-backbone dir copies that same config.json, so this key is present
    # in both cases.
    return cfg["model_config"]


def is_videomaev2_dir(checkpoint_dir: str | Path) -> bool:
    """True if `checkpoint_dir` looks like a VideoMAEv2 checkpoint (original or
    LoRA-merged): a config.json whose model_type starts with 'VideoMAEv2'."""
    cfg_path = Path(checkpoint_dir) / "config.json"
    if not cfg_path.exists():
        return False
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    return str(cfg.get("model_type", "")).lower().startswith("videomaev2")


def resolve_videomaev2_source(checkpoint_id: str | Path) -> tuple[Path, str | None]:
    """Resolve a local dir or an HF repo id to a local VideoMAEv2 checkpoint dir.

    Returns (dir, revision); revision is the hub commit sha, None for a local dir.
    """
    if Path(checkpoint_id).exists():
        if not is_videomaev2_dir(checkpoint_id):
            raise SystemExit(f"{checkpoint_id} is not a VideoMAEv2 checkpoint dir")
        return Path(checkpoint_id), None

    from huggingface_hub import snapshot_download

    local_dir = Path(snapshot_download(str(checkpoint_id)))
    if not is_videomaev2_dir(local_dir):
        raise SystemExit(f"{checkpoint_id} is not a VideoMAEv2 checkpoint repo")
    # .../snapshots/<sha>/ -- the sha is the backbone identity the token cache keys on.
    revision = local_dir.name if local_dir.parent.name == "snapshots" else None
    return local_dir, revision


def build_videomaev2_vit(checkpoint_dir: str | Path) -> tuple[VideoMAEv2ViT, dict]:
    """Instantiate the ViT from a checkpoint dir's config (no weights loaded)."""
    mc = _read_model_config(checkpoint_dir)
    model = VideoMAEv2ViT(
        img_size=int(mc["img_size"]), patch_size=int(mc["patch_size"]), in_chans=int(mc.get("in_chans", 3)),
        embed_dim=int(mc["embed_dim"]), depth=int(mc["depth"]), num_heads=int(mc["num_heads"]),
        mlp_ratio=float(mc.get("mlp_ratio", 4.0)), qkv_bias=bool(mc.get("qkv_bias", True)),
        drop_rate=float(mc.get("drop_rate", 0.0)), attn_drop_rate=float(mc.get("attn_drop_rate", 0.0)),
        layer_norm_eps=float(mc.get("layer_norm_eps", 1e-6)), num_frames=int(mc["num_frames"]),
        tubelet_size=int(mc["tubelet_size"]), use_mean_pooling=bool(mc.get("use_mean_pooling", True)),
    )
    return model, mc


def load_videomaev2_weights(model: VideoMAEv2ViT, checkpoint_dir: str | Path) -> None:
    """Load model.safetensors into `model`, stripping the hub's `model.` prefix
    if present (a merged backbone stores keys unprefixed)."""
    from safetensors.torch import load_file

    state = load_file(str(Path(checkpoint_dir) / "model.safetensors"))
    stripped = {(k[len("model."):] if k.startswith("model.") else k): v for k, v in state.items()}
    result = model.load_state_dict(stripped, strict=False)
    # pos_embed is a non-persistent buffer, so it never appears in either list.
    unexpected = list(result.unexpected_keys)
    missing = [k for k in result.missing_keys if k != "pos_embed"]
    if missing or unexpected:
        raise RuntimeError(
            f"VideoMAEv2 weight load mismatch for {checkpoint_dir}: missing={missing}, "
            f"unexpected={unexpected}"
        )
