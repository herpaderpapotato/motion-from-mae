"""DNX temporal head: frozen video-backbone token packs -> per-frame HL-Gauss
position distribution + activity logits.

Spec: docs/disposition_next/02_architecture.md. The backbone (VideoMAE-B,
frozen) lives in src/data/videomae_features.py and is not part of this
module -- this starts at the slot-token cache boundary: input is the pooled
token pack [B, S, 5, D] (doc03 cache format), not raw video.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.hlgauss import HLGAUSS_N_BINS, hlgauss_decode

DNX_CONFIG_KEYS = {
    "backbone_hidden_dim", "d_model", "n_layers", "n_heads", "mlp_ratio",
    "dropout", "periodicity_max_lag", "periodicity_dim", "n_bins",
    "max_slots", "use_rope", "rope_base", "use_periodicity", "n_pool_tokens",
}

# Tokens per slot in the doc02/doc03 cache format (full mean + 4 quadrants).
# Checkpoints trained before the pooling layout was configurable carry no
# `n_pool_tokens` key, so this is what they get from `extract_dnx_config`.
LEGACY_N_POOL_TOKENS = 5


def extract_dnx_config(config: dict[str, object]) -> dict[str, object]:
    """Filter a checkpoint config dict down to DispositionNext constructor kwargs."""
    return {key: config[key] for key in DNX_CONFIG_KEYS if key in config}


# Buffers introduced after checkpoints already existed. Filled from the live
# model rather than from a hardcoded zero, so a head that sets one keeps it.
LEGACY_ABSENT_BUFFERS = ("token_dc",)


def upgrade_dnx_state(
    state: dict[str, torch.Tensor], model: "DispositionNext",
) -> dict[str, torch.Tensor]:
    """Add buffers a checkpoint predates so `load_state_dict` can stay strict."""
    upgraded = dict(state)
    model_state = model.state_dict()
    for name in LEGACY_ABSENT_BUFFERS:
        if name not in upgraded and name in model_state:
            upgraded[name] = model_state[name].detach().clone()
    return upgraded


class SlotProjector(nn.Module):
    """doc02: concat PxD -> Linear -> LayerNorm -> GELU -> Dropout.

    P is the cache's tokens-per-slot (5 for the doc02 full+quadrants layout,
    more for the finer pooling pyramids in
    `src/data/videomae_features.POOLING_LAYOUTS`). Only the input width changes;
    everything downstream of the projection is unaffected.
    """

    def __init__(
        self,
        backbone_hidden_dim: int,
        d_model: int,
        dropout: float = 0.1,
        n_pool_tokens: int = LEGACY_N_POOL_TOKENS,
    ):
        super().__init__()
        self.n_pool_tokens = n_pool_tokens
        self.net = nn.Sequential(
            nn.Linear(n_pool_tokens * backbone_hidden_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: [B, S, P, D] -> [B, S, d_model]."""
        b, s, n_pool, d = tokens.shape
        return self.net(tokens.reshape(b, s, n_pool * d))


class PeriodicityBranch(nn.Module):
    """doc02 banded temporal self-similarity: makes stroke period/phase and
    absence-of-periodicity explicit and appearance-invariant (targets F2).

    A single [B,S,S] matmul computes the full cosine-similarity matrix, then the
    128 lag bands are extracted in one gather (doc02's "single matmul + banded
    gather" note) rather than 128 separate dot products -- or, as this was
    originally written, 128 separate `torch.diagonal` calls.
    """

    def __init__(self, d_model: int, max_lag: int = 64, out_dim: int = 128):
        super().__init__()
        self.max_lag = max_lag
        self.lags = list(range(-max_lag, max_lag))  # 128 lags: -64..63
        n_lags = len(self.lags)
        # doc02 says "learnable temperature (init 5.0)"; parameterised in log
        # space so it can't go negative during training (init still == 5.0).
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(5.0)))
        self.mlp = nn.Sequential(
            nn.Linear(2 * n_lags, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
        )
        # (S, device) -> (gather index [S, n_lags], validity mask [S, n_lags]).
        # Plain dict, not a buffer: these are derived tables, not state, and must
        # not land in the checkpoint. RoPE lets S vary, hence the cache rather
        # than a fixed table -- in practice length-bucketed batches keep this to
        # one or two entries.
        self._band_cache: dict[tuple[int, torch.device], tuple[torch.Tensor, torch.Tensor]] = {}

    def _band_tables(self, s: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        cached = self._band_cache.get((s, device))
        if cached is not None:
            return cached
        n_lags = len(self.lags)
        t = torch.arange(s, device=device)
        lag = torch.arange(-self.max_lag, self.max_lag, device=device)
        # The loop this replaces computes, for both the lag>=0 and lag<0
        # branches alike, sim[t, li] = full_sim[t - lags[li], t] wherever that
        # row index is in range. After padding full_sim's ROW axis by max_lag on
        # both sides and transposing, that row lives at column
        # (t - lag) + max_lag == t + n_lags - li, which is always within
        # [0, s + n_lags), so no clamping is needed.
        idx = t[:, None] + n_lags - torch.arange(n_lags, device=device)[None, :]
        src_row = t[:, None] - lag[None, :]
        mask = ((src_row >= 0) & (src_row <= s - 1)).to(torch.float32)
        self._band_cache[(s, device)] = (idx, mask)
        return idx, mask

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        """e: [B, S, d_model] projected slot embeddings -> [B, S, out_dim]."""
        b, s, _ = e.shape
        idx, mask = self._band_tables(s, e.device)

        e32 = F.normalize(e.float(), dim=-1)  # fp32 cosine sim, doc02
        full_sim = torch.bmm(e32, e32.transpose(1, 2))  # [B, S, S], single matmul

        padded = F.pad(full_sim, (0, 0, self.max_lag, self.max_lag)).transpose(1, 2).contiguous()
        sim = torch.gather(padded, 2, idx.expand(b, -1, -1)) * mask
        sim = sim * self.log_temperature.exp()

        feat = torch.cat([sim, mask.expand(b, -1, -1)], dim=-1).to(e.dtype)  # [B, S, 2*n_lags]
        return self.mlp(feat)

    def lag_gather_reference(self, full_sim: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Original per-lag `torch.diagonal` loop, kept as the correctness
        oracle for the gather above (asserted equivalent in the model tests).
        Returns the un-temperatured (sim, mask) pair."""
        b, s, _ = full_sim.shape
        sim = full_sim.new_zeros(b, s, len(self.lags), dtype=torch.float32)
        mask = full_sim.new_zeros(b, s, len(self.lags), dtype=torch.float32)
        for li, lag in enumerate(self.lags):
            diag = torch.diagonal(full_sim, offset=lag, dim1=1, dim2=2)  # [B, S-|lag|]
            length = diag.shape[1]
            if length == 0:
                continue
            start = max(lag, 0)
            sim[:, start:start + length, li] = diag
            mask[:, start:start + length, li] = 1.0
        return sim, mask


class RotaryEmbedding(nn.Module):
    """Standard RoPE cos/sin tables, base 10000, applied on the head dim."""

    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # [T, head_dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [T, head_dim]
        return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """q, k: [B, n_heads, T, head_dim]; cos, sin: [T, head_dim]."""
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


class RotarySelfAttention(nn.Module):
    """Full self-attention (SDPA) with optional RoPE on q/k, doc02."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor | None,
        sin: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, t, d = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, n_heads, T, head_dim]
        if cos is not None:
            q, k = apply_rope(q, k, cos, sin)

        attn_mask = None
        if key_padding_mask is not None and not bool(key_padding_mask.all()):
            attn_mask = torch.zeros(b, 1, 1, t, device=x.device, dtype=q.dtype)
            attn_mask = attn_mask.masked_fill(~key_padding_mask[:, None, None, :], float("-inf"))

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(b, t, d)
        return self.proj(out)


class TransformerBlock(nn.Module):
    """Pre-norm block: LayerNorm only, no BatchNorm anywhere (doc02)."""

    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = RotarySelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        hidden = d_model * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, d_model), nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor | None,
        sin: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin, key_padding_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class DispositionNext(nn.Module):
    """DNX temporal head (doc02). Dimension-agnostic in the backbone hidden
    size: the slot projector consumes 5*backbone_hidden_dim, so the same head
    works for any backbone candidate's token cache.

    Input: tokens [B, S, P, D] (S = number of 2-frame slots, P =
    `n_pool_tokens`), valid [B, S] bool slot mask (padding). Output: per-frame
    (2 per slot) position bin logits + activity logits + decoded position
    expectation.
    """

    def __init__(
        self,
        backbone_hidden_dim: int = 768,
        d_model: int = 384,
        n_layers: int = 6,
        n_heads: int = 6,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        periodicity_max_lag: int = 64,
        periodicity_dim: int = 128,
        n_bins: int = HLGAUSS_N_BINS,
        max_slots: int = 512,
        use_rope: bool = True,
        rope_base: float = 10000.0,
        use_periodicity: bool = True,
        n_pool_tokens: int = LEGACY_N_POOL_TOKENS,
    ):
        super().__init__()
        self.backbone_hidden_dim = backbone_hidden_dim
        self.d_model = d_model
        self.n_bins = n_bins
        self.use_rope = use_rope
        self.max_slots = max_slots
        self.use_periodicity = use_periodicity
        self.n_pool_tokens = n_pool_tokens

        # Frozen V-JEPA 2.1 tokens carry a large constant offset; only ~10% of a
        # token varies with time. Per pooled token, not one shared vector -- the
        # pooling pyramid's cells sit at systematically different offsets. Rides in
        # the checkpoint, so inference needs no flag. Zeros = pre-DC behaviour.
        self.register_buffer("token_dc", torch.zeros(n_pool_tokens, backbone_hidden_dim))

        self.slot_projector = SlotProjector(backbone_hidden_dim, d_model, dropout, n_pool_tokens)
        if use_periodicity:
            # P1-B ablation (doc07 step 6) needs this structural, not just a
            # zeroed loss weight -- it changes what the transformer receives.
            self.periodicity = PeriodicityBranch(d_model, periodicity_max_lag, periodicity_dim)
            self.fuse = nn.Sequential(
                nn.Linear(d_model + periodicity_dim, d_model),
                nn.LayerNorm(d_model),
            )
        else:
            self.periodicity = None
            self.fuse = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
            )

        head_dim = d_model // n_heads
        if use_rope:
            self.rope = RotaryEmbedding(head_dim, base=rope_base)
            self.abs_pos = None
        else:
            # doc02-sanctioned fallback if RoPE proves fiddly; flagged in the
            # Phase 1 report which path is actually in use.
            self.rope = None
            self.abs_pos = nn.Parameter(torch.zeros(1, max_slots, d_model))
            nn.init.trunc_normal_(self.abs_pos, std=0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, mlp_ratio, dropout) for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        self.position_head = nn.Linear(d_model, 2 * n_bins)
        self.activity_head = nn.Linear(d_model, 2)

    def forward(self, tokens: torch.Tensor, valid: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """
        Args:
            tokens: [B, S, P, D].
            valid: [B, 2*S] bool, **frame-level** (doc03's granularity, same
                mask used for loss terms) -- NOT slot-level. A slot is treated
                as valid internally (for attention masking) iff at least one
                of its 2 frames is valid, matching doc03's slot-drop rule
                ("slots whose 2 frames are both padding are dropped").
        """
        b, s, n_pool, d = tokens.shape
        if d != self.backbone_hidden_dim:
            raise ValueError(f"tokens hidden dim {d} != backbone_hidden_dim {self.backbone_hidden_dim}")
        if n_pool != self.n_pool_tokens:
            # A 5-token cache fed to a pyramid3 head (or vice versa) would
            # otherwise fail deep inside the projector's matmul with an opaque
            # shape error; name both numbers instead.
            raise ValueError(
                f"tokens have {n_pool} pooled tokens per slot but this head was built for "
                f"{self.n_pool_tokens} -- the checkpoint's pooling layout and the token "
                "cache's do not match"
            )
        if valid is None:
            slot_valid = torch.ones(b, s, dtype=torch.bool, device=tokens.device)
        else:
            if valid.shape[1] != 2 * s:
                raise ValueError(f"valid must be frame-level [B, 2*S]; got {tuple(valid.shape)} for S={s}")
            slot_valid = valid.view(b, s, 2).any(dim=-1)

        tokens = tokens - self.token_dc.to(tokens.dtype)

        e = self.slot_projector(tokens)  # [B, S, d_model]
        if self.use_periodicity:
            p = self.periodicity(e)  # [B, S, periodicity_dim]
            x = self.fuse(torch.cat([e, p], dim=-1))  # [B, S, d_model]
        else:
            x = self.fuse(e)

        cos = sin = None
        if self.use_rope:
            cos, sin = self.rope(s, tokens.device, x.dtype)
        else:
            if s > self.max_slots:
                raise ValueError(f"sequence length {s} exceeds max_slots={self.max_slots} for learned abs. pos.")
            x = x + self.abs_pos[:, :s, :]

        for block in self.blocks:
            x = block(x, cos, sin, slot_valid)
        x = self.final_norm(x)

        pos_logits = self.position_head(x).reshape(b, s * 2, self.n_bins)  # slot s -> frames 2s, 2s+1
        act_logits = self.activity_head(x).reshape(b, s * 2)
        position = hlgauss_decode(pos_logits, self.n_bins)

        return {
            "position_logits": pos_logits,
            "activity_logits": act_logits,
            "position": position,
        }

    def count_parameters(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
