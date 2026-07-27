# DispositionNext (DNX) — Full Technical Description

*Sources: [src/models/disposition_next.py](../src/models/disposition_next.py),
[src/training/dnx_losses.py](../src/training/dnx_losses.py),
[src/data/videomae_features.py](../src/data/videomae_features.py). Parameter counts
below are for the default config and computed analytically; `DispositionNext.count_parameters()`
gives exact numbers for any config.*

## 1. System boundary and data contract

The trainable model is a **temporal head over cached backbone tokens**, not an
end-to-end video network. The pipeline:

```
video (60 fps native)
  → canonical frame sampling (55–65 fps passes through untouched; anything else is
    nearest-timestamp resampled onto 60 Hz)
  → Either the full frame or crop(train on both) to CROP_BOX = (0.16, 0.3333, 0.8333, 1.0) fractions (center-bottom), resize to
    backbone resolution (224×224), ImageNet-style normalization
  → frozen VideoMAEv2-B, non-overlapping 16-frame windows (stride = window). VideoMAE-B ssv2 is an alternative, but the specialization did not perform well for all scenes tested.
  → per-window: 1568 patch tokens [8 slots × 14×14 grid × 768]
  → pool each slot's 14×14 grid to 5 tokens (full mean + TL/TR/BL/BR quadrant means)
  → cache [S, 5, 768] float16 to HDF5 (lzf + shuffle), with self-describing metadata
    (backbone id/revision, crop box, feature fps, normalization, format version)
  → DNX head: [B, S, 5, 768] + frame-level valid mask [B, 2S]
  → outputs: position_logits [B, 2S, 64], activity_logits [B, 2S], position [B, 2S]
```

**Backbone I/O** (architecture out of scope here): input `pixel_values`
`[B, 16, 3, 224, 224]`; output `last_hidden_state` `[B, 1568, 768]`, token order
(temporal, height, width) row-major, so `reshape(B, 8, 14, 14, 768)` is valid. The
loader restores `q_bias`/`v_bias` that `transformers`' `from_pretrained` silently
zero-inits due to a checkpoint key-layout mismatch (`_fix_videomae_qkv_bias`), and
asserts the token-grid reshape once with a dummy clip. Geometry (tubelet, patch,
hidden, window, resolution) is always read from the checkpoint config, never
hardcoded — the head is dimension-agnostic in `backbone_hidden_dim`, so alternative
backbones (V-JEPA 2 ViT-L, D=1024, 64-frame windows; vendored VideoMAEv1) drop in via
the same cache format.

## 2. Layer-by-layer rundown (default config)

Defaults: `backbone_hidden_dim=768, d_model=384, n_layers=6, n_heads=6, mlp_ratio=4,
dropout=0.1, periodicity_max_lag=64, periodicity_dim=128, n_bins=64, use_rope=True`.

| # | Module | Layers | Shapes | Params |
|---|---|---|---|---|
| 1 | `SlotProjector` | `Linear(3840→384) → LayerNorm → GELU → Dropout` | `[B,S,5,768] → [B,S,384]` | 1.476M |
| 2 | `PeriodicityBranch` | fp32 L2-normalize → `bmm` cosine sim `[B,S,S]` → 128 diagonal gathers (lags −64…63) → ×exp(log_temperature) → concat sim+mask `[B,S,256]` → `Linear(256→128) → GELU → Linear(128→128) → LayerNorm` | `[B,S,384] → [B,S,128]` | 0.050M |
| 3 | Fuse | `Linear(512→384) → LayerNorm` (or `Linear(384→384)` when periodicity is structurally disabled) | `[B,S,512] → [B,S,384]` | 0.198M |
| 4 | Positional encoding | RoPE (base 10000) on q/k per head (head_dim 64), **or** learned absolute embeddings `[1, max_slots=512, 384]` as the sanctioned fallback | — | 0 (RoPE) |
| 5 | Trunk ×6 | Pre-norm block: `x + Attn(LN(x))`, `x + MLP(LN(x))`; attention = fused-QKV SDPA, 6 heads; MLP = `Linear(384→1536) → GELU → Dropout → Linear(1536→384) → Dropout` | `[B,S,384]` | 6 × 1.774M = 10.647M |
| 6 | `final_norm` | LayerNorm(384) | — | ~0.001M |
| 7 | `position_head` | `Linear(384 → 2×64)` | `[B,S,384] → [B,2S,64]` | 0.049M |
| 8 | `activity_head` | `Linear(384 → 2)` | `[B,S,384] → [B,2S]` | ~0.001M |

**Total ≈ 12.4M trainable** (vs ~86M frozen in VideoMAE-B). Each slot spans 2 frames
(the tubelet size), so slot `s` emits predictions for frames `2s` and `2s+1` — the
frame-rate mismatch between the slot sequence (30 Hz) and the label/output sequence
(60 Hz) is resolved entirely in the output heads.

**Masking semantics:** `valid` is frame-level `[B, 2S]` (the same mask the losses
use); attention uses slot-level validity = "any of the slot's 2 frames valid",
matching the cache's slot-drop rule. The attention mask is only materialized when
padding actually exists (`key_padding_mask.all()` short-circuit), keeping the common
unpadded path on SDPA's fast path.

**Position decode (HL-Gauss):** `position = Σₖ softmax(logits)ₖ · (k+0.5)/64` — the
expectation over bin centers, computed inside `forward` so downstream consumers get a
scalar curve without knowing the bin layout.

## 3. Losses (trainer-weighted; see doc04 naming)

- **L_shift** — shift-marginalized HL-Gauss CE. Per shift `s ∈ {−5..+5}` (feature
  frames, ≈±83 ms at 60 fps): CE between predicted bin distributions and a discretized
  Gaussian target (σ = 0.02 in position units, vs bin width 1/64 ≈ 0.0156 — the target
  spans ~2–3 bins, which is the point: soft targets with inter-bin credit). Labels are
  shifted against unshifted logits; edge frames outside the overlap are sliced away,
  not just masked. Per-sequence shift losses are combined with a softmin
  (`−τ·logsumexp(−L/τ)`, τ = 0.05 ≈ near-hard min), and the per-sequence argmin shift
  is logged as a dataset-lag diagnostic. A 500-step warmup uses only s = 0 to prevent
  early shift-shopping.
- **L_vel** — `1 − Pearson(Δpred, Δlabel)` per sequence, batch-mean. Sequences whose
  label deltas are near-constant (std < 1e-4) contribute a graph-connected zero
  (`0.0 * dp.sum()`, not a detached literal — a detached zero would break `backward()`
  when *every* sequence in a batch is flat).
- **L_act** — masked BCE-with-logits with a global (dataset-level, never per-batch)
  `pos_weight` for hold/active imbalance.
- **L_anchor** — MSE of the decoded expectation against **unshifted** labels; pins the
  absolute placement that shift-marginalization would otherwise let drift.

## 4. Critique and justification

**What's well-motivated:**

- **Frozen backbone + fp16 token cache** converts the problem from video training to
  sequence training: epochs are I/O-cheap, the 86M-param perception cost is paid once
  per video, and backbone choice becomes an empirical bake-off rather than a
  commitment (the head only assumes `5·D` input width).
- **HL-Gauss over scalar MSE regression**: distributional targets give dense,
  well-conditioned gradients, avoid regression-to-the-mean on ambiguous frames, and
  the expectation decode is smooth. The known trade-off is that expectation decoding
  of a *multimodal* predicted distribution can land between modes.
- **The periodicity branch is structural, not just a feature**: banded self-similarity
  is appearance-invariant by construction (cosine similarity of a slot to its
  neighbors), directly exposing stroke period/phase — the failure mode it targets is
  appearance-driven overfitting. Making it removable at the module level (not merely
  down-weighted) keeps the ablation honest: with it off, the transformer's *input*
  changes.
- **RoPE over learned absolute positions**: relative encoding matches the task
  (periodic structure is translation-invariant in time) and removes the `max_slots`
  hard cap; the learned-absolute path is retained as an explicit fallback.
- **Pre-norm + LayerNorm-only**: standard stability choice; BatchNorm is correctly
  avoided given padded variable-length sequences.
- **Softmin shift-marginalization** is a principled answer to label timing jitter —
  cheaper than DTW-style alignment, and the argmin histogram doubles as a dataset
  diagnostic for systematic lag.

**Weaknesses / open questions:**

- **The 5-token spatial pooling is a hard information bottleneck.** Mean-pooling a
  14×14 grid to full+quadrants discards fine spatial layout; motion confined to a
  small region is diluted. Attention pooling or a finer pyramid are unexplored here
  (though any change invalidates the token cache).
- **Full `[S,S]` similarity is O(S²)** in time and memory even though only a
  129-diagonal band is consumed. Fine at training clip lengths; wasteful for
  long-sequence inference.
- **Fixed crop box** assumes the action sits center-bottom; the cache metadata records
  the box and a full-frame variant exists for A/B, but the model itself has no spatial
  attention fallback if framing deviates.
- **Bidirectional attention means no streaming/causal inference** — the model is
  offline by design. A causal variant would need retraining and would lose the
  "look ahead to find the stroke peak" advantage.
- **Expectation decode + L_anchor(MSE) partially reintroduces** the mean-seeking
  behavior HL-Gauss avoids; the anchor weight matters (kept small relative to L_shift).

## 5. Tuning opportunities

- **Capacity:** `d_model` (384 → 256/512), `n_layers` (6 → 4/8), `mlp_ratio`. The
  trunk is 86% of parameters — the first knob for over/underfitting.
- **HL-Gauss shape:** `n_bins` (64) and σ (0.02) jointly set label smoothing; σ below
  bin width degenerates toward one-hot, larger σ blurs precision.
- **Shift set and τ:** widen shifts if the argmin histogram piles up at ±5 (indicates
  systematic dataset lag — fix the data first); τ trades soft averaging vs hard min.
- **Periodicity:** `max_lag` 64 slots ≈ 2.1 s at 30 Hz slots — must exceed the longest
  stroke period of interest; `periodicity_dim`; temperature init. The P1-B ablation
  (structural on/off) is the first-order experiment.
- **Warmup steps** (500), dropout (0.1), and the trainer-owned loss weights.
- **Backbone swap** (VideoMAE-B ssv2 vs V-JEPA 2 ViT-L vs VideoMAEv2) via linear-probe
  bake-off — the cache format and head are already agnostic.
- **Phase 2a LoRA**: unfreeze the last N backbone blocks via LoRA (r/alpha/dropout on
  qkv + attn proj + MLP linears) with gradient checkpointing, trading cache-reuse for
  task-adapted features; a merged checkpoint reloads through the same `load_backbone`
  path.
- **Crop-box jitter and color jitter** augmentations (already plumbed through
  `crop_resize_normalize`) — temporally consistent per clip by design.

## 6. Optimization opportunities

- **Vectorize the lag gather**: the Python loop over 128 `torch.diagonal` calls can
  become a single `F.unfold`/`as_strided` banded extraction; for long S, compute only
  the band (O(S·L)) instead of the full O(S²) matrix.
- **Vectorize `velocity_pearson_loss`**: currently a Python loop over the batch;
  masked batched centering/normalization removes it (kept simple deliberately — it's
  loss-side only, and correctness around the flat-guard is subtle).
- **`torch.compile`** on the head: small dense modules with static shapes per bucket —
  a good fusion candidate; the periodicity loop is the main graph-break risk (another
  reason to vectorize it).
- **SDPA already routes to fused/flash kernels**; the mask short-circuit keeps the
  unpadded path eligible. Keeping padding out of batches (length-bucketed sampling)
  preserves that.
- **Precision:** the head is small enough that bf16 autocast is nearly free; cosine
  similarity is already pinned to fp32 where it matters.
- **Inference:** token extraction dominates end-to-end cost (frozen backbone over all
  windows); `batch_windows` auto-scales by window size/resolution
  (`default_batch_windows`), and the fp16 cache means re-runs skip it entirely.
  The head itself can process very long S in one pass (RoPE has no length cap), memory
  permitting — chunked/sliding-window inference only needs to respect the ±64-slot
  periodicity context and attention's global receptive field.

## 7. Reproducibility notes

- Checkpoint configs round-trip through `DNX_CONFIG_KEYS` / `extract_dnx_config`, so a
  checkpoint fully determines head construction.
- The token cache is versioned (`format_version`, `format`) and refuses backbone-id
  mismatches on read; crop box, normalization, and feature fps are stored in the file.
- `use_rope=False` checkpoints are architecturally different (learned `abs_pos`,
  `max_slots` cap) — the flag is part of the config for exactly this reason.
