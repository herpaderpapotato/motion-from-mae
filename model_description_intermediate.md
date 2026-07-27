# Model Architecture — Intermediate Technical Overview

*Assumes basic ML knowledge: you know what a layer, an activation function, and a loss
function are. Source: [src/models/disposition_next.py](../src/models/disposition_next.py),
[src/training/dnx_losses.py](../src/training/dnx_losses.py),
[src/data/videomae_features.py](../src/data/videomae_features.py).*

## Task

Per-frame regression + classification over video: for each frame, predict a **position**
value in `[0, 1]` (a funscript-style motion curve) and an **activity** probability
(is motion happening). Videos run at 60 fps natively.

## Two-stage pipeline

### Stage 1 — Frozen VideoMAE-B feature extractor (inputs/outputs only)

A pre-trained VideoMAE-Base transformer (`MCG-NJU/videomae-base-finetuned-ssv2`) is
used as a **frozen** backbone — its weights are never updated.

- **Input:** 16-frame windows of video, cropped to the center-bottom region
  (x: 25–75%, y: 50–100% of the frame), resized to 224×224, and normalized. Shape
  `[B, 16, 3, 224, 224]`.
- **Output:** 1568 patch tokens of dimension 768 per window — the 16 frames are
  grouped into 8 two-frame "slots" (VideoMAE's tubelet size is 2), each covered by a
  14×14 spatial grid of tokens.

The 14×14 grid per slot is then **pooled to 5 tokens**: the mean of all patches, plus
the mean of each quadrant (TL/TR/BL/BR). This gives one token pack of shape `[5, 768]`
per 2-frame slot, cached to disk as float16 HDF5. The trained model never sees pixels —
only these cached packs. So for a clip of `S` slots the trainable model's input is
`[B, S, 5, 768]` plus a boolean validity mask for padding.

### Stage 2 — The DNX temporal head (the trainable model, ~12.4M parameters)

A small transformer that operates over the slot sequence:

1. **Slot projector.** The 5 tokens are concatenated (5×768 = 3840) and passed through
   `Linear(3840 → 384) → LayerNorm → GELU → Dropout(0.1)`, producing one 384-dim
   embedding per slot. GELU is the standard smooth ReLU-variant used in transformers;
   LayerNorm keeps activations well-scaled without depending on batch statistics
   (there is deliberately **no BatchNorm** anywhere — batch statistics are unreliable
   with variable-length, padded sequences).

2. **Periodicity branch.** Computes the cosine similarity between every pair of slot
   embeddings (`[S, S]` matrix, in float32 for numerical stability), then reads off
   128 diagonals — the similarity of each slot to the slot `lag` steps away, for lags
   −64…+63. Rhythmic motion produces periodic stripes in this matrix, so these
   "lag features" directly encode stroke period and phase in a way that doesn't depend
   on scene appearance. A validity mask for out-of-range lags is concatenated (256
   values total) and an MLP (`Linear → GELU → Linear → LayerNorm`) maps them to a
   128-dim feature. A learnable temperature (parameterized in log-space so it stays
   positive) scales the similarities. This branch exists because raw appearance
   features generalize poorly across scenes — periodicity is appearance-invariant.

3. **Fusion.** The 384-dim slot embedding and the 128-dim periodicity feature are
   concatenated and fused back to 384 dims (`Linear → LayerNorm`).

4. **Transformer trunk.** 6 pre-norm transformer blocks (d_model 384, 6 attention
   heads, MLP hidden 1536 with GELU, dropout 0.1). "Pre-norm" means LayerNorm is
   applied *before* attention/MLP inside each residual branch — the standard choice
   for stable training. Position information comes from **RoPE** (rotary position
   embeddings) applied to queries and keys, which encodes *relative* distance between
   slots rather than absolute index — appropriate because a stroke pattern means the
   same thing at minute 1 or minute 30. Padded slots are excluded via an attention
   mask. Full (bidirectional) attention lets every slot see the whole clip.

5. **Output heads.** After a final LayerNorm:
   - **Position head:** `Linear(384 → 2×64)` — each slot covers 2 frames, and each
     frame gets logits over **64 bins** spanning positions 0–1.
   - **Activity head:** `Linear(384 → 2)` — one logit per frame.

## Why bins instead of a single regression output? (HL-Gauss)

Rather than regressing a scalar with MSE, the model predicts a **distribution** over
64 position bins. The training target for a label `y` is a discretized Gaussian
(σ = 0.02) centered on `y`, and the loss is cross-entropy between the softmax of the
predicted logits and that target. At inference the position is decoded as the
expectation: `Σ softmax(logits)ₖ · center_k`. This "HL-Gauss" formulation gives
smoother gradients than one-hot classification, avoids MSE's tendency to regress to
the mean when uncertain, and lets nearby bins share credit.

## Loss functions (four terms, weighted by the trainer)

| Term | What it is | Why |
|---|---|---|
| **L_shift** (position) | HL-Gauss cross-entropy, evaluated at 11 label time-shifts (−5…+5 frames) and combined with a softmin (τ = 0.05) that mostly picks the best-matching shift | Human labels have small timing jitter; punishing exact-frame alignment teaches the model to hedge. The softmin says "be right at *some* nearby alignment." A 500-step warmup uses only shift 0 so the model can't exploit shift-shopping before it has learned anything. |
| **L_vel** (velocity) | `1 − Pearson correlation` between predicted and labeled frame-to-frame deltas, per sequence | Position CE can be minimized by a flat-ish curve of the right average height; correlating the *derivatives* forces the wave shape and timing to match. Flat/hold sequences contribute 0 (with gradient-graph connectivity preserved) so they don't produce a degenerate correlation. |
| **L_act** (activity) | Binary cross-entropy with logits on the activity head, with a class-balance `pos_weight` computed once over the whole training set | Hold/inactive frames outnumber active ones; the weight keeps the classifier from collapsing to the majority class. BCE-with-logits is used (never `log(sigmoid(x))`) for numerical safety. |
| **L_anchor** (anchor) | MSE between the decoded expectation and the **unshifted** labels | Shift-marginalization alone lets the absolute placement drift; a small MSE term anchors the curve to the true positions. |

All losses are masked so padded frames never contribute.

## Design summary

The frozen backbone converts pixels → cached semantic tokens once; the small trainable
head learns the temporal task on top. This separates expensive perception (86M frozen
parameters, run once per video) from cheap sequence reasoning (12.4M trained
parameters), making training fast and data-efficient, and allowing the same head design
to be re-used with different backbones (the head only assumes an input hidden size,
not a specific backbone).
