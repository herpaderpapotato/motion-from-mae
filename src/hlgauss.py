"""HL-Gauss bin decoding (inference half of the training repo's dnx_losses)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

HLGAUSS_N_BINS = 64

# Half-width in bins of the mode-local decode window; 0 is a plain argmax.
HLGAUSS_MODE_RADIUS = 2


def hlgauss_bin_centers(n_bins: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return (torch.arange(n_bins, device=device, dtype=dtype) + 0.5) / n_bins


def hlgauss_decode(logits: torch.Tensor, n_bins: int) -> torch.Tensor:
    """Expectation decode: p = sum_k softmax(logits)_k * c_k."""
    probs = F.softmax(logits, dim=-1)
    centers = hlgauss_bin_centers(n_bins, logits.device, logits.dtype)
    return (probs * centers).sum(dim=-1)


def hlgauss_decode_mode(
    probs: torch.Tensor, n_bins: int, radius: int = HLGAUSS_MODE_RADIUS,
) -> torch.Tensor:
    """Expectation over the +/-radius bins around the peak, renormalised.

    `probs` are normalised probabilities, not logits.
    """
    centers = hlgauss_bin_centers(n_bins, probs.device, probs.dtype)
    peak = probs.argmax(dim=-1, keepdim=True)
    bins = torch.arange(n_bins, device=probs.device).expand_as(probs)
    windowed = probs * ((bins - peak).abs() <= radius)
    return (windowed * centers).sum(dim=-1) / windowed.sum(dim=-1).clamp_min(1e-12)
