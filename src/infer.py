"""Sliding-window head inference over pooled tokens, plus post-decode filters."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from src.disposition_next import DispositionNext
from src.hlgauss import HLGAUSS_MODE_RADIUS, hlgauss_decode_mode

CROP_SLOTS = 256
OVERLAP = 0.5
HOLD_GATE_ACTIVITY_THRESHOLD = 0.35
HOLD_GATE_MIN_RUN_S = 0.75
DECODE_MODES = ("expectation", "mode")

# Confidence scaling. The head is distributional (HL-Gauss over n_bins), so the
# shape of the blended bin distribution is an uncertainty signal for free. Both
# axes are monotone in uncertainty but UNCALIBRATED: they rank frames, they are
# not probabilities or error bars until checked against labelled frames.
#
# C1 spread: the distribution's std in position units. The floor is the sigma
# the head was trained against (dnx_losses.HLGAUSS_SIGMA) -- measured p1 over
# 36k frames of a real video is 0.021, so the sharpest frames do reach it. The
# ceiling is the std of a uniform distribution over [0, 1]: as spread out as
# knowing nothing at all. Measured spread over that same video was p25 0.140,
# p50 0.202, p95 0.279, which lands the median mid-scale.
CONF_SPREAD_FLOOR = 0.02
CONF_SPREAD_CEIL = 0.2887
# C2 agreement: |expectation - mode| decode gap, zero for any single symmetric
# peak, growing only when probability mass sits away from the peak -- the model
# split between two positions rather than merely vague. Ceiling is the measured
# p95 (0.252): by then the two decodes disagree about which stroke this is.
CONF_GAP_CEIL = 0.25


def decode_blended_positions(
    probs: np.ndarray, n_bins: int, decode: str = "expectation", radius: int = HLGAUSS_MODE_RADIUS,
) -> np.ndarray:
    """[T, n_bins] blended bin distribution -> positions in [0, 1]."""
    if decode not in DECODE_MODES:
        raise ValueError(f"unknown decode {decode!r}; expected one of {list(DECODE_MODES)}")
    if decode == "expectation":
        centers = (np.arange(n_bins) + 0.5) / n_bins
        return (probs * centers[None, :]).sum(axis=-1)
    return hlgauss_decode_mode(torch.from_numpy(probs), n_bins, radius=radius).numpy()


def distribution_confidence(
    probs: np.ndarray, n_bins: int, radius: int = HLGAUSS_MODE_RADIUS,
) -> tuple[np.ndarray, np.ndarray]:
    """[T, n_bins] blended distribution -> (spread_conf, agreement_conf) in [0, 1].

    Higher is more confident in both. See the CONF_* constants for the mapping;
    they are display scaling, not calibration.
    """
    centers = (np.arange(n_bins) + 0.5) / n_bins
    mean = (probs * centers[None, :]).sum(axis=-1)
    variance = (probs * (centers[None, :] - mean[:, None]) ** 2).sum(axis=-1)
    std = np.sqrt(np.clip(variance, 0.0, None))
    spread = 1.0 - (std - CONF_SPREAD_FLOOR) / (CONF_SPREAD_CEIL - CONF_SPREAD_FLOOR)

    mode = hlgauss_decode_mode(torch.from_numpy(probs), n_bins, radius=radius).numpy()
    agreement = 1.0 - np.abs(mean - mode) / CONF_GAP_CEIL

    return np.clip(spread, 0.0, 1.0), np.clip(agreement, 0.0, 1.0)


def sliding_window_predict_dnx(
    model: DispositionNext, tokens: np.ndarray, device: torch.device,
    crop_slots: int = CROP_SLOTS, overlap: float = OVERLAP,
    decode: str = "expectation", decode_radius: int = HLGAUSS_MODE_RADIUS,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Sliding crop_slots-slot windows blended with Bartlett weights in BIN
    PROBABILITY space (not decoded scalars), renormalised, then decoded.
    Activity is blended the same way in sigmoid space.

    Returns (position [T], activity [T], confidence), all in [0, 1],
    T = 2 * n_slots. `confidence` holds "spread" and "agreement" -- see
    `distribution_confidence`. They are read off the same blended distribution
    the position comes from, so they cost nothing extra to produce.
    """
    s = tokens.shape[0]
    n_bins = model.n_bins
    t_feat = s * 2
    stride = max(1, int(round(crop_slots * (1 - overlap))))

    prob_sum = np.zeros((t_feat, n_bins), dtype=np.float64)
    act_sum = np.zeros(t_feat, dtype=np.float64)
    weight_sum = np.zeros(t_feat, dtype=np.float64)

    if s <= crop_slots:
        starts = [0]
    else:
        starts = list(range(0, s - crop_slots + 1, stride))
        if starts[-1] + crop_slots < s:
            starts.append(s - crop_slots)

    bartlett_frame = np.bartlett(max(2, crop_slots * 2)).astype(np.float64) + 0.01

    with torch.no_grad():
        for start in starts:
            end = min(start + crop_slots, s)
            n_slots = end - start
            window = np.zeros((crop_slots, tokens.shape[1], tokens.shape[-1]), dtype=np.float32)
            window[:n_slots] = tokens[start:end]
            valid = np.zeros(crop_slots * 2, dtype=bool)
            valid[:n_slots * 2] = True

            tok_t = torch.from_numpy(window).unsqueeze(0).to(device, dtype=torch.float32)
            valid_t = torch.from_numpy(valid).unsqueeze(0).to(device, dtype=torch.bool)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = model(tok_t, valid_t)
            probs = F.softmax(out["position_logits"][0].float(), dim=-1).cpu().numpy()
            act = torch.sigmoid(out["activity_logits"][0]).float().cpu().numpy()

            n_frames = n_slots * 2
            w = bartlett_frame[:n_frames]
            t0 = start * 2
            prob_sum[t0:t0 + n_frames] += probs[:n_frames] * w[:, None]
            act_sum[t0:t0 + n_frames] += act[:n_frames] * w
            weight_sum[t0:t0 + n_frames] += w

    weight_sum = np.maximum(weight_sum, 1e-8)
    probs_blend = prob_sum / weight_sum[:, None]
    probs_blend = probs_blend / np.clip(probs_blend.sum(axis=-1, keepdims=True), 1e-12, None)
    position = decode_blended_positions(probs_blend, n_bins, decode, decode_radius)
    activity = act_sum / weight_sum
    spread, agreement = distribution_confidence(probs_blend, n_bins, decode_radius)
    return position, activity, {"spread": spread, "agreement": agreement}


def smooth_positions(position: np.ndarray, mode: str, window: int, polyorder: int = 2) -> np.ndarray:
    """Post-decode temporal smoothing. 'median' removes single-frame direction
    reversals at some cost in amplitude; 'savgol' is gentler."""
    if mode == "none" or window <= 1:
        return position
    if mode == "median":
        from scipy.signal import medfilt

        if window % 2 == 0:
            window += 1
        return medfilt(position.astype(np.float64), window)
    if mode == "savgol":
        from scipy.signal import savgol_filter

        if window % 2 == 0:
            window += 1
        if window <= polyorder:
            window = polyorder + 1 + (polyorder % 2)
        if len(position) < window:
            return position
        return np.clip(savgol_filter(position.astype(np.float64), window, polyorder), 0.0, 1.0)
    raise ValueError(f"Unknown smoothing mode: {mode}")


def apply_hold_gate(
    position: np.ndarray, activity: np.ndarray, fps: float,
    activity_threshold: float = HOLD_GATE_ACTIVITY_THRESHOLD, min_run_s: float = HOLD_GATE_MIN_RUN_S,
) -> np.ndarray:
    """Clamp position to its running median while activity stays below
    `activity_threshold` for at least `min_run_s`. The clamp value is the running
    median at the start of the run, held for its duration."""
    n = len(position)
    min_run = max(1, int(round(min_run_s * fps)))
    below = activity < activity_threshold
    gated = position.copy()

    running_median = np.array([np.median(position[max(0, i - min_run + 1):i + 1]) for i in range(n)])

    run_start = None
    for i in range(n + 1):
        is_below = i < n and below[i]
        if is_below and run_start is None:
            run_start = i
        elif not is_below and run_start is not None:
            if i - run_start >= min_run:
                gated[run_start:i] = running_median[run_start]
            run_start = None
    return gated
