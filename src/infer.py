"""Sliding-window head inference over pooled tokens, plus the hold gate and
confidence axes derived from the decoded distribution."""

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
# shape of the blended bin distribution is an uncertainty signal for free. All
# three axes are monotone in uncertainty but UNCALIBRATED: they rank frames
# within a video, they are not error bars until checked against labelled frames.
#
# C1 anomaly: raw distribution spread is NOT usable directly -- it scales with
# stroke speed (measured corr +0.30 against |velocity|, mean spread 0.130 at the
# slowest quintile vs 0.218 at the fastest), so it peaks at every turnaround and
# says more about the stroke than about the prediction. Dividing by speed
# over-corrects rather than fixing it: spread behaves like A + B*|v|, so the
# ratio is dominated by A/|v| wherever the stroke slows, and the measured
# correlation came out WORSE (+0.69, merely inverted). Regressing spread on
# speed and keeping the residual removes the term properly -- measured -0.06.
# So C1 reads "vaguer than this video's own strokes at this speed usually are",
# which is relative to the video; C2 carries the absolute level.
# The fitted slope ranges 0.2-7.2 across chunks of one video, far too
# content-dependent to hardcode, so it is fitted per run.
CONF_RESIDUAL_HALF_RANGE = 0.085  # measured |p5| / p95 of the residual
# C2 agreement: |expectation - mode| decode gap, zero for any single symmetric
# peak, growing only when probability mass sits away from the peak -- the model
# split between two positions rather than merely vague. Measured independent of
# speed (corr -0.07). Ceiling is the p95 of the smoothed gap.
CONF_GAP_CEIL = 0.25
# Both per-frame axes are differences of noisy per-frame decodes; a ~150 ms
# window at 60 fps takes the jitter out without blurring a stroke (median stroke
# is 15 frames).
CONF_SMOOTH_FRAMES = 9
# C3 aggregates per stroke, the unit review actually happens in. Segmentation
# reuses the wave postprocessor's tuned turning-point detection.
STROKE_PROMINENCE = 0.035
STROKE_MIN_DISTANCE_S = 0.15


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


def blend_moments(
    probs: np.ndarray, n_bins: int, radius: int = HLGAUSS_MODE_RADIUS,
) -> dict[str, np.ndarray]:
    """[T, n_bins] blended distribution -> raw uncertainty moments per frame.

    "spread_std" is the distribution's std in position units, "mode_gap" the
    |expectation - mode| decode disagreement. Unscaled: `confidence_axes` turns
    them into display axes, since that needs the final position track too.
    """
    centers = (np.arange(n_bins) + 0.5) / n_bins
    mean = (probs * centers[None, :]).sum(axis=-1)
    variance = (probs * (centers[None, :] - mean[:, None]) ** 2).sum(axis=-1)
    mode = hlgauss_decode_mode(torch.from_numpy(probs), n_bins, radius=radius).numpy()
    return {"spread_std": np.sqrt(np.clip(variance, 0.0, None)),
            "mode_gap": np.abs(mean - mode)}


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average that keeps its edges: the divisor is the count of
    samples actually in the window, not the nominal width."""
    window = min(int(window), len(values))
    if window < 2:
        return values.astype(np.float64)
    kernel = np.ones(window)
    counts = np.convolve(np.ones(len(values)), kernel, mode="same")
    return np.convolve(values.astype(np.float64), kernel, mode="same") / counts


def stroke_spans(position: np.ndarray, fps: float) -> list[tuple[int, int]]:
    """[start, end) spans between successive turning points of the position track."""
    from src.postprocess import detect_troughs

    if len(position) < 3:
        return [(0, len(position))]
    troughs = detect_troughs(position, fps, STROKE_PROMINENCE, STROKE_MIN_DISTANCE_S)
    peaks = detect_troughs(-position, fps, STROKE_PROMINENCE, STROKE_MIN_DISTANCE_S)
    edges = np.unique(np.concatenate([[0], troughs, peaks, [len(position)]]))
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def confidence_axes(
    position: np.ndarray, moments: dict[str, np.ndarray], fps: float,
) -> dict[str, np.ndarray]:
    """Final position track + raw moments -> C1/C2/C3 in [0, 1], higher = more
    confident.

    Takes the position that will actually be written, so C3's stroke boundaries
    are the ones visible in the script.
    """
    n = len(position)
    if n == 0:
        return {k: np.zeros(0) for k in ("C1", "C2", "C3")}

    speed = _moving_average(np.abs(np.gradient(position)) if n > 1 else np.zeros(n),
                            CONF_SMOOTH_FRAMES)
    spread = moments["spread_std"]
    if n >= 32 and speed.std() > 1e-9:
        slope, intercept = np.polyfit(speed, spread, 1)
        expected = slope * speed + intercept
    else:  # too short to fit a trend; fall back to the plain level
        expected = np.full(n, spread.mean())
    residual = _moving_average(spread - expected, CONF_SMOOTH_FRAMES)

    # 50 = as sharp as this video's strokes usually are at this speed.
    c1 = np.clip(0.5 - residual / (2.0 * CONF_RESIDUAL_HALF_RANGE), 0.0, 1.0)
    c2 = np.clip(1.0 - _moving_average(moments["mode_gap"], CONF_SMOOTH_FRAMES) / CONF_GAP_CEIL,
                 0.0, 1.0)

    # A stroke is only as trustworthy as its weaker signal; medians inside the
    # stroke so one bad frame does not condemn it.
    c3 = np.empty(n)
    for start, end in stroke_spans(position, fps):
        c3[start:end] = min(np.median(c1[start:end]), np.median(c2[start:end]))
    return {"C1": c1, "C2": c2, "C3": c3}


def sliding_window_predict_dnx(
    model: DispositionNext, tokens: np.ndarray, device: torch.device,
    crop_slots: int = CROP_SLOTS, overlap: float = OVERLAP,
    decode: str = "expectation", decode_radius: int = HLGAUSS_MODE_RADIUS,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Sliding crop_slots-slot windows blended with Bartlett weights in BIN
    PROBABILITY space (not decoded scalars), renormalised, then decoded.
    Activity is blended the same way in sigmoid space.

    Returns (position [T], activity [T], moments), T = frames_per_slot * n_slots. `moments`
    holds the raw per-frame uncertainty of the same blended distribution the
    position is decoded from (see `blend_moments`), so it costs nothing extra;
    `confidence_axes` scales it for display.
    """
    s = tokens.shape[0]
    n_bins = model.n_bins
    fps = model.frames_per_slot
    t_feat = s * fps
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

    bartlett_frame = np.bartlett(max(2, crop_slots * fps)).astype(np.float64) + 0.01

    with torch.no_grad():
        for start in starts:
            end = min(start + crop_slots, s)
            n_slots = end - start
            window = np.zeros((crop_slots, tokens.shape[1], tokens.shape[-1]), dtype=np.float32)
            window[:n_slots] = tokens[start:end]
            valid = np.zeros(crop_slots * fps, dtype=bool)
            valid[:n_slots * fps] = True

            tok_t = torch.from_numpy(window).unsqueeze(0).to(device, dtype=torch.float32)
            valid_t = torch.from_numpy(valid).unsqueeze(0).to(device, dtype=torch.bool)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = model(tok_t, valid_t)
            probs = F.softmax(out["position_logits"][0].float(), dim=-1).cpu().numpy()
            act = torch.sigmoid(out["activity_logits"][0]).float().cpu().numpy()

            n_frames = n_slots * fps
            w = bartlett_frame[:n_frames]
            t0 = start * fps
            prob_sum[t0:t0 + n_frames] += probs[:n_frames] * w[:, None]
            act_sum[t0:t0 + n_frames] += act[:n_frames] * w
            weight_sum[t0:t0 + n_frames] += w

    weight_sum = np.maximum(weight_sum, 1e-8)
    probs_blend = prob_sum / weight_sum[:, None]
    probs_blend = probs_blend / np.clip(probs_blend.sum(axis=-1, keepdims=True), 1e-12, None)
    position = decode_blended_positions(probs_blend, n_bins, decode, decode_radius)
    activity = act_sum / weight_sum
    return position, activity, blend_moments(probs_blend, n_bins, decode_radius)


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
