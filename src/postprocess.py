"""Wave-aware postprocessing for per-frame DispositionTCN predictions.

The pipeline follows the wave analysis document at a practical level:

1. Sanitize and mildly low-pass filter the 1D prediction signal.
2. Detect trough-to-trough cycles and normalize only the cycles that clear the
   configured amplitude and minimum-frequency thresholds.
3. Detect sustained slow gradients outside those oscillatory regions.
4. Stretch each gradient segment toward the full [0, 1] range.

The functions are written for direct use from prediction scripts and can also be
applied offline to saved arrays or funscripts resampled back to frame labels.
"""

from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt, find_peaks, savgol_filter


@dataclass(slots=True)
class WavePostprocessConfig:
    lowpass_cutoff_hz: float = 8.0
    lowpass_order: int = 2
    trough_prominence: float = 0.035
    trough_distance_seconds: float = 0.15
    min_cycle_amplitude: float = 0.3
    min_cycle_frequency_hz: float = 0.5
    gradient_smoothing: str = "savgol"
    gradient_window_seconds: float = 0.45
    gradient_polyorder: int = 2
    gradient_sigma_seconds: float = 0.12
    gradient_floor: float = 0.00075
    gradient_low_quantile: float = 0.25
    gradient_high_quantile: float = 0.9
    min_gradient_length_seconds: float = 0.6
    min_gradient_range: float = 0.08
    min_gradient_sign_balance: float = 0.65
    merge_gap_seconds: float = 0.15
    stretch_mode: str = "tanh"
    stretch_gain: float = 4.0
    chunk_seconds: float | None = 300.0
    chunk_overlap_seconds: float = 2.0

    def to_dict(self) -> dict[str, float | str | None]:
        return asdict(self)


def sanitize_signal(values: np.ndarray) -> np.ndarray:
    """Return a float32 1D signal with non-finite values linearly filled."""
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 1:
        raise ValueError(f"Expected a 1D prediction array, got shape {array.shape}")
    if array.size == 0:
        return array.copy()

    cleaned = array.astype(np.float32, copy=True)
    finite_mask = np.isfinite(cleaned)
    if finite_mask.all():
        return cleaned
    if not finite_mask.any():
        return np.zeros_like(cleaned)

    indices = np.arange(cleaned.size, dtype=np.float32)
    cleaned[~finite_mask] = np.interp(indices[~finite_mask], indices[finite_mask], cleaned[finite_mask])
    return cleaned


def apply_zero_phase_lowpass(
    values: np.ndarray,
    fps: float,
    cutoff_hz: float,
    order: int = 2,
) -> np.ndarray:
    """Apply a mild zero-phase Butterworth low-pass filter when viable."""
    if values.size < 4 or fps <= 0.0 or cutoff_hz <= 0.0:
        return values.astype(np.float32, copy=True)

    nyquist_hz = 0.5 * fps
    if cutoff_hz >= nyquist_hz:
        return values.astype(np.float32, copy=True)

    b, a = butter(max(1, int(order)), cutoff_hz / nyquist_hz, btype="lowpass")
    padlen = 3 * (max(len(a), len(b)) - 1)
    if values.size <= padlen:
        return values.astype(np.float32, copy=True)
    return np.asarray(filtfilt(b, a, values), dtype=np.float32)


def detect_troughs(
    values: np.ndarray,
    fps: float,
    prominence: float,
    min_distance_seconds: float,
) -> np.ndarray:
    """Detect local minima for trough-to-trough cycle segmentation."""
    if values.size < 3:
        return np.zeros((0,), dtype=np.int64)

    distance = max(1, int(round(max(0.0, min_distance_seconds) * fps)))
    kwargs: dict[str, int | float] = {"distance": distance}
    if prominence > 0.0:
        kwargs["prominence"] = float(prominence)
    troughs, _ = find_peaks(-np.asarray(values, dtype=np.float32), **kwargs)
    return troughs.astype(np.int64, copy=False)


def normalize_cycles(
    values: np.ndarray,
    reference: np.ndarray,
    trough_indices: np.ndarray,
    fps: float,
    min_cycle_amplitude: float,
    min_cycle_frequency_hz: float,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Normalize qualifying trough-to-trough cycles to the full [0, 1] range."""
    working = np.asarray(values, dtype=np.float32).copy()
    cycle_mask = np.zeros(working.shape, dtype=bool)
    cycle_segments: list[tuple[int, int]] = []
    if trough_indices.size < 2 or fps <= 0.0 or min_cycle_frequency_hz <= 0.0:
        return working, cycle_mask, cycle_segments

    max_period_samples = max(1, int(np.floor(fps / min_cycle_frequency_hz)))
    for start, end in zip(trough_indices[:-1], trough_indices[1:]):
        start_idx = int(start)
        end_idx = int(end)
        if end_idx <= start_idx + 1:
            continue

        period = end_idx - start_idx
        if period > max_period_samples:
            continue

        ref_segment = reference[start_idx:end_idx + 1]
        if ref_segment.size < 3:
            continue
        if float(np.ptp(ref_segment)) < float(min_cycle_amplitude):
            continue

        segment = working[start_idx:end_idx + 1]
        seg_min = float(segment.min())
        seg_range = float(segment.max() - seg_min)
        if seg_range <= 1e-6:
            continue

        working[start_idx:end_idx + 1] = (segment - seg_min) / seg_range
        cycle_mask[start_idx:end_idx + 1] = True
        cycle_segments.append((start_idx, end_idx + 1))

    return working, cycle_mask, cycle_segments


def smooth_gradient_reference(
    values: np.ndarray,
    fps: float,
    method: str,
    window_seconds: float,
    polyorder: int,
    sigma_seconds: float,
) -> np.ndarray:
    """Return a heavily smoothed copy used for gradient and plateau analysis."""
    mode = method.lower()
    if mode == "none" or values.size < 5:
        return values.astype(np.float32, copy=True)

    if mode == "gaussian":
        sigma = max(0.5, float(sigma_seconds) * fps)
        return np.asarray(gaussian_filter1d(values, sigma=sigma, mode="nearest"), dtype=np.float32)

    if mode != "savgol":
        raise ValueError(f"Unsupported gradient smoothing mode: {method}")

    window = max(5, int(round(max(0.0, window_seconds) * fps)))
    if window % 2 == 0:
        window += 1
    if window >= values.size:
        window = values.size - 1 if values.size % 2 == 0 else values.size
    if window < 5:
        return values.astype(np.float32, copy=True)

    poly = min(max(1, int(polyorder)), window - 2)
    return np.asarray(savgol_filter(values, window_length=window, polyorder=poly, mode="interp"), dtype=np.float32)


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    if mask.size == 0:
        return []

    padded = np.concatenate((np.array([False]), mask.astype(bool), np.array([False])))
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return [(int(start), int(end)) for start, end in zip(edges[::2], edges[1::2])]


def _merge_runs(runs: list[tuple[int, int]], max_gap: int) -> list[tuple[int, int]]:
    if not runs:
        return []
    merged = [runs[0]]
    for start, end in runs[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= max_gap:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return merged


def detect_gradient_segments(
    reference: np.ndarray,
    fps: float,
    exclude_mask: np.ndarray,
    config: WavePostprocessConfig,
) -> tuple[list[tuple[int, int]], np.ndarray, dict[str, float]]:
    """Find sustained slow-gradient segments that are distinct from oscillations."""
    if reference.size < 3:
        return [], np.zeros(reference.shape, dtype=bool), {
            "gradient_low_threshold": 0.0,
            "gradient_high_threshold": 0.0,
        }

    gradient = np.gradient(reference).astype(np.float32)
    abs_gradient = np.abs(gradient)
    eligible = ~np.asarray(exclude_mask, dtype=bool)
    eligible_values = abs_gradient[eligible]
    positive = eligible_values[eligible_values > 0.0]
    if positive.size == 0:
        return [], np.zeros(reference.shape, dtype=bool), {
            "gradient_low_threshold": 0.0,
            "gradient_high_threshold": 0.0,
        }

    low_threshold = max(float(config.gradient_floor), float(np.quantile(positive, config.gradient_low_quantile)))
    high_threshold = float(np.quantile(positive, config.gradient_high_quantile))
    if high_threshold < low_threshold:
        high_threshold = low_threshold

    candidate_mask = eligible & (abs_gradient >= low_threshold) & (abs_gradient <= high_threshold)
    raw_runs = _true_runs(candidate_mask)
    max_gap = max(0, int(round(max(0.0, config.merge_gap_seconds) * fps)))
    min_length = max(2, int(round(max(0.0, config.min_gradient_length_seconds) * fps)))
    merged_runs = _merge_runs(raw_runs, max_gap)

    kept_runs: list[tuple[int, int]] = []
    gradient_mask = np.zeros(reference.shape, dtype=bool)
    for start, end in merged_runs:
        if end - start < min_length:
            continue

        grad_segment = gradient[start:end]
        significant = np.abs(grad_segment) >= low_threshold
        if not np.any(significant):
            continue

        sign_balance = abs(float(np.mean(np.sign(grad_segment[significant]))))
        segment_range = float(np.ptp(reference[start:end]))
        if sign_balance < config.min_gradient_sign_balance:
            continue
        if segment_range < config.min_gradient_range:
            continue

        kept_runs.append((start, end))
        gradient_mask[start:end] = True

    return kept_runs, gradient_mask, {
        "gradient_low_threshold": low_threshold,
        "gradient_high_threshold": high_threshold,
    }


def stretch_gradient_segments(
    values: np.ndarray,
    segments: list[tuple[int, int]],
    mode: str,
    gain: float,
    min_segment_range: float,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Stretch each slow-gradient segment to stronger output extremes."""
    working = np.asarray(values, dtype=np.float32).copy()
    applied_segments: list[tuple[int, int]] = []

    for start, end in segments:
        segment = working[start:end]
        if segment.size < 2:
            continue

        seg_min = float(segment.min())
        seg_max = float(segment.max())
        seg_range = seg_max - seg_min
        if seg_range < float(min_segment_range):
            continue

        if mode == "linear":
            stretched = (segment - seg_min) / max(seg_range, 1e-6)
        elif mode == "tanh":
            center = float(segment.mean())
            spread = float(segment.std())
            if spread <= 1e-6:
                spread = max(seg_range / 6.0, 1e-6)
            stretched = 0.5 * (np.tanh(float(gain) * (segment - center) / spread) + 1.0)
        else:
            raise ValueError(f"Unsupported stretch mode: {mode}")

        working[start:end] = np.asarray(stretched, dtype=np.float32)
        applied_segments.append((start, end))

    return working, applied_segments


def _single_pass_postprocess(
    values: np.ndarray,
    fps: float,
    config: WavePostprocessConfig,
) -> tuple[np.ndarray, dict[str, object]]:
    cleaned = sanitize_signal(values)
    filtered = apply_zero_phase_lowpass(cleaned, fps, config.lowpass_cutoff_hz, config.lowpass_order)
    trough_indices = detect_troughs(filtered, fps, config.trough_prominence, config.trough_distance_seconds)

    normalized, cycle_mask, cycle_segments = normalize_cycles(
        cleaned,
        filtered,
        trough_indices,
        fps,
        config.min_cycle_amplitude,
        config.min_cycle_frequency_hz,
    )

    gradient_reference = smooth_gradient_reference(
        filtered,
        fps,
        config.gradient_smoothing,
        config.gradient_window_seconds,
        config.gradient_polyorder,
        config.gradient_sigma_seconds,
    )
    gradient_segments, gradient_mask, gradient_thresholds = detect_gradient_segments(
        gradient_reference,
        fps,
        cycle_mask,
        config,
    )

    stretched, applied_gradient_segments = stretch_gradient_segments(
        normalized,
        gradient_segments,
        mode=config.stretch_mode.lower(),
        gain=config.stretch_gain,
        min_segment_range=config.min_gradient_range,
    )

    clipped = np.clip(stretched, 0.0, 1.0).astype(np.float32, copy=False)
    stats: dict[str, object] = {
        "num_samples": int(cleaned.size),
        "num_troughs": int(trough_indices.size),
        "num_normalized_cycles": int(len(cycle_segments)),
        "num_gradient_segments": int(len(applied_gradient_segments)),
        "num_cycle_samples": int(cycle_mask.sum()),
        "num_gradient_samples": int(gradient_mask.sum()),
        "num_clipped_samples": int(np.count_nonzero((stretched < 0.0) | (stretched > 1.0))),
        "cycle_segments": cycle_segments,
        "gradient_segments": applied_gradient_segments,
        **gradient_thresholds,
    }
    return clipped, stats


def _chunk_weights(length: int, overlap: int, fade_in: bool, fade_out: bool) -> np.ndarray:
    weights = np.ones(length, dtype=np.float32)
    if overlap <= 0 or length <= 1:
        return weights

    ramp_len = min(overlap, length)
    ramp = np.linspace(0.0, 1.0, ramp_len + 2, dtype=np.float32)[1:-1]
    if fade_in:
        weights[:ramp_len] *= ramp
    if fade_out:
        weights[-ramp_len:] *= ramp[::-1]
    return weights


def postprocess_predictions(
    predictions: np.ndarray,
    fps: float,
    config: WavePostprocessConfig | None = None,
    return_stats: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    """Apply wave-aware postprocessing to a 1D prediction sequence."""
    cfg = config or WavePostprocessConfig()
    values = np.asarray(predictions, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError(f"Expected a 1D prediction array, got shape {values.shape}")
    if values.size == 0:
        empty = values.astype(np.float32, copy=True)
        stats = {
            "num_samples": 0,
            "num_chunks": 0,
            "num_normalized_cycles": 0,
            "num_gradient_segments": 0,
            "config": cfg.to_dict(),
        }
        return (empty, stats) if return_stats else empty

    chunk_frames = 0
    overlap_frames = 0
    if cfg.chunk_seconds is not None and cfg.chunk_seconds > 0.0 and fps > 0.0:
        chunk_frames = int(round(cfg.chunk_seconds * fps))
        overlap_frames = int(round(max(0.0, cfg.chunk_overlap_seconds) * fps))

    if chunk_frames <= 0 or values.size <= chunk_frames or chunk_frames <= overlap_frames + 2:
        processed, stats = _single_pass_postprocess(values, fps, cfg)
        stats["num_chunks"] = 1
        stats["config"] = cfg.to_dict()
        return (processed, stats) if return_stats else processed

    step = max(1, chunk_frames - overlap_frames)
    blended = np.zeros(values.shape, dtype=np.float32)
    weights = np.zeros(values.shape, dtype=np.float32)
    aggregate_stats: dict[str, object] = {
        "num_samples": int(values.size),
        "num_chunks": 0,
        "num_troughs": 0,
        "num_normalized_cycles": 0,
        "num_gradient_segments": 0,
        "num_cycle_samples": 0,
        "num_gradient_samples": 0,
        "num_clipped_samples": 0,
        "gradient_low_threshold": 0.0,
        "gradient_high_threshold": 0.0,
        "cycle_segments": [],
        "gradient_segments": [],
        "config": cfg.to_dict(),
    }

    for start in range(0, values.size, step):
        end = min(values.size, start + chunk_frames)
        chunk = values[start:end]
        processed_chunk, chunk_stats = _single_pass_postprocess(chunk, fps, cfg)
        taper = _chunk_weights(
            len(chunk),
            overlap_frames,
            fade_in=start > 0,
            fade_out=end < values.size,
        )
        blended[start:end] += processed_chunk * taper
        weights[start:end] += taper
        aggregate_stats["num_chunks"] = int(aggregate_stats["num_chunks"]) + 1
        aggregate_stats["num_troughs"] = int(aggregate_stats["num_troughs"]) + int(chunk_stats["num_troughs"])
        aggregate_stats["num_normalized_cycles"] = int(aggregate_stats["num_normalized_cycles"]) + int(chunk_stats["num_normalized_cycles"])
        aggregate_stats["num_gradient_segments"] = int(aggregate_stats["num_gradient_segments"]) + int(chunk_stats["num_gradient_segments"])
        aggregate_stats["num_cycle_samples"] = int(aggregate_stats["num_cycle_samples"]) + int(chunk_stats["num_cycle_samples"])
        aggregate_stats["num_gradient_samples"] = int(aggregate_stats["num_gradient_samples"]) + int(chunk_stats["num_gradient_samples"])
        aggregate_stats["num_clipped_samples"] = int(aggregate_stats["num_clipped_samples"]) + int(chunk_stats["num_clipped_samples"])
        aggregate_stats["gradient_low_threshold"] = max(
            float(aggregate_stats["gradient_low_threshold"]),
            float(chunk_stats["gradient_low_threshold"]),
        )
        aggregate_stats["gradient_high_threshold"] = max(
            float(aggregate_stats["gradient_high_threshold"]),
            float(chunk_stats["gradient_high_threshold"]),
        )

        aggregate_stats["cycle_segments"].extend(
            (start + seg_start, start + seg_end) for seg_start, seg_end in chunk_stats["cycle_segments"]
        )
        aggregate_stats["gradient_segments"].extend(
            (start + seg_start, start + seg_end) for seg_start, seg_end in chunk_stats["gradient_segments"]
        )

        if end >= values.size:
            break

    safe_weights = np.maximum(weights, 1e-6)
    processed = np.clip(blended / safe_weights, 0.0, 1.0).astype(np.float32, copy=False)
    return (processed, aggregate_stats) if return_stats else processed


__all__ = [
    "WavePostprocessConfig",
    "apply_zero_phase_lowpass",
    "detect_gradient_segments",
    "detect_troughs",
    "normalize_cycles",
    "postprocess_predictions",
    "sanitize_signal",
    "smooth_gradient_reference",
    "stretch_gradient_segments",
]