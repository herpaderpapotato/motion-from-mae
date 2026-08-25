"""Dense per-frame position track -> keyframes.

Savgol lowpass -> extrema seed -> greedy pchip refine -> device pass; the
pipeline benchmarked in motion_from_mae/scripts/benchmark_simplify.py (stages
10-13) and applied there by scripts/simplify_funscripts.py. Positions are in
0-100 funscript units and indices are frames of the dense grid.
"""

from __future__ import annotations

import bisect
import heapq

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import find_peaks, savgol_filter

# Extrema policy, in 0-100 position units.
EXTREMA_DISTANCE_S = 0.2
EXTREMA_PROMINENCE_FRAC = 0.1

# Device realism defaults: strokes below MIN_AMPLITUDE are not renderable, and
# most playback stacks degrade above ~1 point per MIN_GAP_MS.
MIN_AMPLITUDE = 5.0
MIN_GAP_MS = 55.0

SMOOTH_WINDOW_S = 0.1
MAX_ERR = 3.0

# Below this the seed/refine stages have nothing to work with.
MIN_FRAMES = 8


def smooth(y: np.ndarray, fps: float, window_s: float) -> np.ndarray:
    """Savitzky-Golay lowpass. Frame-scale jitter otherwise becomes fake extrema."""
    if window_s <= 0:
        return y.copy()
    win = int(round(window_s * fps))
    win = max(5, win + (win + 1) % 2)  # odd, >= 5
    if win >= len(y):
        return y.copy()
    return savgol_filter(y, win, 2)


def extrema_indices(y: np.ndarray, fps: float, prominence: float) -> np.ndarray:
    distance = max(1, int(round(EXTREMA_DISTANCE_S * fps)))
    peaks, _ = find_peaks(y, prominence=prominence, distance=distance)
    troughs, _ = find_peaks(-y, prominence=prominence, distance=distance)
    return np.unique(np.concatenate([peaks, troughs, [0, len(y) - 1]])).astype(int)


# A pchip segment depends only on the two knots either side of it, so a rebuild
# over a window of knots reproduces the global fit exactly outside its first and
# last segment. GUARD >= 2 gives that margin; the extra knot is slack.
_GUARD = 3


def _segment_errors(y: np.ndarray, knots: list[int], i_lo: int, i_hi: int
                    ) -> list[tuple[float, int]]:
    """Worst pchip error and its frame for segments i_lo..i_hi of `knots`."""
    sub = np.asarray(knots[max(0, i_lo - _GUARD): min(len(knots), i_hi + 2 + _GUARD)])
    fit = PchipInterpolator(sub.astype(np.float64), y[sub])
    a, b = knots[i_lo], knots[i_hi + 1]
    err = np.abs(fit(np.arange(a, b + 1, dtype=np.float64)) - y[a:b + 1])

    out = []
    for i in range(i_lo, i_hi + 1):
        lo, hi = knots[i] - a + 1, knots[i + 1] - a  # interior frames only
        if hi <= lo:
            out.append((0.0, -1))
            continue
        k = int(np.argmax(err[lo:hi]))
        out.append((float(err[lo + k]), a + lo + k))
    return out


def greedy_pchip_indices(y: np.ndarray, seed: np.ndarray, max_err: float,
                         max_points: int) -> np.ndarray:
    """Insert the worst-reconstructed frame until pchip error is within budget.

    The error metric is the one playback actually uses, so `max_err` is a real
    guarantee: no frame is ever further than this from the dense prediction.

    Segment errors are tracked in a heap and only the segments a new knot can
    perturb are rescored, instead of refitting the whole curve per insertion.
    """
    knots = sorted(set(seed.tolist()) | {0, len(y) - 1})
    if len(knots) < 2:
        return np.array(knots, dtype=int)

    heap: list[tuple[float, int, int, int]] = []
    version: dict[int, int] = {}

    def rescore(i_lo: int, i_hi: int) -> None:
        i_lo, i_hi = max(0, i_lo), min(len(knots) - 2, i_hi)
        if i_hi < i_lo:
            return
        for i, (err, frame) in zip(range(i_lo, i_hi + 1), _segment_errors(y, knots, i_lo, i_hi)):
            left = knots[i]
            v = version.get(left, 0) + 1
            version[left] = v
            if frame >= 0:
                heapq.heappush(heap, (-err, left, frame, v))

    rescore(0, len(knots) - 2)

    while len(knots) < max_points and heap:
        neg_err, left, frame, v = heapq.heappop(heap)
        if version.get(left) != v:  # segment has since been split or rescored
            continue
        if -neg_err <= max_err:
            break
        p = bisect.bisect_left(knots, frame)
        knots.insert(p, frame)
        rescore(p - _GUARD, p + _GUARD - 1)
    return np.array(knots, dtype=int)


def device_pass(y: np.ndarray, idx: np.ndarray, fps: float, min_amp: float,
                min_gap_ms: float) -> np.ndarray:
    """Drop unrenderable micro-strokes and points closer than the device can act."""
    idx = list(idx)

    # Amplitude: collapse a vertex whose excursion from both neighbours is tiny.
    # Dropping one can only expose its left neighbour, so rewind by one rather
    # than rescanning from the start.
    k = 1
    while k < len(idx) - 1:
        a, b, c = y[idx[k - 1]], y[idx[k]], y[idx[k + 1]]
        if abs(b - a) < min_amp and abs(b - c) < min_amp:
            del idx[k]
            k = max(1, k - 1)
        else:
            k += 1

    # Spacing: keep the point with the larger local excursion when too close.
    min_gap_frames = min_gap_ms * fps / 1000.0
    out = [idx[0]]
    for k in idx[1:]:
        if k - out[-1] < min_gap_frames and len(out) > 1:
            prev_amp = abs(y[out[-1]] - y[out[-2]])
            if abs(y[k] - y[out[-2]]) >= prev_amp:
                out[-1] = k
            continue
        out.append(k)
    if out[-1] != idx[-1]:
        out.append(idx[-1])
    return np.array(sorted(set(out)), dtype=int)


def simplify(dense: np.ndarray, fps: float, *, smooth_window_s: float = SMOOTH_WINDOW_S,
             prominence_frac: float = EXTREMA_PROMINENCE_FRAC, max_err: float = MAX_ERR,
             min_amp: float = MIN_AMPLITUDE, min_gap_ms: float = MIN_GAP_MS) -> np.ndarray:
    """Kept indices into `dense` (0-100 positions on a uniform `fps` grid)."""
    ref_smooth = smooth(dense, fps, smooth_window_s)
    p95, p5 = np.percentile(dense, 95), np.percentile(dense, 5)
    prominence = prominence_frac * max(p95 - p5, 1e-6)
    seed = extrema_indices(ref_smooth, fps, prominence)
    refined = greedy_pchip_indices(dense, seed, max_err, len(dense))
    return device_pass(dense, refined, fps, min_amp, min_gap_ms)


def reconstruction_stats(dense: np.ndarray, idx: np.ndarray) -> dict:
    """Error, in 0-100 position units, of the kept points against the dense track.

    Both interpolations are reported because they are both real: the greedy
    refine budgets pchip error, while most players draw straight lines between
    actions, which is the larger of the two.
    """
    x = np.arange(len(dense), dtype=np.float64)
    knots = idx.astype(np.float64)
    recon = {"pchip": PchipInterpolator(knots, dense[idx])(x),
             "linear": np.interp(x, knots, dense[idx])}
    return {
        "actions_raw": int(len(dense)),
        "actions_simplified": int(len(idx)),
        "kept_fraction": round(float(len(idx) / max(len(dense), 1)), 5),
        **{f"{name}_error": {"mean": round(float(np.abs(r - dense).mean()), 3),
                             "p95": round(float(np.percentile(np.abs(r - dense), 95)), 3),
                             "max": round(float(np.abs(r - dense).max()), 3)}
           for name, r in recon.items()},
    }
