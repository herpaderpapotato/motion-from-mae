"""Dense per-frame positions -> funscript JSON."""

from __future__ import annotations

import numpy as np


def predictions_to_funscript(
    positions: np.ndarray,
    fps: float,
    start_time: float = 0.0,
    metadata: dict[str, object] | None = None,
    frame_times: np.ndarray | None = None,
) -> dict:
    """One action per frame, no thinning.

    `frame_times` is the source's own per-frame presentation time in seconds
    (absolute, already including start_time). It beats the synthesised i/fps
    grid whenever a container's nominal rate disagrees with its real cadence --
    measured 59.9401 declared vs 59.9297 actual on one 50-minute source, which
    is half a second of drift by the end.
    """
    actions = []
    for i, pos in enumerate(positions):
        at_s = float(frame_times[i]) if frame_times is not None else (i / fps + start_time)
        at_ms = int(round(at_s * 1000.0))
        pos_int = max(0, min(100, int(round(float(pos) * 100.0))))
        actions.append({"at": at_ms, "pos": pos_int})
    funscript = {"version": "1.0", "inverted": False, "range": 100, "actions": actions}
    if metadata:
        funscript["metadata"] = metadata
    return funscript
