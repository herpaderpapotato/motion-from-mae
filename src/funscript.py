"""Dense per-frame positions -> funscript JSON."""

from __future__ import annotations

import numpy as np


def predictions_to_funscript(
    positions: np.ndarray,
    fps: float,
    start_time: float = 0.0,
    metadata: dict[str, object] | None = None,
) -> dict:
    """One action per frame, no thinning."""
    actions = []
    for i, pos in enumerate(positions):
        at_ms = int(round((i / fps + start_time) * 1000.0))
        pos_int = max(0, min(100, int(round(float(pos) * 100.0))))
        actions.append({"at": at_ms, "pos": pos_int})
    funscript = {"version": "1.0", "inverted": False, "range": 100, "actions": actions}
    if metadata:
        funscript["metadata"] = metadata
    return funscript
