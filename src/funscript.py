"""Dense per-frame positions -> funscript JSON."""

from __future__ import annotations

import numpy as np


def predictions_to_funscript(
    positions: np.ndarray,
    fps: float,
    start_time: float = 0.0,
    metadata: dict[str, object] | None = None,
    frame_times: np.ndarray | None = None,
    axes: dict[str, dict] | None = None,
) -> dict:
    """One action per frame, no thinning.

    `frame_times` is the source's own per-frame presentation time in seconds
    (absolute, already including start_time). It beats the synthesised i/fps
    grid whenever a container's nominal rate disagrees with its real cadence --
    measured 59.9401 declared vs 59.9297 actual on one 50-minute source, which
    is half a second of drift by the end.

    `axes` maps an axis id ("C1", "R1", ...) to {"values": array in [0, 1],
    "metadata": dict}, written as the multi-axis `axes` list alongside the main
    track. Extra axes share the main track's timestamps, so they must be the
    same length as `positions`; the file is bumped to version 1.1 when any are
    present.
    """
    times_ms = [
        int(round((float(frame_times[i]) if frame_times is not None
                   else (i / fps + start_time)) * 1000.0))
        for i in range(len(positions))
    ]

    def to_actions(values: np.ndarray) -> list[dict]:
        return [{"at": times_ms[i], "pos": max(0, min(100, int(round(float(v) * 100.0))))}
                for i, v in enumerate(values)]

    funscript = {"version": "1.0", "inverted": False, "range": 100,
                 "actions": to_actions(positions)}
    if axes:
        for axis_id, axis in axes.items():
            if len(axis["values"]) != len(positions):
                raise ValueError(
                    f"axis {axis_id} has {len(axis['values'])} values but the main track has "
                    f"{len(positions)}; extra axes reuse the main track's timestamps")
        funscript["version"] = "1.1"
        funscript["axes"] = [
            {"id": axis_id, "metadata": axis.get("metadata", {}),
             "actions": to_actions(axis["values"])}
            for axis_id, axis in axes.items()
        ]
    if metadata:
        funscript["metadata"] = metadata
    return funscript
