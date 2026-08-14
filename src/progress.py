"""Console feedback helpers: named timed steps, quiet Hugging Face cache lookups.

Several phases of a run (hub resolution, container indexing, ffmpeg transcode,
token readback) take seconds to minutes with nothing on screen. `step` names the
wait while it is happening and reports what it cost.
"""

from __future__ import annotations

import time
from contextlib import contextmanager


class _Step:
    """Handle yielded by `step`; `note` adds text to the completion line."""

    def __init__(self) -> None:
        self.notes: list[str] = []

    def note(self, message: str) -> None:
        self.notes.append(message)


@contextmanager
def step(message: str, enabled: bool = True, indent: str = "  "):
    """Print `message ...` up front, then the result and elapsed time when done."""
    handle = _Step()
    if not enabled:
        yield handle
        return
    print(f"{indent}{message} ... ", end="", flush=True)
    t0 = time.perf_counter()
    try:
        yield handle
    except BaseException:
        print("failed", flush=True)
        raise
    print(f"{' '.join(handle.notes) or 'ok'} ({time.perf_counter() - t0:.1f}s)", flush=True)


def hub_fetch(fetch, what: str):
    """Resolve an HF repo through `fetch(local_files_only=...)`.

    A cache hit stays silent -- hub_hub's own "Fetching N files" bars fire on
    every run and say nothing about a download that isn't happening. A real
    download is announced and keeps its progress bars.
    """
    from huggingface_hub.utils import (
        are_progress_bars_disabled,
        disable_progress_bars,
        enable_progress_bars,
    )

    was_disabled = are_progress_bars_disabled()
    disable_progress_bars()
    try:
        return fetch(local_files_only=True)
    except Exception:
        pass
    finally:
        if not was_disabled:
            enable_progress_bars()
    print(f"Downloading {what} from Hugging Face (first run only) ...", flush=True)
    return fetch(local_files_only=False)


def human_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"
    return f"{int(seconds // 3600)}h{int(seconds % 3600) // 60:02d}m"
