"""Hugging Face resolution that stays current without narrating a no-op.

Every run asks the hub which revision a repo is on, so a re-published checkpoint
is actually picked up instead of being served from the cache forever. The check
is one API call and prints nothing when the cache already matches; only a real
download is announced. Unreachable hub falls back to the cache with a warning,
and HF_HUB_OFFLINE (or --offline) skips the check entirely.
"""

from __future__ import annotations

import os
from pathlib import Path

REVISION_CHECK_TIMEOUT_S = 10.0
OFFLINE_ENV = "HF_HUB_OFFLINE"


def is_offline() -> bool:
    return os.environ.get(OFFLINE_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def set_offline() -> None:
    os.environ[OFFLINE_ENV] = "1"


def hub_revision(path: Path) -> str | None:
    """The commit sha a cached hub path sits under, or None for a local file.

    Hub cache layout is <repo>/snapshots/<sha>/[file]; both a snapshot dir and a
    single downloaded file inside one are resolved.
    """
    path = Path(path)
    for candidate in (path, path.parent):
        if candidate.parent.name == "snapshots":
            return candidate.name
    return None


def _quiet_fetch(fetch, **kwargs):
    """`fetch` with hub_hub's own progress bars muted (they fire on cache hits too)."""
    from huggingface_hub.utils import (
        are_progress_bars_disabled,
        disable_progress_bars,
        enable_progress_bars,
    )

    was_disabled = are_progress_bars_disabled()
    disable_progress_bars()
    try:
        return fetch(**kwargs)
    finally:
        if not was_disabled:
            enable_progress_bars()


def _remote_revision(repo_id: str) -> str:
    from huggingface_hub import HfApi

    return HfApi().repo_info(str(repo_id), timeout=REVISION_CHECK_TIMEOUT_S).sha


def resolve(fetch, repo_id: str, what: str, update_note: str = "",
            revision: str | None = None) -> Path:
    """Resolve `repo_id` to a local path through `fetch(local_files_only=, revision=)`.

    Returns the cached path when it is already the hub's current revision,
    otherwise downloads that revision. `fetch` must accept `local_files_only`
    and `revision` (both `snapshot_download` and `hf_hub_download` do).

    An explicit `revision` pins the run to that commit/tag and skips the
    freshness check -- a pin that could still be updated out from under you
    would not be a pin.
    """
    if revision:
        try:
            return Path(_quiet_fetch(fetch, local_files_only=True, revision=revision))
        except Exception:
            if is_offline():
                raise
        print(f"Downloading {what} @ {revision} from Hugging Face ...", flush=True)
        return Path(fetch(local_files_only=False, revision=revision))

    try:
        cached = Path(_quiet_fetch(fetch, local_files_only=True))
    except Exception:
        cached = None
    cached_rev = hub_revision(cached) if cached is not None else None

    if is_offline():
        if cached is None:
            raise RuntimeError(f"{what} is not in the local HF cache and {OFFLINE_ENV} is set")
        return cached

    try:
        remote_rev = _remote_revision(repo_id)
    except Exception as exc:
        if cached is None:
            raise  # no cache to fall back on: the hub error IS the real error
        print(f"  [warn] could not check {repo_id} for updates ({type(exc).__name__}); "
              f"using the cached revision {(cached_rev or 'local')[:7]}")
        return cached

    if cached is not None and cached_rev == remote_rev:
        return cached

    if cached_rev:
        print(f"Updating {what}: {cached_rev[:7]} -> {remote_rev[:7]}"
              + (f" {update_note}" if update_note else ""), flush=True)
    else:
        print(f"Downloading {what} @ {remote_rev[:7]} from Hugging Face ...", flush=True)
    return Path(fetch(local_files_only=False, revision=remote_rev))
