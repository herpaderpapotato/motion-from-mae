"""Resumable, content-identified token cache for --video extraction.

Keyed by (video file identity, backbone identity, VR-crop/framing/pooling params)
and written incrementally, so a re-run reuses finished work and a crash loses at
most the in-flight chunk. Filenames match the training repo's, so caches built
there are readable here.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np

log = logging.getLogger(__name__)

CACHE_FORMAT_VERSION = 1
DEFAULT_CACHE_DIR = Path("data/video_token_cache")


def _fingerprint_video_file(video_path: Path) -> str:
    """Cheap identity fingerprint: resolved path + size + mtime, NOT a full
    content hash/CRC -- hashing a many-GB / hours-long video file would
    itself take minutes, defeating the point of a fast resumable cache.
    Catches the common "different file" cases (replaced/re-encoded/different
    video); does not catch a byte-identical file silently touched to a new
    mtime (rare in practice, and the cache would just be needlessly rebuilt
    in that case -- a correctness-safe failure mode, not a stale-cache risk).
    """
    stat = video_path.stat()
    raw = f"{video_path.resolve()}|{stat.st_size}|{int(stat.st_mtime)}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _fingerprint_backbone(geometry: Any) -> str:
    """Identity for the backbone that produced (or will produce) the tokens.
    HF hub checkpoints have a real revision hash already (see `load_backbone`);
    local merged backbones (Phase 2a/2b LoRA-merged artifacts) don't, so fall
    back to the weight file's size+mtime.

    Both local layouts are handled: a VideoMAEv2 checkpoint DIR holding
    model.safetensors, and a V-JEPA 2.1 `.pt` where backbone_id IS the weight
    file. Without the second case a re-merge written to the same path keeps the
    old fingerprint, and the cache serves tokens from the previous weights."""
    if geometry.backbone_revision:
        raw = f"{geometry.backbone_id}|{geometry.backbone_revision}"
    else:
        source = Path(geometry.backbone_id)
        weight_file = source if source.is_file() else source / "model.safetensors"
        if weight_file.exists():
            wstat = weight_file.stat()
            raw = f"{geometry.backbone_id}|{wstat.st_size}|{int(wstat.st_mtime)}"
        else:
            raw = str(geometry.backbone_id)
    # A non-default 2.1 window produces genuinely different features from the
    # same weights. Empty for the default, so no existing cache is orphaned.
    from src.backbone import window_cache_tag

    raw += window_cache_tag(getattr(geometry, "window", None), getattr(geometry, "family", None))
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def cache_path_for_video(
    video_path: Path,
    geometry: Any,
    vr_mode: bool,
    sbs_crop: str,
    start_frame: int,
    end_frame: int,
    cache_dir: Path | None = None,
    crop_box: tuple[float, float, float, float] | None = None,
    pooling: str | None = None,
    preprocess: bool = False,
) -> Path:
    """Cache identity includes the requested [start_frame, end_frame) source
    range, so a whole-video run (the common case -- no --start/--duration)
    gets one cache entry that later whole-video runs naturally reuse/resume,
    while a run scoped to a snippet (--start/--duration) gets its own
    (smaller) entry rather than forcing every run to extract the full video
    regardless of what was actually asked for.
    """
    cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    video_fp = _fingerprint_video_file(video_path)
    backbone_fp = _fingerprint_backbone(geometry)
    vr_tag = f"vr{sbs_crop[0]}" if vr_mode else "novr"
    # Prediction always decodes every native source frame now (no canonical-fps
    # resampling); the tag is kept constant rather than removed so cache
    # filenames stay a stable, parseable shape.
    fps_tag = "native"
    range_tag = f"f{start_frame}-{end_frame}"
    # Framing changes the tokens (doc03 centre-bottom crop vs the whole eye vs a
    # --crop-box of your own), so it must be part of the cache identity --
    # otherwise runs with different framing on the same video/backbone would
    # silently share (and corrupt) one cache entry.
    from src.backbone import crop_box_tag

    view_tag = crop_box_tag(crop_box)
    # Constant: this repo always decodes at native resolution. Kept in the name
    # so cache files stay interchangeable with the training repo's.
    size_tag = "srcnative"
    # The spatial pooling layout changes the token pack's shape and content, so
    # it is part of the cache identity for the same reason view/size are. The
    # default layout contributes no tag, so every cache file written before
    # pooling was configurable keeps its exact name and stays reusable.
    from src.backbone import DEFAULT_POOLING

    pool_tag = "" if (pooling or DEFAULT_POOLING) == DEFAULT_POOLING else f"_{pooling}"
    # ffmpeg's crop+resize is close to but not bit-identical with the in-process
    # torch one (~0.99 token correlation), so the two paths must not share a
    # cache entry. Empty when off, so existing cache files keep their names.
    pre_tag = "_pre" if preprocess else ""

    def _sanitize(s: str, max_len: int) -> str:
        # geometry.slug is a short clean tag for HF-hub backbones, but for a
        # local (2a/2b merged) backbone loaded without an explicit
        # slug_override it falls back to load_backbone's raw-path-derived
        # default -- which can still contain OS path separators (backslashes
        # on Windows) and other characters that are not safe to embed
        # directly as one filename component. Sanitize defensively rather
        # than assume callers always pass a clean slug.
        return "".join(c if (c.isalnum() or c in "-_") else "_" for c in s)[:max_len]

    safe_stem = _sanitize(video_path.stem, 80)
    safe_slug = _sanitize(str(geometry.slug), 60)
    return cache_dir / f"{safe_stem}__{video_fp}__{safe_slug}-{backbone_fp}__{vr_tag}_{fps_tag}_{view_tag}_{size_tag}{pool_tag}{pre_tag}_{range_tag}_tok_v{CACHE_FORMAT_VERSION}.h5"


class ResumableTokenCache:
    """Incrementally-written token cache for one (video, backbone,
    decode-params) identity. `/tokens` is a resizable HDF5 dataset;
    `completed_slots` (an attr, kept in lockstep with the dataset's real
    length) tracks how much is validly written, so a later run -- or a
    resumed one -- always knows exactly where to pick up."""

    def __init__(self, path: Path):
        self.path = path

    def load_progress(self) -> tuple[int, int | None]:
        """Returns (completed_slots, total_expected_slots), (0, None) if no
        usable cache exists yet."""
        if not self.path.exists():
            return 0, None
        try:
            with h5py.File(str(self.path), "r") as f:
                if int(f.attrs.get("format_version", -1)) != CACHE_FORMAT_VERSION:
                    return 0, None
                completed = int(f.attrs.get("completed_slots", 0))
                total = f.attrs.get("total_expected_slots")
                return completed, (int(total) if total is not None else None)
        except OSError:
            log.warning("Corrupt/unreadable video token cache at %s, starting fresh", self.path)
            return 0, None

    def init_or_resume(
        self,
        total_expected_slots: int,
        hidden_dim: int,
        metadata: dict[str, Any],
        n_pool_tokens: int = 5,
    ) -> int:
        """Create the cache if absent, or reuse it if it already matches this
        run's expected size (doc03 convention: never reuse a cache that
        doesn't match what's being asked for -- start clean instead of
        silently mixing data). Returns the slot count to resume from (0 for
        a fresh cache).

        `n_pool_tokens` defaults to the legacy 5 so existing callers are
        unaffected; the pooling layout is also part of the cache filename
        (see `cache_path_for_video`), so a resume can never straddle two
        layouts."""
        completed, total = self.load_progress()
        if self.path.exists() and total == total_expected_slots:
            return completed

        with h5py.File(str(self.path), "w") as f:
            f.create_dataset(
                "tokens", shape=(0, n_pool_tokens, hidden_dim),
                maxshape=(total_expected_slots, n_pool_tokens, hidden_dim),
                dtype=np.float16,
                chunks=(min(128, max(1, total_expected_slots)), n_pool_tokens, hidden_dim),
                compression="lzf", shuffle=True,
            )
            f.attrs["format_version"] = CACHE_FORMAT_VERSION
            f.attrs["completed_slots"] = 0
            f.attrs["total_expected_slots"] = total_expected_slots
            for key, value in metadata.items():
                if value is None:
                    continue
                if isinstance(value, (bool, int, float, str)):
                    f.attrs[key] = value
                else:
                    f.attrs[key] = json.dumps(value)
        return 0

    # HDF5 chunk length used by `init_or_resume`. Writes smaller than this land
    # inside a compressed chunk and force a decompress-recompress of the whole
    # chunk, so buffer up to at least this much before touching the file.
    _DS_CHUNK_SLOTS = 128

    def open(self) -> "ResumableTokenCache":
        """Open the file once for the whole extraction.

        Without this, `append` reopens the HDF5 file per decode chunk -- on an 8K
        source that is a ~5-slot write per call and thousands of open/resize/
        flush/close cycles on a file that grows past 500 MB, which dominated
        extraction time. Held open, writes are batched to whole dataset chunks
        and flushed periodically; `completed_slots` still only ever advances to
        what has actually been flushed, so crash-resume is unchanged apart from
        losing at most `flush_interval_slots` of work.
        """
        self._fh = h5py.File(str(self.path), "a")
        self._ds = self._fh["tokens"]
        self._buf: list[np.ndarray] = []
        self._buffered = 0
        return self

    def __enter__(self) -> "ResumableTokenCache":
        return self.open()

    def __exit__(self, *exc_info) -> None:
        self.close()

    def __del__(self) -> None:
        # Last-resort flush: if extraction dies mid-loop, still persist whatever
        # is buffered rather than silently dropping it on interpreter teardown.
        try:
            self.close()
        except Exception:
            pass

    def close(self) -> None:
        fh = getattr(self, "_fh", None)
        if fh is None:
            return
        try:
            self._drain(force=True)
        finally:
            fh.close()
            self._fh = None
            self._ds = None

    def _drain(self, force: bool = False) -> None:
        """Write buffered slots to the dataset in whole-chunk blocks."""
        if not self._buf:
            return
        if not force and self._buffered < self._DS_CHUNK_SLOTS:
            return
        block = np.concatenate(self._buf, axis=0)
        self._buf = []
        self._buffered = 0
        ds = self._ds
        start = ds.shape[0]
        ds.resize(start + block.shape[0], axis=0)
        ds[start:start + block.shape[0]] = block
        # completed_slots must never claim more than is on disk.
        self._fh.attrs["completed_slots"] = start + block.shape[0]
        self._fh.flush()

    def append(self, pooled: np.ndarray) -> None:
        """Append pooled tokens [n,5,D] and advance completed_slots.

        Inside a `with` block the write is buffered and flushed a dataset-chunk
        at a time; outside one it falls back to the original open-per-append
        behaviour so existing callers keep working unchanged.
        """
        if pooled.shape[0] == 0:
            return
        pooled = pooled.astype(np.float16)
        if getattr(self, "_fh", None) is not None:
            self._buf.append(pooled)
            self._buffered += pooled.shape[0]
            self._drain()
            return
        with h5py.File(str(self.path), "a") as f:
            ds = f["tokens"]
            start = ds.shape[0]
            ds.resize(start + pooled.shape[0], axis=0)
            ds[start:start + pooled.shape[0]] = pooled
            f.attrs["completed_slots"] = start + pooled.shape[0]
            f.flush()

    def read_all(self) -> np.ndarray:
        with h5py.File(str(self.path), "r") as f:
            return np.asarray(f["tokens"][:], dtype=np.float16)

    def is_complete(self) -> bool:
        completed, total = self.load_progress()
        return total is not None and completed >= total
