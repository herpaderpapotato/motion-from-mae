"""Video -> pooled backbone tokens, CUDA decode only.

Decodes every native source frame in the requested window (no fps resampling),
crops one SBS eye for VR, then runs the backbone per decode chunk so peak memory
does not scale with video length.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.backbone import (
    CROP_BOX,
    DEFAULT_POOLING,
    FULL_FRAME_CROP_BOX,
    clip_tokens_from_frames,
    compile_backbone,
    crop_resize_normalize,
    load_backbone,
    pooling_num_tokens,
)
from src.progress import step
from src.token_cache import ResumableTokenCache, cache_path_for_video


def decoder_timebase(video_path: Path) -> tuple[float, int]:
    """(fps, total_frames) for time <-> frame-index conversion.

    Uses the stream's nominal rate (ffprobe r_frame_rate), not its average, so
    the frame grid and the cache identity keyed on it stay stable for a file.
    This is NOT what times the funscript: see `source_frame_times`, which reads
    the real per-frame timestamps. The two disagree by 0.49 s over 50 minutes on
    a source that declares 60000/1001 and actually runs at 59.9297.
    """
    fps = None
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate", "-of", "json", str(video_path)],
            capture_output=True, text=True, check=True,
        )
        rate = json.loads(out.stdout)["streams"][0]["r_frame_rate"]
        num, _, den = rate.partition("/")
        candidate = float(num) / float(den or 1.0)
        if candidate > 0:
            fps = candidate
    except Exception as exc:
        # Leading newline: this can fire inside a `step` line still awaiting its result.
        print(f"\n  [warn] no r_frame_rate for {Path(video_path).name} ({type(exc).__name__}); "
              f"falling back to the average frame rate, which drifts on long files")

    from torchcodec.decoders import VideoDecoder

    # Header-only probe. The default "exact" seek mode builds a full frame index,
    # which is a demux scan of the entire file -- measured 55 s on a 24 GiB 8K
    # source -- and none of it is used here: this handle decodes nothing, and
    # under --preprocess torchcodec never touches the source at all. Fall back to
    # the scan only when the container carries no frame count of its own.
    meta = VideoDecoder(str(video_path), device="cpu", dimension_order="NHWC",
                        seek_mode="approximate").metadata
    total_frames = int(meta.num_frames or 0)
    if total_frames <= 0:
        meta = VideoDecoder(str(video_path), device="cpu", dimension_order="NHWC").metadata
        total_frames = int(meta.num_frames or 0)
    return float(fps if fps is not None else (meta.average_fps or 30.0)), total_frames


def source_frame_times(video_path: Path, min_frames: int = 0) -> np.ndarray | None:
    """Every video frame's presentation time in seconds, from the container index.

    The funscript is played against the SOURCE, so its actions have to carry the
    source's real frame times. Synthesising them from a single fps is only right
    when the stream's declared rate is its actual one, and it often isn't: this
    library's 6K masters declare 60000/1001 (59.94006) but run at 59.9297, which
    is 0.49 s of accumulated skew over 50 minutes. Reading the index costs ~3.5 s
    for 179k frames over SMB.

    Returns None if the table is unreadable or too short to cover the run, so the
    caller can fall back to the uniform grid.
    """
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "packet=pts_time", "-of", "csv=p=0", str(video_path)],
            capture_output=True, text=True, check=True,
        )
    except Exception:
        return None
    values = []
    for token in out.stdout.split():
        try:
            values.append(float(token))
        except ValueError:  # a packet without a pts makes the whole table unusable
            return None
    if len(values) < max(1, min_frames):
        return None
    return np.sort(np.asarray(values, dtype=np.float64))


class EyeDecoder:
    """torchcodec CUDA decoder that yields one SBS eye, uncropped and unresized."""

    def __init__(self, video_path: Path, device: torch.device, vr_mode: bool, sbs_crop: str):
        from torchcodec.decoders import VideoDecoder

        # seek_mode "exact" scans the container's frame index once on open (~5 s
        # for a 25 GB 8K file) but then reads forward without re-seeking.
        # "approximate" re-seeks to a keyframe every so often, and on a long-GOP
        # 8K source each of those stalls the decode for 2.5-3.5 s -- the periodic
        # hitch in the progress bar, and ~45% of total decode time.
        self.decoder = VideoDecoder(
            str(video_path), device=str(device), dimension_order="NHWC", seek_mode="exact",
        )
        self.vr_mode = vr_mode
        self.sbs_crop = sbs_crop
        self.decode_batch = self._auto_decode_batch()

    def _auto_decode_batch(self) -> int:
        """Frames per decode call, sized to keep the float32 intermediate in
        crop_resize_normalize under ~2 GB (8K SBS is ~199 MB/frame)."""
        meta = self.decoder.metadata
        src_w, src_h = getattr(meta, "width", None), getattr(meta, "height", None)
        if not src_w or not src_h:
            return 16
        eff_w = src_w // 2 if self.vr_mode else src_w
        bytes_per_frame = eff_w * src_h * 3 * 4
        return max(1, min(64, int(2 * 1024 ** 3 / bytes_per_frame)))

    def frames_at(self, indices: list[int]) -> torch.Tensor:
        frames = self.decoder.get_frames_at(indices)
        raw = frames.data if hasattr(frames, "data") else frames
        if not self.vr_mode:
            return raw
        half_w = raw.shape[2] // 2
        eye = raw[:, :, :half_w, :] if self.sbs_crop == "left" else raw[:, :, half_w:, :]
        return eye.contiguous()


class PreprocessedDecoder:
    """Reads a preprocess-cache clip, already cropped and at backbone resolution.

    Source frame indices are translated by the clip's first_source_frame, so
    callers index it exactly as they would the source video.
    """

    def __init__(self, clip_path: Path, device: torch.device, first_source_frame: int,
                 decode_batch: int = 256):
        from torchcodec.decoders import VideoDecoder

        self.decoder = VideoDecoder(
            str(clip_path), device=str(device), dimension_order="NHWC", seek_mode="exact",
        )
        self.offset = first_source_frame
        self.decode_batch = decode_batch

    def frames_at(self, indices: list[int]) -> torch.Tensor:
        frames = self.decoder.get_frames_at([i - self.offset for i in indices])
        return frames.data if hasattr(frames, "data") else frames


def extract_video_tokens(
    video_path: Path,
    backbone_id: str,
    device: torch.device,
    vr_mode: bool,
    sbs_crop: str,
    start_time: float | None,
    duration: float | None,
    batch_windows: int | None = None,
    model=None,
    geometry=None,
    use_cache: bool = False,
    cache_dir: Path | None = None,
    show_progress: bool = True,
    crop_box: tuple[float, float, float, float] | None = None,
    pooling: str | None = None,
    preprocess: bool = False,
    preprocess_dir: Path | None = None,
    backbone_img_size: int | None = None,
    compile_model: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Returns (tokens [S, P, D] float16, frame_idx [T] int32 relative to
    start_time, metadata).

    `backbone_img_size` comes from the head's `data_config` and applies to
    V-JEPA 2.1 only: it runs at any resolution (RoPE), so rebuilding it at the
    release default when the head was trained on another one would silently feed
    the head different features."""
    if model is None:
        with step(f"loading backbone {backbone_id}", show_progress):
            model, geometry = load_backbone(backbone_id, device=device, img_size=backbone_img_size)
        if compile_model:
            compile_backbone(model)

    with step("reading source metadata", show_progress) as st:
        fps_src, total_frames = decoder_timebase(video_path)
        st.note(f"{total_frames} frames @ {fps_src:.3f} fps ({total_frames / max(fps_src, 1e-6) / 60:.1f} min)")

    start_frame = int(round((start_time or 0.0) * fps_src))
    end_frame = total_frames if duration is None else min(total_frames, start_frame + int(round(duration * fps_src)))
    n_window_frames = max(0, end_frame - start_frame)

    rel_indices = np.arange(n_window_frames, dtype=np.int64)
    feature_fps = fps_src
    idx_list = (rel_indices + start_frame).tolist()

    window = geometry.window
    frames_per_slot = geometry.frames_per_slot
    total_expected_slots = -(-len(idx_list) // frames_per_slot)
    pooling = pooling or DEFAULT_POOLING

    def _make_metadata() -> dict:
        return {
            "backbone_id": geometry.backbone_id, "feature_fps": feature_fps,
            "frames_per_slot": frames_per_slot, "n_source_frames": int(len(rel_indices)),
            "pooling": pooling, "n_pool_tokens": pooling_num_tokens(pooling),
        }

    cache: ResumableTokenCache | None = None
    resume_from_slot = 0
    if use_cache:
        cache_path = cache_path_for_video(
            Path(video_path), geometry, vr_mode, sbs_crop, start_frame, end_frame, cache_dir,
            crop_box=crop_box, pooling=pooling, preprocess=preprocess,
        )
        cache = ResumableTokenCache(cache_path)
        cache_meta = {
            "video_path": str(Path(video_path).resolve()), "backbone_id": geometry.backbone_id,
            "feature_fps": feature_fps, "frames_per_slot": frames_per_slot,
            "vr_mode": vr_mode, "sbs_crop": sbs_crop,
            "crop_box": list(crop_box) if crop_box is not None else None,
        }
        resume_from_slot = cache.init_or_resume(
            total_expected_slots, geometry.hidden_dim, cache_meta,
            n_pool_tokens=pooling_num_tokens(pooling),
        )
        if resume_from_slot >= total_expected_slots > 0:
            if show_progress:
                print(f"  token cache hit: {cache_path.name}")
            with step("reading cached tokens", show_progress):
                tokens = cache.read_all()
            return tokens, rel_indices.astype(np.int32), _make_metadata()
        if resume_from_slot > 0 and show_progress:
            print(f"  resuming token cache at slot {resume_from_slot}/{total_expected_slots} "
                  f"({100 * resume_from_slot / total_expected_slots:.1f}%)")

    remaining_idx_list = idx_list[resume_from_slot * frames_per_slot:]

    # The preprocess cache has the SBS-eye crop, the frame-view crop and the
    # resize already baked in, so decoding it needs neither of those steps.
    decode_crop_box = crop_box
    decoder = None
    if preprocess and remaining_idx_list:
        from src import preprocess as pp

        try:
            clip_path, first_frame = pp.ensure(
                Path(video_path), vr_mode, sbs_crop,
                crop_box if crop_box is not None else CROP_BOX, geometry.resize,
                start_frame, end_frame, fps_src, cache_dir=preprocess_dir,
                quiet=not show_progress,
            )
            with step("opening preprocessed clip", show_progress):
                decoder = PreprocessedDecoder(clip_path, device, first_frame)
            decode_crop_box = FULL_FRAME_CROP_BOX
        except pp.PreprocessUnavailable as exc:
            print(f"  [warn] preprocess unavailable ({exc}); decoding the source directly")
    if decoder is None:
        # seek_mode="exact" indexes the container up front -- minutes on a 25 GB
        # 8K source, and the longest unexplained pause in the whole run.
        with step("indexing source frames (exact seek)", show_progress):
            decoder = EyeDecoder(Path(video_path), device, vr_mode, sbs_crop)
    decode_chunk = max(1, decoder.decode_batch)
    if cache is not None:
        cache.open()
    pooled_chunks: list[np.ndarray] = []
    carry: torch.Tensor | None = None
    progress = tqdm(
        total=len(idx_list), initial=resume_from_slot * frames_per_slot, unit="frame",
        desc="  extracting tokens", disable=not show_progress,
    )
    # Backbone windows are non-overlapping and self-contained, so splitting on
    # window boundaries and pooling each piece separately gives identical tokens
    # to pooling the whole video at once, without holding every normalized frame.
    for i in range(0, len(remaining_idx_list), decode_chunk):
        batch_idx = remaining_idx_list[i:i + decode_chunk]
        raw = decoder.frames_at(batch_idx)
        normalized = crop_resize_normalize(raw, geometry, device, crop_box=decode_crop_box)
        del raw

        if carry is not None:
            normalized = torch.cat([carry, normalized], dim=0)
            carry = None

        n_complete = (normalized.shape[0] // window) * window
        if n_complete > 0:
            pooled = clip_tokens_from_frames(
                model, geometry, normalized[:n_complete], device,
                batch_windows=batch_windows, pooling=pooling,
            )
            pooled_chunks.append(pooled)
            if cache is not None:
                cache.append(pooled)
        if normalized.shape[0] > n_complete:
            carry = normalized[n_complete:].clone()  # clone: drop the chunk's storage, keep the tail
        del normalized
        progress.update(len(batch_idx))

    if carry is not None and carry.shape[0] > 0:
        pooled = clip_tokens_from_frames(
            model, geometry, carry, device, batch_windows=batch_windows, pooling=pooling,
        )
        pooled_chunks.append(pooled)
        if cache is not None:
            cache.append(pooled)
    progress.close()

    if cache is not None:
        cache.close()  # flush the buffered tail before reading back
        with step("reading cached tokens", show_progress):
            tokens = cache.read_all()
    else:
        tokens = (
            np.concatenate(pooled_chunks, axis=0) if pooled_chunks
            else np.zeros((0, pooling_num_tokens(pooling), geometry.hidden_dim), dtype=np.float16)
        )

    return tokens, rel_indices.astype(np.int32), _make_metadata()
