"""ffmpeg/NVDEC preprocess cache: source video -> cropped, backbone-resolution clip.

Decoding an 8K source is the throughput ceiling for extraction (~140 frame/s of
NVDEC, whatever else the pipeline does). This bakes the SBS-eye crop, the
frame-view crop and the resize into a small cached clip once, so later runs over
the same window decode at thousands of frame/s instead.

The resize uses scale_cuda's plain bilinear because that is what
`crop_resize_normalize` does: NVDEC's own `-resize` applies a wide antialiasing
filter, which shifts pooled tokens well off the in-process path (token
correlation 0.88 vs 0.99 for bilinear). Maybe that's better, one day i'll test.
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path

from src.progress import step

DEFAULT_PREPROCESS_DIR = Path("data/video_preprocess_cache")

# Container codec name -> cuvid decoder. Only these can crop in NVDEC (`-crop`),
# because speeeeeeed
CUVID_DECODERS = {
    "h264": "h264_cuvid", "hevc": "hevc_cuvid", "vp8": "vp8_cuvid", "vp9": "vp9_cuvid",
    "av1": "av1_cuvid", "mpeg1video": "mpeg1_cuvid", "mpeg2video": "mpeg2_cuvid",
    "mpeg4": "mpeg4_cuvid", "vc1": "vc1_cuvid",
}

ENCODERS = (
    ["-c:v", "hevc_nvenc", "-preset", "p7", "-rc", "constqp", "-qp", "16"],
    ["-c:v", "h264_nvenc", "-preset", "p7", "-rc", "constqp", "-qp", "16"],
    ["-c:v", "libx264", "-preset", "veryfast", "-crf", "14"],
)

_RANGE_RE = re.compile(r"_f(\d+)-(\d+)\.mp4$")


class PreprocessUnavailable(RuntimeError):
    pass


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def _run_with_progress(cmd: list[str], total_frames: int, desc: str) -> subprocess.CompletedProcess:
    """Run ffmpeg with a frame progress bar.

    `-progress pipe:1` writes machine-readable key=value blocks to stdout, so the
    bar is driven by ffmpeg's own frame counter. stderr goes to a temp file
    rather than a pipe: nothing reads it while the transcode runs, and a full
    pipe buffer would deadlock ffmpeg.
    """
    from tqdm import tqdm

    cmd = [cmd[0], "-progress", "pipe:1", "-nostats", *cmd[1:]]
    with tempfile.TemporaryFile("w+", encoding="utf-8", errors="replace") as err:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err, text=True, bufsize=1)
        bar = tqdm(total=total_frames, unit="frame", desc=desc, unit_scale=False, leave=True)
        try:
            for line in proc.stdout:
                key, _, value = line.strip().partition("=")
                value = value.strip()
                if key == "frame" and value.isdigit():
                    bar.update(min(int(value), total_frames) - bar.n)
                elif key == "speed" and value not in ("", "N/A", "0x"):
                    bar.set_postfix_str(value, refresh=False)
        finally:
            proc.stdout.close()
            returncode = proc.wait()
            bar.close()
        err.seek(0)
        return subprocess.CompletedProcess(cmd, returncode, "", err.read())


def _ffmpeg_has(kind: str, name: str) -> bool:
    out = _run(["ffmpeg", "-hide_banner", f"-{kind}"]).stdout
    return any(line.split()[1:2] == [name] for line in out.splitlines() if line.strip())


def probe_source(video_path: Path) -> dict:
    out = _run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height", "-of", "json", str(video_path),
    ])
    if out.returncode:
        raise PreprocessUnavailable(f"ffprobe failed on {video_path}: {out.stderr.strip()[:200]}")
    s = json.loads(out.stdout)["streams"][0]
    return {"codec": s["codec_name"], "width": int(s["width"]), "height": int(s["height"])}


def source_crop_pixels(
    src_w: int, src_h: int, vr_mode: bool, sbs_crop: str,
    crop_box: tuple[float, float, float, float],
) -> tuple[int, int, int, int]:
    """(left, top, width, height) in source pixels, matching crop_resize_normalize's
    rounding on the eye it would have been handed."""
    eye_x0 = (src_w // 2) if (vr_mode and sbs_crop == "right") else 0
    eye_w = (src_w // 2) if vr_mode else src_w
    eye_h = src_h
    x1, y1, x2, y2 = crop_box
    x1_px, x2_px = int(round(x1 * eye_w)), int(round(x2 * eye_w))
    y1_px = int(round(y1 * eye_h))
    y2_px = eye_h if y2 >= 1.0 else int(round(y2 * eye_h))
    left, top = (eye_x0 + x1_px) & ~1, y1_px & ~1
    w, h = (x2_px - x1_px) & ~1, (y2_px - y1_px) & ~1
    return left, top, min(w, src_w - left), min(h, src_h - top)



def _identity(
    video_path: Path, vr_mode: bool, sbs_crop: str,
    crop_box: tuple[float, float, float, float], resize: tuple[int, int],
) -> str:
    from src.token_cache import _fingerprint_video_file

    vr_tag = f"vr{sbs_crop[0]}" if vr_mode else "novr"
    box = "-".join(f"{v:.4f}" for v in crop_box)
    safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in video_path.stem)[:80]
    return f"{safe}__{_fingerprint_video_file(video_path)}__{vr_tag}__{box}__{resize[0]}x{resize[1]}"


def find_covering(
    cache_dir: Path, identity: str, start_frame: int, end_frame: int,
) -> tuple[Path, int] | None:
    """An existing cache clip whose frame range contains [start_frame, end_frame),
    as (path, first_source_frame). Lets a whole-video preprocess serve every
    later --start-time/--duration run."""
    best: tuple[Path, int] | None = None
    best_span = None
    for path in sorted(cache_dir.glob(f"{identity}_f*.mp4")):
        m = _RANGE_RE.search(path.name)
        side = path.with_suffix(".json")
        if not m or not side.exists():
            continue
        lo, hi = int(m.group(1)), int(m.group(2))
        if lo > start_frame or hi < end_frame:
            continue
        try:
            meta = json.loads(side.read_text())
        except (OSError, ValueError):
            continue
        offset, n = int(meta["first_source_frame"]), int(meta["n_frames"])
        if offset > start_frame or offset + n < end_frame:
            continue
        if best_span is None or (hi - lo) < best_span:
            best, best_span = (path, offset), hi - lo
    return best


def build(
    video_path: Path,
    out_path: Path,
    vr_mode: bool,
    sbs_crop: str,
    crop_box: tuple[float, float, float, float],
    resize: tuple[int, int],
    start_frame: int,
    end_frame: int,
    fps: float,
    quiet: bool = False,
    media_path: Path | None = None,
) -> tuple[Path, int]:
    """Transcode [start_frame, end_frame) to a cropped/resized clip. Returns
    (path, first_source_frame). `media_path` is read in place of `video_path`."""
    media_path = media_path or video_path
    with step("probing source", not quiet) as st:
        info = probe_source(media_path)
        st.note(f"{info['width']}x{info['height']} {info['codec']}")
    decoder = CUVID_DECODERS.get(info["codec"])
    if decoder is None or not _ffmpeg_has("decoders", decoder):
        raise PreprocessUnavailable(
            f"no cuvid decoder for codec {info['codec']!r}; preprocessing needs NVDEC-side "
            f"cropping (run with --no-preprocess)"
        )
    encoder = next((e for e in ENCODERS if _ffmpeg_has("encoders", e[1])), None)
    if encoder is None:
        raise PreprocessUnavailable("no usable ffmpeg encoder found (nvenc or libx264)")

    left, top, w, h = source_crop_pixels(info["width"], info["height"], vr_mode, sbs_crop, crop_box)
    right, bottom = info["width"] - (left + w), info["height"] - (top + h)

    # Input -ss lands on the first frame at or after the seek point, which may not
    # be start_frame exactly; back off a couple of frames and over-request, then
    # let the sidecar's first_source_frame carry the real alignment.
    guard = 2
    ss = max(0, start_frame - guard) / fps
    n_frames = (end_frame - start_frame) + 2 * guard

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".partial.mp4")
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{ss:.6f}",
        "-hwaccel", "cuda", "-hwaccel_output_format", "cuda", "-c:v", decoder,
        "-crop", f"{top}x{bottom}x{left}x{right}",
        "-i", str(media_path),
        "-an", "-sn", "-dn",
        # The clip is addressed by FRAME INDEX with a source-frame offset, so the
        # one thing that must hold is clip frame k == source frame k+offset.
        # ffmpeg's default -fps_mode auto is free to duplicate or drop frames to
        # force CFR on a source whose real cadence isn't its nominal rate (this
        # library's are: 59.9297 measured vs 59.9401 nominal). passthrough hands
        # every frame over with its own timestamp instead.
        "-fps_mode", "passthrough",
        "-vf", f"scale_cuda={resize[1]}:{resize[0]}:interp_algo=bilinear:format=nv12",
        "-frames:v", str(n_frames), "-copyts",
        *encoder, str(tmp),
    ]
    if quiet:
        proc = _run(cmd)
    else:
        print(f"  preprocess: crop {w}x{h}+{left}+{top} -> {resize[1]}x{resize[0]}, "
              f"{decoder} -> {encoder[1]}, frames [{start_frame}, {end_frame})")
        # Bar total is the requested span, not the guard-padded -frames:v count:
        # the guard frames are overhead, and counting them leaves a finished
        # transcode sitting at 99%.
        proc = _run_with_progress(cmd, end_frame - start_frame, "  transcoding")
    if proc.returncode or not tmp.exists():
        tmp.unlink(missing_ok=True)
        raise PreprocessUnavailable(f"ffmpeg preprocess failed: {proc.stderr.strip()[-400:]}")

    with step("checking clip", not quiet) as st:
        first_pts, n_written = _probe_result(tmp)
        st.note(f"{n_written} frames")
    first_source_frame = int(round(first_pts * fps))
    if first_source_frame > start_frame or first_source_frame + n_written < end_frame:
        tmp.unlink(missing_ok=True)
        raise PreprocessUnavailable(
            f"preprocess covered frames [{first_source_frame}, {first_source_frame + n_written}), "
            f"short of the requested [{start_frame}, {end_frame})"
        )
    tmp.replace(out_path)
    out_path.with_suffix(".json").write_text(json.dumps({
        "source": str(video_path.resolve()), "first_source_frame": first_source_frame,
        "n_frames": n_written, "fps": fps, "vr_mode": vr_mode, "sbs_crop": sbs_crop,
        "crop_box": list(crop_box), "resize": list(resize),
        "source_crop": [left, top, w, h], "encoder": encoder[1],
    }, indent=1))
    return out_path, first_source_frame


def _probe_result(path: Path) -> tuple[float, int]:
    out = _run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "packet=pts_time", "-of", "json", str(path),
    ])
    if out.returncode:
        raise PreprocessUnavailable(f"ffprobe failed on {path}: {out.stderr.strip()[:200]}")
    packets = json.loads(out.stdout).get("packets", [])
    times = sorted(float(p["pts_time"]) for p in packets if "pts_time" in p)
    if not times:
        raise PreprocessUnavailable(f"preprocessed clip {path} has no decodable packets")
    return times[0], len(times)


def cached_clip(
    video_path: Path,
    vr_mode: bool,
    sbs_crop: str,
    crop_box: tuple[float, float, float, float],
    resize: tuple[int, int],
    start_frame: int,
    end_frame: int,
    cache_dir: Path | None = None,
) -> tuple[Path, int] | None:
    cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_PREPROCESS_DIR
    if not cache_dir.is_dir():
        return None
    identity = _identity(Path(video_path), vr_mode, sbs_crop, crop_box, resize)
    return find_covering(cache_dir, identity, start_frame, end_frame)


def ensure(
    video_path: Path,
    vr_mode: bool,
    sbs_crop: str,
    crop_box: tuple[float, float, float, float],
    resize: tuple[int, int],
    start_frame: int,
    end_frame: int,
    fps: float,
    cache_dir: Path | None = None,
    quiet: bool = False,
    media_path: Path | None = None,
) -> tuple[Path, int]:
    """Reuse or build the preprocess cache for this window. Returns
    (clip path, first_source_frame). The cache is keyed on `video_path`; a build
    reads `media_path` if given (a local copy of it)."""
    cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_PREPROCESS_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    identity = _identity(Path(video_path), vr_mode, sbs_crop, crop_box, resize)
    hit = find_covering(cache_dir, identity, start_frame, end_frame)
    if hit is not None:
        if not quiet:
            print(f"  preprocess cache hit: {hit[0].name}")
        return hit
    out_path = cache_dir / f"{identity}_f{start_frame}-{end_frame}.mp4"
    return build(Path(video_path), out_path, vr_mode, sbs_crop, crop_box, resize,
                 start_frame, end_frame, fps, quiet=quiet, media_path=media_path)
