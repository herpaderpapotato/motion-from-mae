"""Predict funscript positions from a video with a trained DispositionNext head.

    python predict.py --video video.mp4 --out video.funscript --vr --frame-view crop
    python predict.py --video video.mp4 --out video.funscript --start-time 1106.3 --duration 200
    python predict.py --video FOLDER    # every .mp4 under it without a funscript
    python predict.py                   # file/folder picker

Defaults to the published head (herpaderpapotato/motion_from_mae_alt), which names its
own backbone (override with --backbone); both are pulled from the HF cache on first
use. CUDA required.

Output is simplified to keyframes by default and the dense per-frame track is kept
beside it as <name>.raw.funscript; --no-simplify writes the dense track alone.
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.backbone import (
    CROP_BOX,
    FULL_FRAME_CROP_BOX,
    compile_backbone,
    crop_box_tag,
    load_backbone,
    warmup_backbone,
)
from src.checkpoint import (
    DEFAULT_CHECKPOINT,
    load_dnx_model,
    resolve_interleave_for_head,
    resolve_pooling_for_head,
)
from src.extract import extract_video_tokens, source_frame_times, source_read_needed
from src.funscript import predictions_to_funscript
from src.hlgauss import HLGAUSS_MODE_RADIUS
from src.infer import (
    CONF_GAP_CEIL,
    CONF_RESIDUAL_HALF_RANGE,
    CONF_SMOOTH_FRAMES,
    CROP_SLOTS,
    DECODE_MODES,
    HOLD_GATE_ACTIVITY_THRESHOLD,
    HOLD_GATE_MIN_RUN_S,
    STROKE_MIN_DISTANCE_S,
    STROKE_PROMINENCE,
    apply_hold_gate,
    confidence_axes,
    sliding_window_predict_dnx,
)
from src.ofsp import add_to_project
from src.preprocess import DEFAULT_PREPROCESS_DIR
from src.progress import human_duration, step
from src.simplify import (
    EXTREMA_PROMINENCE_FRAC,
    MAX_ERR,
    MIN_AMPLITUDE,
    MIN_FRAMES,
    MIN_GAP_MS,
    SMOOTH_WINDOW_S,
    reconstruction_stats,
    simplify,
)
from src.token_cache import DEFAULT_CACHE_DIR

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

FRAME_VIEWS = ("auto", "crop", "full")
DEFAULT_LOCAL_COPY_DIR = Path("data/source_staging")
RAW_TAG = ".raw"


def parse_crop_box(text: str) -> tuple[float, float, float, float]:
    """"x1,y1,x2,y2" as fractions of the (already SBS-cropped) eye."""
    parts = [p for p in text.replace(" ", "").split(",") if p]
    try:
        values = tuple(float(p) for p in parts)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number in {text!r}")
    if len(values) != 4:
        raise argparse.ArgumentTypeError(f"expected four comma-separated values, got {len(values)}")
    x1, y1, x2, y2 = values
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise argparse.ArgumentTypeError(
            f"need 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1, got {text!r}")
    return values


def raw_path_for(path: Path) -> Path:
    """The dense sidecar beside a simplified script: name.funscript ->
    name.raw.funscript, name.001.funscript -> name.001.raw.funscript."""
    return path.with_name(path.stem + RAW_TAG + path.suffix)


def non_colliding_path(path: Path, companions: tuple = (), max_tries: int = 1000) -> Path:
    """`path` if free, else <stem>.### with the first unused number.

    Runs are cheap to repeat and expensive to lose, so a second run never
    overwrites the first one's funscript. A number is only free when the files
    `companions` derive from it are free too, so the raw sidecar keeps the same
    number as the script it belongs to.
    """
    def free(candidate: Path) -> bool:
        return not candidate.exists() and not any(c(candidate).exists() for c in companions)

    if free(path):
        return path
    for n in range(1, max_tries):
        candidate = path.with_name(f"{path.stem}.{n:03d}{path.suffix}")
        if free(candidate):
            return candidate
    raise RuntimeError(f"no free filename for {path} after {max_tries} tries")


def collect_videos(target: Path, exclude_dirs: tuple[Path, ...] = (),
                   include_scripted: bool = False) -> list[Path]:
    """Videos to process. A folder is searched recursively; a named file is always kept.

    Folder search skips anything already scripted -- a batch is meant to be
    re-runnable over a growing library without redoing finished work, and
    `include_scripted` (--force/--overwrite) is how you ask for a redo -- and
    anything under `exclude_dirs`, which are our own caches of generated clips.
    """
    if not target.is_dir():
        return [target]
    excluded = [d.resolve() for d in exclude_dirs if d.is_dir()]
    # return [v for v in sorted(target.rglob("*.mp4"))
    #         if (include_scripted or not v.with_suffix(".funscript").exists())
    #         and not any(v.resolve().is_relative_to(d) for d in excluded)]
    videos = [v for v in target.rglob("*.mp4")
              if (include_scripted or not v.with_suffix(".funscript").exists())
              and not any(v.resolve().is_relative_to(d) for d in excluded)]
    return sorted(videos, key=lambda v: v.stat().st_mtime, reverse=True)


def copy_to_local(src: Path, dest_dir: Path, verbose: bool) -> Path:
    from tqdm import tqdm

    dest_dir.mkdir(parents=True, exist_ok=True)
    size = src.stat().st_size
    free = shutil.disk_usage(dest_dir).free
    if free < size:
        raise OSError(f"--local-copy: {dest_dir} has {free / 1024 ** 3:.1f} GiB free, "
                      f"{src.name} needs {size / 1024 ** 3:.1f} GiB")
    dest = dest_dir / src.name
    tmp = dest.with_name(dest.name + ".partial")
    try:
        with open(src, "rb") as fi, open(tmp, "wb") as fo, tqdm(
                total=size, unit="B", unit_scale=True, unit_divisor=1024,
                desc="  copying to local", disable=not verbose) as bar:
            while chunk := fi.read(64 << 20):
                fo.write(chunk)
                bar.update(len(chunk))
        tmp.replace(dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return dest


# --------------------------------------------------------------------------- #
# GUI
# --------------------------------------------------------------------------- #

def pick_target() -> Path | None:
    """Minimal picker: choose a file or a folder, then Run. Returns None if closed."""
    import tkinter as tk
    from tkinter import filedialog

    chosen: list[Path] = []
    root = tk.Tk()
    root.title("Predict funscripts")
    root.resizable(False, False)

    var = tk.StringVar(value="")
    frame = tk.Frame(root, padx=12, pady=12)
    frame.pack()
    tk.Label(frame, text="Video file, or folder to search recursively:").grid(
        row=0, column=0, columnspan=3, sticky="w")
    tk.Entry(frame, textvariable=var, width=64, state="readonly").grid(
        row=1, column=0, columnspan=3, pady=(4, 8), sticky="we")

    run = tk.Button(frame, text="Run", width=12, state="disabled")

    def choose_file() -> None:
        f = filedialog.askopenfilename(parent=root, title="Select video",
                                       filetypes=[("Video", "*.mp4"),
                                                  ("All files", "*.*")])
        if f:
            var.set(f)
            run.config(state="normal")

    def choose_dir() -> None:
        d = filedialog.askdirectory(parent=root, title="Select folder")
        if d:
            var.set(d)
            run.config(state="normal")

    def start() -> None:
        chosen.append(Path(var.get()))
        root.destroy()

    run.config(command=start)
    tk.Button(frame, text="Select file...", width=14, command=choose_file).grid(
        row=2, column=0, sticky="w")
    tk.Button(frame, text="Select folder...", width=14, command=choose_dir).grid(
        row=2, column=1, sticky="w", padx=6)
    run.grid(row=2, column=2, sticky="e")

    root.mainloop()
    return chosen[0] if chosen else None


# --------------------------------------------------------------------------- #

def process(video: Path, out: Path | None, args: argparse.Namespace, model, data_cfg: dict,
            device: torch.device, pooling, crop_box, frame_view: str, backbone_id: str,
            model_meta: dict, backbone=None, geometry=None,
            media_path: Path | None = None) -> str:
    """Predict one video and write its funscript. Returns a one-line status.
    `media_path` is a local copy of `video` to read instead; caches and output
    stay keyed on `video`."""
    verbose = not args.no_progress
    tokens, frame_idx, meta = extract_video_tokens(
        video, backbone_id, device, args.vr, args.sbs_crop,
        args.start_time, args.duration,
        model=backbone, geometry=geometry,
        use_cache=args.token_cache, cache_dir=args.token_cache_dir,
        show_progress=verbose, crop_box=crop_box, pooling=pooling,
        preprocess=args.preprocess, preprocess_dir=args.preprocess_dir,
        backbone_img_size=data_cfg.get("backbone_img_size"),
        backbone_window=data_cfg.get("backbone_window"),
        compile_model=args.compile and backbone is None,
        interleave_input=resolve_interleave_for_head(model, data_cfg),
        media_path=media_path,
    )
    feature_fps = float(meta["feature_fps"])
    if verbose:
        print(f"  tokens: {tokens.shape[0]} slots x {tokens.shape[1]} x {tokens.shape[2]} "
              f"@ {feature_fps:.3f} fps")

    t0 = time.perf_counter()
    crop_slots = args.dnx_crop_slots
    stride_slots = min(max(1, args.dnx_stride or max(1, crop_slots // 2)), crop_slots)
    position, activity, moments = sliding_window_predict_dnx(
        model, tokens, device, crop_slots=crop_slots, overlap=1.0 - (stride_slots / crop_slots),
        decode=args.decode, decode_radius=args.decode_radius,
    )
    position = position[:len(frame_idx)]
    activity = activity[:len(frame_idx)]
    moments = {k: v[:len(frame_idx)] for k, v in moments.items()}

    if args.hold_gate:
        position = apply_hold_gate(
            position, activity, feature_fps,
            activity_threshold=args.hold_gate_threshold, min_run_s=args.hold_gate_min_duration,
        )

    if verbose:
        print(f"  head: {len(position)} frames in {time.perf_counter() - t0:.2f}s, "
              f"mean {position.mean():.3f}, std {position.std():.3f}")
    # Built from the FINAL position: C1 divides by its speed and C3 segments it
    # into strokes, so both have to see the track that gets written.
    axes = None
    if args.confidence_axes:
        conf = confidence_axes(position, moments, feature_fps)
        if verbose:
            print("  confidence (mean of 100): "
                  + ", ".join(f"{k} {100 * v.mean():.0f}" for k, v in conf.items()))
        axes = {
            "C1": {"values": conf["C1"],
                   "metadata": {"name": "confidence_anomaly",
                                "description": "Distribution spread with the stroke-speed trend "
                                               "regressed out: 50 = as sharp as this video's "
                                               "strokes usually are at this speed, 100 = much "
                                               "sharper, 0 = much vaguer. Relative to the video",
                                "residual_half_range": CONF_RESIDUAL_HALF_RANGE,
                                "smooth_frames": CONF_SMOOTH_FRAMES, "calibrated": False}},
            "C2": {"values": conf["C2"],
                   "metadata": {"name": "confidence_agreement",
                                "description": "100 = expectation and mode decodes agree, "
                                               "0 = split between two positions",
                                "scale_ceil": CONF_GAP_CEIL,
                                "smooth_frames": CONF_SMOOTH_FRAMES, "calibrated": False}},
            "C3": {"values": conf["C3"],
                   "metadata": {"name": "confidence_per_stroke",
                                "description": "min(median C1, median C2) held across each "
                                               "stroke: which strokes to review, not which "
                                               "frames",
                                "prominence": STROKE_PROMINENCE,
                                "min_distance_s": STROKE_MIN_DISTANCE_S, "calibrated": False}},
        }

    # After the confidence axes: their stroke prominence is an absolute threshold
    # on the model's own scale.
    normalization = None
    if args.normalize:
        lo, hi = float(position.min()), float(position.max())
        normalization = {"applied": hi > lo, "source_min": lo, "source_max": hi}
        if hi > lo:
            position = (position - lo) / (hi - lo)
        if verbose:
            print(f"  normalize: [{lo:.3f}, {hi:.3f}] -> [0, 1]"
                  + ("" if hi > lo else " skipped, flat track"))

    # Action times come from the source's own frame timestamps; a stream whose
    # declared rate isn't its real one drifts the whole script otherwise.
    start_frame = int(round(args.start_time * feature_fps))
    frame_times = None
    if args.timing == "source-pts":
        with step("reading frame timestamps", verbose) as st:
            times, from_cache = source_frame_times(
                video, min_frames=start_frame + len(position),
                cache_dir=args.token_cache_dir if args.token_cache else None,
                media_path=media_path,
            )
            if times is None:
                st.note("unavailable, using the uniform fps grid")
            else:
                frame_times = times[start_frame:start_frame + len(position)]
                measured = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else feature_fps
                st.note(f"{len(times)} frames, measured {measured:.4f} fps "
                        f"(declared {feature_fps:.4f})" + (", cached" if from_cache else ""))

    # Both variants are written from explicit per-frame times so the simplified
    # one keeps the timestamps of the frames it kept, not a re-indexed grid.
    timing = "source_pts" if frame_times is not None else "uniform_fps"
    if frame_times is None:
        frame_times = np.arange(len(position), dtype=np.float64) / feature_fps + args.start_time

    metadata = {
        "timing": timing,
        "creator": "VideoToMotion", "type": "basic", "model": "disposition_next",
        "output_fps": feature_fps, "start_time_seconds": args.start_time,
        "hold_gate": args.hold_gate,
        "normalize": normalization,
        "crop_slots": crop_slots, "stride_slots": stride_slots,
        "frame_view": frame_view,
        "crop_box": list(crop_box if crop_box is not None else CROP_BOX),
        "pooling": pooling, "decode": args.decode,
        "decode_radius": args.decode_radius if args.decode == "mode" else None,
        **model_meta,
    }

    simplifying = args.simplify and len(position) >= MIN_FRAMES
    out_path = out or video.with_suffix(".funscript")
    if not args.overwrite:
        out_path = non_colliding_path(out_path, companions=(raw_path_for,) if simplifying else ())
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def write(path: Path, positions, times, ax, extra: dict) -> dict:
        script = predictions_to_funscript(
            positions, fps=feature_fps, start_time=args.start_time, frame_times=times,
            axes=ax, metadata={**metadata, **extra})
        with open(path, "w") as fh:
            json.dump(script, fh)
        return script

    if not simplifying:
        extra = {"variant": "dense"}
        if args.simplify:  # asked for, but there is nothing to simplify
            extra["simplification"] = {"applied": False,
                                       "reason": f"only {len(position)} frames"}
        funscript = write(out_path, position, frame_times, axes, extra)
    else:
        # Simplify the quantised track the raw file holds, so the reported error
        # is between the two files that actually get written.
        dense = np.clip(np.rint(position * 100.0), 0.0, 100.0)
        t_s = time.perf_counter()
        kept = simplify(dense, feature_fps, max_err=args.simplify_max_err,
                        min_amp=args.simplify_min_amp, min_gap_ms=args.simplify_min_gap_ms)
        stats = reconstruction_stats(dense, kept)
        if verbose:
            print(f"  simplify: {len(dense)} -> {len(kept)} actions "
                  f"({100 * stats['kept_fraction']:.1f}%), linear error mean "
                  f"{stats['linear_error']['mean']:.2f} max {stats['linear_error']['max']:.2f} "
                  f"[{time.perf_counter() - t_s:.1f}s]")

        raw_path = raw_path_for(out_path)
        write(raw_path, position, frame_times, axes,
              {"variant": "dense", "simplification": {"applied": False,
                                                      "simplified_file": out_path.name}})
        sub_axes = ({k: {**v, "values": v["values"][kept]} for k, v in axes.items()}
                    if axes else None)
        funscript = write(
            out_path, dense[kept] / 100.0, frame_times[kept], sub_axes,
            {"variant": "simplified",
             "simplification": {
                 "applied": True,
                 "method": "savgol -> extrema -> greedy pchip -> device pass",
                 "params": {"max_err": args.simplify_max_err, "min_amp": args.simplify_min_amp,
                            "min_gap_ms": args.simplify_min_gap_ms,
                            "smooth_window_s": SMOOTH_WINDOW_S,
                            "prominence_frac": EXTREMA_PROMINENCE_FRAC},
                 "raw_file": raw_path.name,
                 **stats}})

    status = f"-> {out_path} ({len(funscript['actions'])} actions"
    if simplifying:
        status += f" from {len(position)}, raw -> {raw_path.name}"
    status += f", axes {'+'.join(a['id'] for a in funscript['axes'])})" if axes else ")"

    if args.ofsp:
        # The funscript is already on disk; a project failure shouldn't fail the video.
        try:
            status += ", " + add_to_project(video.with_suffix(".ofsp"), video, out_path, funscript)
        except Exception as exc:
            status += f", ofsp FAILED: {type(exc).__name__}: {exc}"

    if args.save_activity:
        act_path = out_path.with_suffix(".activity.npy")
        np.save(act_path, activity)
        status += f", activity -> {act_path.name}"
    return status


_DLL_DIR_HANDLES = []


def _has_ffmpeg_dlls(d: Path) -> bool:
    try:
        return d.is_dir() and any(d.glob("avutil-*.dll"))
    except OSError:
        return False


def _find_shared_ffmpeg_bin() -> Path | None:
    candidates = []
    if os.environ.get("FFMPEG_DIR"):
        root = Path(os.environ["FFMPEG_DIR"])
        candidates += [root / "bin", root]
    on_path = shutil.which("ffmpeg")
    if on_path:
        candidates.append(Path(on_path).parent)
    candidates += [Path(p) for p in os.environ.get("PATH", "").split(os.pathsep) if p]
    local = os.environ.get("LOCALAPPDATA")
    if local:
        winget = Path(local) / "Microsoft" / "WinGet" / "Packages"
        candidates += sorted(winget.glob("BtbN.FFmpeg.GPL.Shared.*_*/ffmpeg-*/bin"), reverse=True)
    return next((d for d in candidates if _has_ffmpeg_dlls(d)), None)


def ensure_windows_ffmpeg() -> None:
    """WINDOWS.MD steps 1+3: shared FFmpeg DLLs in the env's Library\\bin, ffprobe on PATH.

    Falls back to a shared build found elsewhere (FFMPEG_DIR, PATH, winget) by
    registering it for DLL loading and prepending it to PATH for this process.
    Must run before anything imports torchcodec.
    """
    if sys.platform != "win32":
        return
    env_bins = [Path(sys.prefix) / "Library" / "bin"]
    if os.environ.get("CONDA_PREFIX"):
        env_bins.append(Path(os.environ["CONDA_PREFIX"]) / "Library" / "bin")
    dlls_ok = any(_has_ffmpeg_dlls(d) for d in env_bins)
    probe_ok = shutil.which("ffprobe") is not None
    if dlls_ok and probe_ok:
        return

    src = _find_shared_ffmpeg_bin()
    if src is None:
        static = shutil.which("ffmpeg")
        if not dlls_ok:
            hint = f"; {static} has no avutil-*.dll beside it (static build?)" if static else ""
            print(f"[warn] no shared FFmpeg build found{hint}. torchcodec will fail to load: "
                  f"see WINDOWS.MD steps 1 and 3, or set FFMPEG_DIR to a gpl-shared build")
        else:
            print("[warn] ffprobe not on PATH: frame rate falls back to the container average, "
                  "which drifts on long files (WINDOWS.MD step 3)")
        return

    # torch/torchcodec load with LOAD_LIBRARY_SEARCH_DEFAULT_DIRS, which ignores PATH,
    # so the DLL directory has to be registered explicitly; PATH covers ffmpeg/ffprobe.
    if not dlls_ok:
        _DLL_DIR_HANDLES.append(os.add_dll_directory(str(src)))
    os.environ["PATH"] = str(src) + os.pathsep + os.environ.get("PATH", "")
    missing = [what for what, ok in (("FFmpeg DLLs", dlls_ok), ("ffprobe", probe_ok)) if not ok]
    print(f"[warn] {' and '.join(missing)} not in the environment; using {src} for this run.")
    if os.environ.get("CONDA_PREFIX"):
        print(f"  To make it permanent (WINDOWS.MD step 3):\n"
              f"    Copy-Item \"{src}\\*\" \"{env_bins[-1]}\"")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict disposition with DispositionNext (DNX)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video", type=Path, default=None,
                        help="Video file, or folder searched recursively for .mp4 without a "
                             "funscript. Omit to open a picker")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
                        help="HF repo id, .safetensors export, or training .pt")
    parser.add_argument("--checkpoint-revision", type=str, default=None,
                        help="Pin the HF head to a commit sha or tag (default: the hub's "
                             "current revision, re-checked every run)")
    parser.add_argument("--backbone", type=str, default=None,
                        help="Backbone HF repo id or local path, overriding the one the head "
                             "names in its data_config")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output funscript path (default: <video>.funscript). Single video only")
    parser.add_argument("--force", action="store_true",
                        help="Folder mode: also process videos that already have a funscript, "
                             "writing a new numbered one beside the old")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace an existing funscript instead of numbering around it. "
                             "Implies --force")
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--vr", dest="vr", action="store_true", default=True,
                        help="VR / SBS video -- crop a single eye before decode")
    parser.add_argument("--no-vr", dest="vr", action="store_false")
    parser.add_argument("--sbs-crop", type=str, default="left", choices=["left", "right"],
                        help="Which SBS half (eye) to use when --vr is set")
    parser.add_argument("--frame-view", type=str, default="crop", choices=list(FRAME_VIEWS),
                        help="Spatial framing fed to the backbone: 'crop' is the centre-bottom "
                             "crop, 'full' the whole eye. 'auto' follows the checkpoint's "
                             "data_config['frame_mode']")
    parser.add_argument("--crop-box", type=parse_crop_box, default=None, metavar="X1,Y1,X2,Y2",
                        help="Override --frame-view with an explicit box, as fractions of the "
                             f"eye. The 'crop' view is {','.join(f'{v:.4f}' for v in CROP_BOX)} "
                             "and 'full' is 0,0,1,1. Caches (tokens and --preprocess clips) are "
                             "keyed by the box, so each one is built once and reused")

    parser.add_argument("--timing", choices=["source-pts", "nominal-fps"], default="source-pts",
                        help="Where action timestamps come from: the source's own per-frame "
                             "presentation times, or a uniform grid at the stream's declared "
                             "frame rate. They differ whenever a container's nominal rate isn't "
                             "its real one (measured: 0.49 s of drift over a 50-minute file)")
    parser.add_argument("--start-time", type=float, default=0.0,
                        help="Start time in seconds")
    parser.add_argument("--duration", type=float, default=None,
                        help="Duration in seconds (default: to the end)")

    parser.add_argument("--hold-gate", action="store_true",
                        help="Clamp position to its running median during sustained low activity")
    parser.add_argument("--hold-gate-threshold", type=float, default=HOLD_GATE_ACTIVITY_THRESHOLD)
    parser.add_argument("--hold-gate-min-duration", type=float, default=HOLD_GATE_MIN_RUN_S,
                        help="Minimum sustained-low-activity duration in seconds")
    parser.add_argument("--normalize", action="store_true",
                        help="Min-max rescale the predicted track to span 0-100 before it is "
                             "written and simplified (applies to the .raw sidecar too)")
    parser.add_argument("--save-activity", action="store_true", help="Write a sidecar .activity.npy")
    parser.add_argument("--ofsp", dest="ofsp", action="store_true", default=True,
                        help="Add the written funscript as a track to <video>.ofsp (OpenFunscripter "
                             "project, which OFS opens in place of the video). An existing project "
                             "is backed up to <video>.ofsp.<timestamp>.backup first")
    parser.add_argument("--no-ofsp", dest="ofsp", action="store_false")
    parser.add_argument("--confidence-axes", dest="confidence_axes", action="store_true", default=False,
                        help="Write confidence as extra funscript axes C1 (speed-corrected "
                             "sharpness), C2 "
                             "(decode agreement) and C3 (per-stroke aggregate), 0-100 like pos, "
                             "higher = more confident. Uncalibrated: a ranking, not an error bar")
    parser.add_argument("--no-confidence-axes", dest="confidence_axes", action="store_false",
                        help="Position track only (~1/3 the file size)")

    parser.add_argument("--dnx-crop-slots", type=int, default=CROP_SLOTS,
                        help=f"Head window length in slots (default {CROP_SLOTS} = {CROP_SLOTS * 2} frames, "
                             f"or {CROP_SLOTS} frames for an interleave_input head)")
    parser.add_argument("--dnx-stride", type=int, default=None,
                        help="Head window stride in slots (default: half the window)")
    parser.add_argument("--use-raw-weights", action="store_true",
                        help="Use raw (non-EMA) weights instead of EMA")
    parser.add_argument("--decode", choices=list(DECODE_MODES), default="expectation",
                        help="'expectation' is the mean of the blended bin distribution; 'mode' "
                             "takes the expectation over only the +/---decode-radius bins around "
                             "the peak. Decode-time only")
    parser.add_argument("--decode-radius", type=int, default=HLGAUSS_MODE_RADIUS,
                        help="--decode mode only: half-width in bins; 0 is a plain argmax")

    parser.add_argument("--simplify", dest="simplify", action="store_true", default=True,
                        help="Reduce the per-frame track to keyframes (savgol, extrema seed, "
                             "greedy pchip refine, device pass) and keep the dense track as "
                             "<name>.raw.funscript")
    parser.add_argument("--no-simplify", dest="simplify", action="store_false",
                        help="Write the dense per-frame track only, with no .raw sidecar")
    parser.add_argument("--simplify-max-err", type=float, default=MAX_ERR,
                        help="Greedy pchip error budget in 0-100 position units: no frame ends "
                             "up further than this from the dense track")
    parser.add_argument("--simplify-min-amp", type=float, default=MIN_AMPLITUDE,
                        help="Drop strokes smaller than this (0-100), which a device cannot render")
    parser.add_argument("--simplify-min-gap-ms", type=float, default=MIN_GAP_MS,
                        help="Minimum spacing between kept actions")

    parser.add_argument("--token-cache", dest="token_cache", action="store_true", default=True,
                        help="Cache/resume extracted tokens, and the source probe "
                             "(frame count, declared fps, frame timestamps)")
    parser.add_argument("--no-token-cache", dest="token_cache", action="store_false")
    parser.add_argument("--token-cache-dir", type=Path, default=DEFAULT_CACHE_DIR)

    parser.add_argument("--preprocess", dest="preprocess", action="store_true", default=False,
                        help="Bake the eye/frame-view crop and the resize into a cached "
                             "backbone-resolution clip with ffmpeg+NVDEC, so this and later "
                             "runs over the same window skip decoding the full-resolution source")
    parser.add_argument("--no-preprocess", dest="preprocess", action="store_false")
    parser.add_argument("--preprocess-dir", type=Path, default=DEFAULT_PREPROCESS_DIR)
    parser.add_argument("--local-copy", type=Path, nargs="?", const=DEFAULT_LOCAL_COPY_DIR,
                        default=None, metavar="DIR",
                        help="Copy the source into DIR (default %(const)s when given without "
                             "one), read it from there, and delete the copy afterwards. Only "
                             "when the run would read the source's frames: skipped if the "
                             "frame times are cached and either the token cache is complete "
                             "or (--preprocess) the clip exists")
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile the backbone blocks. Costs ~25s of compile once, then "
                             "measured 1.10x at 384 and 1.27x at 224 on a 3090. Needs triton "
                             "(on Windows: pip install triton-windows)")
    parser.add_argument("--offline", action="store_true",
                        help="Never contact the hub: use whatever revision is already in the HF "
                             "cache. Otherwise every run checks the hub and pulls a newer one")
    parser.add_argument("--no-progress", action="store_true",
                        help="Quiet: no progress bars or per-phase status lines, one line per video")

    args = parser.parse_args()
    ensure_windows_ffmpeg()
    if args.offline:
        from src.hub import set_offline

        set_offline()

    target = args.video or pick_target()
    if target is None:
        print("cancelled")
        return
    target = Path(target)
    if not target.exists():
        parser.error(f"not found: {target}")

    redo = args.force or args.overwrite
    videos = collect_videos(target, exclude_dirs=(args.preprocess_dir, args.token_cache_dir,
                                                  *([args.local_copy] if args.local_copy else [])),
                            include_scripted=redo)
    if not videos:
        print(f"no {'' if redo else 'unscripted '}.mp4 under {target}")
        return
    if args.out is not None and len(videos) > 1:
        parser.error(f"--out takes a single video, but {len(videos)} were found under {target}")

    device = torch.device(args.device)
    if device.type != "cuda":
        parser.error("CUDA is required (torchcodec GPU decode)")
    if not torch.cuda.is_available():
        parser.error("torch.cuda.is_available() is False")
    if device.index is None:
        # Resolve a bare "cuda" to an explicit index: torchcodec wants a concrete
        # one. Then pin it as the current device -- torchcodec's CUDA decoder and
        # torch.compile's inductor/triton kernels both bind to the CURRENT device,
        # which stays cuda:0 unless set, so `--device cuda:1 --compile` otherwise
        # fails with `CUDA error: invalid argument` on the decoded frames.
        device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(device)
    print(f"Device: {device} ({torch.cuda.get_device_name(device)})")

    model, model_cfg, data_cfg, provenance = load_dnx_model(
        args.checkpoint, device, use_ema=not args.use_raw_weights,
        revision=args.checkpoint_revision)
    if args.crop_box is not None:
        crop_box = args.crop_box
        tag = crop_box_tag(crop_box)
        frame_view = tag if tag in ("crop", "full") else "custom"
    else:
        frame_view = args.frame_view
        if frame_view == "auto":
            frame_view = "full" if data_cfg.get("frame_mode") == "full" else "crop"
        crop_box = FULL_FRAME_CROP_BOX if frame_view == "full" else None
    pooling = resolve_pooling_for_head(model)

    # Load the backbone ONCE for the whole batch. Extraction would otherwise
    # resolve the hub repo, rebuild it and re-pay the torch.compile cost per
    # video -- the repeated "Fetching N files" and the long silent start.
    backbone_id = args.backbone or data_cfg["backbone_id"]
    verbose = not args.no_progress
    with step(f"Loading backbone {backbone_id}", verbose, indent="") as st:
        backbone, geometry = load_backbone(
            backbone_id, device=device, img_size=data_cfg.get("backbone_img_size"),
            window=data_cfg.get("backbone_window"))
        rev = geometry.backbone_revision
        st.note((f"{rev[:7]}, " if rev else "")
                + f"{geometry.slug}, {geometry.window}-frame windows @ {geometry.resize[0]}px"
                + (" (--backbone override)" if args.backbone else ""))
    # Provenance for every funscript written this run: which head, which weights
    # inside it, and which backbone produced the features.
    model_meta = {**provenance, "backbone_id": geometry.backbone_id,
                  "backbone_revision": geometry.backbone_revision}
    if args.compile:
        with step("Compiling backbone blocks (one-off)", verbose, indent=""):
            compile_backbone(backbone)
            warmup_backbone(backbone, geometry, device)

    box_note = ("" if frame_view != "custom"
                else "(" + ",".join(f"{v:g}" for v in crop_box) + ") ")
    print(f"Settings: frame_view={frame_view} {box_note}pooling={pooling} decode={args.decode} "
          f"vr={'on (' + args.sbs_crop + ' eye)' if args.vr else 'off'} "
          f"preprocess={'on' if args.preprocess else 'off'} "
          f"token_cache={'on' if args.token_cache else 'off'} "
          f"simplify={'on' if args.simplify else 'off'} "
          f"normalize={'on' if args.normalize else 'off'}"
          + (" overwrite" if args.overwrite else (" force" if args.force else "")))
    print(f"\n{len(videos)} video(s) to process under {target}")

    t0 = time.perf_counter()
    done = failed = 0
    for i, video in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}] {video}  ({video.stat().st_size / 1024 ** 3:.1f} GiB)", flush=True)
        t_video = time.perf_counter()
        staged = None
        try:
            if args.local_copy is not None:
                if source_read_needed(
                        video, geometry, args.vr, args.sbs_crop, args.start_time, args.duration,
                        use_cache=args.token_cache, cache_dir=args.token_cache_dir,
                        crop_box=crop_box, pooling=pooling, preprocess=args.preprocess,
                        preprocess_dir=args.preprocess_dir,
                        interleave_input=resolve_interleave_for_head(model, data_cfg),
                        need_frame_times=args.timing == "source-pts"):
                    staged = copy_to_local(video, args.local_copy, verbose)
                elif verbose:
                    print("  local copy: skipped, the run is served from caches")
            status = process(video, args.out, args, model, data_cfg, device,
                             pooling, crop_box, frame_view, backbone_id, model_meta,
                             backbone=backbone, geometry=geometry, media_path=staged)
        except Exception as exc:  # keep going through a batch
            status = f"FAILED: {type(exc).__name__}: {exc}"
            failed += 1
        else:
            done += 1
        finally:
            if staged is not None:
                staged.unlink(missing_ok=True)
        print(f"  {status}  [{human_duration(time.perf_counter() - t_video)}]", flush=True)
    print(f"\n{done} processed, {failed} failed in {human_duration(time.perf_counter() - t0)}")


if __name__ == "__main__":
    main()
