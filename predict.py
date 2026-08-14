"""Predict funscript positions from a video with a trained DispositionNext head.

    python predict.py --video video.mp4 --out video.funscript --vr --frame-view crop
    python predict.py --video video.mp4 --out video.funscript --start-time 1106.3 --duration 200
    python predict.py --video FOLDER    # every .mp4 under it without a funscript
    python predict.py                   # file/folder picker

Defaults to the published head (herpaderpapotato/motion_from_mae_alt), which names its
own backbone; both are pulled from the HF cache on first use. CUDA required.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.backbone import (
    FULL_FRAME_CROP_BOX,
    compile_backbone,
    load_backbone,
    warmup_backbone,
)
from src.checkpoint import DEFAULT_CHECKPOINT, load_dnx_model, resolve_pooling_for_head
from src.extract import extract_video_tokens, source_frame_times
from src.funscript import predictions_to_funscript
from src.hlgauss import HLGAUSS_MODE_RADIUS
from src.infer import (
    CONF_GAP_CEIL,
    CONF_SPREAD_CEIL,
    CONF_SPREAD_FLOOR,
    CROP_SLOTS,
    DECODE_MODES,
    HOLD_GATE_ACTIVITY_THRESHOLD,
    HOLD_GATE_MIN_RUN_S,
    apply_hold_gate,
    sliding_window_predict_dnx,
    smooth_positions,
)
from src.preprocess import DEFAULT_PREPROCESS_DIR
from src.progress import human_duration, step
from src.token_cache import DEFAULT_CACHE_DIR

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

FRAME_VIEWS = ("auto", "crop", "full")


def non_colliding_path(path: Path, max_tries: int = 1000) -> Path:
    """`path` if free, else <stem>.### with the first unused number.

    Runs are cheap to repeat and expensive to lose, so a second run never
    overwrites the first one's funscript.
    """
    if not path.exists():
        return path
    for n in range(1, max_tries):
        candidate = path.with_name(f"{path.stem}.{n:03d}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"no free filename for {path} after {max_tries} tries")


def collect_videos(target: Path, exclude_dirs: tuple[Path, ...] = ()) -> list[Path]:
    """Videos to process. A folder is searched recursively; a named file is always kept.

    Folder search skips anything already scripted -- a batch is meant to be
    re-runnable over a growing library without redoing finished work -- and
    anything under `exclude_dirs`, which are our own caches of generated clips.
    """
    if not target.is_dir():
        return [target]
    excluded = [d.resolve() for d in exclude_dirs if d.is_dir()]
    return [v for v in sorted(target.rglob("*.mp4"))
            if not v.with_suffix(".funscript").exists()
            and not any(v.resolve().is_relative_to(d) for d in excluded)]


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
            device: torch.device, pooling, crop_box, frame_view: str,
            backbone=None, geometry=None) -> str:
    """Predict one video and write its funscript. Returns a one-line status."""
    verbose = not args.no_progress
    tokens, frame_idx, meta = extract_video_tokens(
        video, data_cfg["backbone_id"], device, args.vr, args.sbs_crop,
        args.start_time, args.duration,
        model=backbone, geometry=geometry,
        use_cache=args.token_cache, cache_dir=args.token_cache_dir,
        show_progress=verbose, crop_box=crop_box, pooling=pooling,
        preprocess=args.preprocess, preprocess_dir=args.preprocess_dir,
        backbone_img_size=data_cfg.get("backbone_img_size"),
        compile_model=args.compile and backbone is None,
    )
    feature_fps = float(meta["feature_fps"])
    if verbose:
        print(f"  tokens: {tokens.shape[0]} slots x {tokens.shape[1]} x {tokens.shape[2]} "
              f"@ {feature_fps:.3f} fps")

    t0 = time.perf_counter()
    crop_slots = args.dnx_crop_slots
    stride_slots = min(max(1, args.dnx_stride or max(1, crop_slots // 2)), crop_slots)
    position, activity, confidence = sliding_window_predict_dnx(
        model, tokens, device, crop_slots=crop_slots, overlap=1.0 - (stride_slots / crop_slots),
        decode=args.decode, decode_radius=args.decode_radius,
    )
    position = position[:len(frame_idx)]
    activity = activity[:len(frame_idx)]
    confidence = {k: v[:len(frame_idx)] for k, v in confidence.items()}

    if args.hold_gate:
        position = apply_hold_gate(
            position, activity, feature_fps,
            activity_threshold=args.hold_gate_threshold, min_run_s=args.hold_gate_min_duration,
        )

    smooth_window = None
    if args.dnx_smooth != "none":
        smooth_window = args.dnx_smooth_window or (3 if args.dnx_smooth == "median" else 5)
        position = smooth_positions(position, args.dnx_smooth, smooth_window, args.dnx_smooth_polyorder)
        if verbose:
            print(f"  smoothing: {args.dnx_smooth} w={smooth_window}")

    if verbose:
        print(f"  head: {len(position)} frames in {time.perf_counter() - t0:.2f}s, "
              f"mean {position.mean():.3f}, std {position.std():.3f}")
        print(f"  confidence: C1 spread {100 * confidence['spread'].mean():.0f}, "
              f"C2 agreement {100 * confidence['agreement'].mean():.0f} (mean of 100)")

    # Extra axes ride the main track's timestamps, so they must be built from
    # the ungated, unsmoothed confidence -- those filters only touch position.
    axes = None
    if args.confidence_axes:
        axes = {
            "C1": {"values": confidence["spread"],
                   "metadata": {"name": "confidence_spread",
                                "description": "100 = bin distribution as sharp as the "
                                               "HL-Gauss training sigma, 0 = too wide to place",
                                "scale_floor": CONF_SPREAD_FLOOR, "scale_ceil": CONF_SPREAD_CEIL,
                                "calibrated": False}},
            "C2": {"values": confidence["agreement"],
                   "metadata": {"name": "confidence_agreement",
                                "description": "100 = expectation and mode decodes agree, "
                                               "0 = split between two positions",
                                "scale_ceil": CONF_GAP_CEIL, "calibrated": False}},
        }

    # Action times come from the source's own frame timestamps; a stream whose
    # declared rate isn't its real one drifts the whole script otherwise.
    start_frame = int(round(args.start_time * feature_fps))
    frame_times = None
    if args.timing == "source-pts":
        with step("reading frame timestamps", verbose) as st:
            times = source_frame_times(video, min_frames=start_frame + len(position))
            if times is None:
                st.note("unavailable, using the uniform fps grid")
            else:
                frame_times = times[start_frame:start_frame + len(position)]
                measured = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else feature_fps
                st.note(f"{len(times)} frames, measured {measured:.4f} fps "
                        f"(declared {feature_fps:.4f})")

    out_path = non_colliding_path(out or video.with_suffix(".funscript"))
    funscript = predictions_to_funscript(
        position, fps=feature_fps, start_time=args.start_time, frame_times=frame_times,
        axes=axes,
        metadata={
            "timing": "source_pts" if frame_times is not None else "uniform_fps",
            "creator": "VideoToMotion", "type": "basic", "model": "disposition_next",
            "output_fps": feature_fps, "start_time_seconds": args.start_time,
            "hold_gate": args.hold_gate,
            "crop_slots": crop_slots, "stride_slots": stride_slots,
            "smooth": args.dnx_smooth, "smooth_window": smooth_window,
            "frame_view": frame_view, "pooling": pooling, "decode": args.decode,
            "decode_radius": args.decode_radius if args.decode == "mode" else None,
        },
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(funscript, fh)
    status = f"-> {out_path} ({len(funscript['actions'])} actions"
    status += f", axes {'+'.join(a['id'] for a in funscript['axes'])})" if axes else ")"

    if args.save_activity:
        act_path = out_path.with_suffix(".activity.npy")
        np.save(act_path, activity)
        status += f", activity -> {act_path.name}"
    return status


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
    parser.add_argument("--out", type=Path, default=None,
                        help="Output funscript path (default: <video>.funscript). Single video only")
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
    parser.add_argument("--save-activity", action="store_true", help="Write a sidecar .activity.npy")
    parser.add_argument("--confidence-axes", dest="confidence_axes", action="store_true", default=True,
                        help="Write per-frame confidence as extra funscript axes C1 (distribution "
                             "spread) and C2 (expectation-vs-mode agreement), 0-100 like pos, "
                             "higher = more confident. Uncalibrated: a ranking, not an error bar")
    parser.add_argument("--no-confidence-axes", dest="confidence_axes", action="store_false",
                        help="Position track only (~1/3 the file size)")

    parser.add_argument("--dnx-crop-slots", type=int, default=CROP_SLOTS,
                        help=f"Head window length in slots (default {CROP_SLOTS} = {CROP_SLOTS * 2} frames)")
    parser.add_argument("--dnx-stride", type=int, default=None,
                        help="Head window stride in slots (default: half the window)")
    parser.add_argument("--dnx-smooth", choices=["none", "median", "savgol"], default="none",
                        help="Post-decode temporal smoothing")
    parser.add_argument("--dnx-smooth-window", type=int, default=None,
                        help="Smoothing window in frames (odd; default 3 median / 5 savgol)")
    parser.add_argument("--dnx-smooth-polyorder", type=int, default=2, help="--dnx-smooth savgol only")
    parser.add_argument("--use-raw-weights", action="store_true",
                        help="Use raw (non-EMA) weights instead of EMA")
    parser.add_argument("--decode", choices=list(DECODE_MODES), default="expectation",
                        help="'expectation' is the mean of the blended bin distribution; 'mode' "
                             "takes the expectation over only the +/---decode-radius bins around "
                             "the peak. Decode-time only")
    parser.add_argument("--decode-radius", type=int, default=HLGAUSS_MODE_RADIUS,
                        help="--decode mode only: half-width in bins; 0 is a plain argmax")

    parser.add_argument("--token-cache", dest="token_cache", action="store_true", default=True,
                        help="Cache/resume extracted tokens")
    parser.add_argument("--no-token-cache", dest="token_cache", action="store_false")
    parser.add_argument("--token-cache-dir", type=Path, default=DEFAULT_CACHE_DIR)

    parser.add_argument("--preprocess", dest="preprocess", action="store_true", default=False,
                        help="Bake the eye/frame-view crop and the resize into a cached "
                             "backbone-resolution clip with ffmpeg+NVDEC, so this and later "
                             "runs over the same window skip decoding the full-resolution source")
    parser.add_argument("--no-preprocess", dest="preprocess", action="store_false")
    parser.add_argument("--preprocess-dir", type=Path, default=DEFAULT_PREPROCESS_DIR)
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

    videos = collect_videos(target, exclude_dirs=(args.preprocess_dir, args.token_cache_dir))
    if not videos:
        print(f"no unscripted .mp4 under {target}")
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

    model, model_cfg, data_cfg = load_dnx_model(
        args.checkpoint, device, use_ema=not args.use_raw_weights,
        revision=args.checkpoint_revision)
    frame_view = args.frame_view
    if frame_view == "auto":
        frame_view = "full" if data_cfg.get("frame_mode") == "full" else "crop"
    use_full_frame = frame_view == "full"
    crop_box = FULL_FRAME_CROP_BOX if use_full_frame else None
    pooling = resolve_pooling_for_head(model)

    # Load the backbone ONCE for the whole batch. Extraction would otherwise
    # resolve the hub repo, rebuild it and re-pay the torch.compile cost per
    # video -- the repeated "Fetching N files" and the long silent start.
    backbone_id = data_cfg["backbone_id"]
    verbose = not args.no_progress
    with step(f"Loading backbone {backbone_id}", verbose, indent="") as st:
        backbone, geometry = load_backbone(
            backbone_id, device=device, img_size=data_cfg.get("backbone_img_size"))
        rev = geometry.backbone_revision
        st.note((f"{rev[:7]}, " if rev else "")
                + f"{geometry.slug}, {geometry.window}-frame windows @ {geometry.resize[0]}px")
    if args.compile:
        with step("Compiling backbone blocks (one-off)", verbose, indent=""):
            compile_backbone(backbone)
            warmup_backbone(backbone, geometry, device)

    print(f"Settings: frame_view={frame_view} pooling={pooling} decode={args.decode} "
          f"vr={'on (' + args.sbs_crop + ' eye)' if args.vr else 'off'} "
          f"preprocess={'on' if args.preprocess else 'off'} "
          f"token_cache={'on' if args.token_cache else 'off'}")
    print(f"\n{len(videos)} video(s) to process under {target}")

    t0 = time.perf_counter()
    done = failed = 0
    for i, video in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}] {video}  ({video.stat().st_size / 1024 ** 3:.1f} GiB)", flush=True)
        t_video = time.perf_counter()
        try:
            status = process(video, args.out, args, model, data_cfg, device,
                             pooling, crop_box, frame_view, backbone=backbone, geometry=geometry)
        except Exception as exc:  # keep going through a batch
            status = f"FAILED: {type(exc).__name__}: {exc}"
            failed += 1
        else:
            done += 1
        print(f"  {status}  [{human_duration(time.perf_counter() - t_video)}]", flush=True)
    print(f"\n{done} processed, {failed} failed in {human_duration(time.perf_counter() - t0)}")


if __name__ == "__main__":
    main()
