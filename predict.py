"""Predict funscript positions from a video with a trained DispositionNext head.

    python predict.py --video video.mp4 --out video.funscript --vr --frame-view crop
    python predict.py --video video.mp4 --out video.funscript --start-time 1106.3 --duration 200

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

from src.backbone import FULL_FRAME_CROP_BOX
from src.checkpoint import DEFAULT_CHECKPOINT, load_dnx_model, resolve_pooling_for_head
from src.extract import extract_video_tokens
from src.funscript import predictions_to_funscript
from src.hlgauss import HLGAUSS_MODE_RADIUS
from src.infer import (
    CROP_SLOTS,
    DECODE_MODES,
    HOLD_GATE_ACTIVITY_THRESHOLD,
    HOLD_GATE_MIN_RUN_S,
    apply_hold_gate,
    sliding_window_predict_dnx,
    smooth_positions,
)
from src.preprocess import DEFAULT_PREPROCESS_DIR
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict disposition with DispositionNext (DNX)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
                        help="HF repo id, .safetensors export, or training .pt")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output funscript path (default: <video>.funscript)")
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
    parser.add_argument("--no-progress", action="store_true", help="Suppress the extraction progress bar")

    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda":
        parser.error("CUDA is required (torchcodec GPU decode)")
    if not torch.cuda.is_available():
        parser.error("torch.cuda.is_available() is False")
    print(f"Using device: {device}")

    model, model_cfg, data_cfg = load_dnx_model(args.checkpoint, device, use_ema=not args.use_raw_weights)
    frame_view = args.frame_view
    if frame_view == "auto":
        frame_view = "full" if data_cfg.get("frame_mode") == "full" else "crop"
    use_full_frame = frame_view == "full"
    crop_box = FULL_FRAME_CROP_BOX if use_full_frame else None
    pooling = resolve_pooling_for_head(model)

    tokens, frame_idx, meta = extract_video_tokens(
        args.video, data_cfg["backbone_id"], device, args.vr, args.sbs_crop,
        args.start_time, args.duration,
        use_cache=args.token_cache, cache_dir=args.token_cache_dir,
        show_progress=not args.no_progress, crop_box=crop_box, pooling=pooling,
        preprocess=args.preprocess, preprocess_dir=args.preprocess_dir,
        backbone_img_size=data_cfg.get("backbone_img_size"),
    )
    feature_fps = float(meta["feature_fps"])
    print(f"DNX tokens: {tokens.shape}, feature_fps={feature_fps:.3f}, pooling={pooling}, "
          f"frame_view={frame_view}, decode={args.decode}")

    t0 = time.perf_counter()
    crop_slots = args.dnx_crop_slots
    stride_slots = min(max(1, args.dnx_stride or max(1, crop_slots // 2)), crop_slots)
    position, activity = sliding_window_predict_dnx(
        model, tokens, device, crop_slots=crop_slots, overlap=1.0 - (stride_slots / crop_slots),
        decode=args.decode, decode_radius=args.decode_radius,
    )
    position = position[:len(frame_idx)]
    activity = activity[:len(frame_idx)]

    if args.hold_gate:
        position = apply_hold_gate(
            position, activity, feature_fps,
            activity_threshold=args.hold_gate_threshold, min_run_s=args.hold_gate_min_duration,
        )

    smooth_window = None
    if args.dnx_smooth != "none":
        smooth_window = args.dnx_smooth_window or (3 if args.dnx_smooth == "median" else 5)
        position = smooth_positions(position, args.dnx_smooth, smooth_window, args.dnx_smooth_polyorder)
        print(f"  smoothing: {args.dnx_smooth} w={smooth_window}")

    print(f"Prediction: {len(position)} frames in {time.perf_counter() - t0:.2f}s  "
          f"mean={position.mean():.4f}  std={position.std():.4f}")

    out_path = non_colliding_path(args.out or args.video.with_suffix(".funscript"))
    funscript = predictions_to_funscript(
        position, fps=feature_fps, start_time=args.start_time,
        metadata={
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
    print(f"Saved funscript -> {out_path}")

    if args.save_activity:
        act_path = out_path.with_suffix(".activity.npy")
        np.save(act_path, activity)
        print(f"Saved activity -> {act_path}")


if __name__ == "__main__":
    main()
