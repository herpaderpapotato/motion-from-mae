# motion_from_mae — inference

Video → funscript with a trained DispositionNext head. CUDA only (torchcodec GPU decode).
```
python predict.py --video video.mp4 --out video.funscript --vr --frame-view crop --start-time 1106.3 --duration 200
```

`--checkpoint` defaults to `herpaderpapotato/motion_from_mae`; the head records the
backbone it needs (`herpaderpapotato/motion_from_mae_extract`) and both are pulled into
the HF cache on first use. It also accepts a local `.safetensors` export or a training `.pt`.

Hub checkpoints are re-checked every run (~1 s), so a re-published head or backbone is
picked up instead of being served stale from the cache; only a real download prints
anything. `--checkpoint-revision <sha|tag>` pins the head and skips the check,
`--offline` (or `HF_HUB_OFFLINE=1`) uses the cache as-is, and an unreachable hub falls
back to the cache with a warning. A new **backbone** revision changes the token cache
key, so cached tokens are re-extracted.

Tokens are cached under `data/video_token_cache/` (`--no-token-cache` to disable,
`--token-cache-dir` to move); a re-run or an interrupted run resumes from there.

Two backbone families are supported, picked from what the head records — nothing to
pass. A **VideoMAEv2** backbone is a checkpoint directory or HF repo (16-frame windows
at 224); a **V-JEPA 2.1** one is a single `.pt` (64-frame windows at 384, RoPE). 2.1
runs at whatever resolution the head was trained on, taken from its
`data_config['backbone_img_size']` — without that a head trained on 384 tokens would
be served the release default instead.

`--preprocess` bakes the eye crop, the frame-view crop and the resize into a cached
clip at the backbone's own input size (224 or 384) with ffmpeg + NVDEC
(`data/video_preprocess_cache/`, `--preprocess-dir` to move). Decoding an 8K source is
the throughput ceiling (~130 frame/s); the cached clip decodes at ~2000 frame/s, so
re-runs over a window are ~5x faster. Needs a `*_cuvid` decoder for the source codec.
ffmpeg's resize is not bit-identical with the in-process one, so predictions shift
slightly (position correlation ~0.99) and the two paths keep separate token caches.

`--compile` torch.compiles the backbone blocks: ~25 s of compile once at startup (the
backbone is loaded and compiled once for the whole batch, not per video), then a
measured 1.10x at 384 and 1.27x at 224 on a 3090. Needs triton (`pip install
triton-windows` on Windows). Worth it on a long run, not on a short one.

There's also a token cache by default which speeds things up if only the head model is updated. `--no-token-cache` to opt out on that.

Action timestamps come from the source's own per-frame presentation times (one ffprobe
index read, ~3.5 s for 179k frames), not from a uniform grid at the declared frame rate.
Some masters declare 60000/1001 but run at 59.9297, which drifts the whole script ~0.5 s
by the end of a 50-minute file. `--timing nominal-fps` restores the old behaviour; on a
genuinely CFR source the two are identical.

Output never overwrites: if `video.funscript` exists the run writes
`video.001.funscript`, then `.002`, and so on.

Confidence is written as three extra funscript axes (version 1.1 `axes` list), on the
same 0-100 integer scale as `pos`, **higher = more confident**:

| axis | signal | reads as |
|---|---|---|
| `C1` | distribution spread, stroke-speed trend regressed out | 50 = as sharp as this video's strokes usually are at this speed; 0 = much vaguer |
| `C2` | \|expectation - mode\| decode gap | whether the head is split between two positions, or just vague |
| `C3` | min of C1/C2 medians, held across each stroke | which strokes to review |

All three come free from the distribution the position is already decoded from. Raw
spread is **not** usable on its own: it scales with stroke speed (measured corr +0.30
against |velocity|), so it peaks at every turnaround. Dividing by speed makes it worse
(+0.69, merely inverted) because spread behaves like `A + B*|v|`; regressing speed out
and keeping the residual gets it to +0.05. That makes C1 relative to the video, while
C2 keeps an absolute scale.

The 0-100 mappings are display scaling, not calibration (`CONF_*` in `src/infer.py`,
each constant measured over 36k frames of real content). They rank frames within a
video — they are **not** error bars, and they measure amplitude uncertainty, not
timing: the training loss is a soft-min over ±5-frame shifts, so a sharp distribution
can still sit a few frames off. `--no-confidence-axes` drops them and the file to ~1/4
the size.

Resulting funscripts should only be used to facilitate funscript creation. Any attempts to use the direct outputs is both unsupported and potentially a safety risk.

| module | what |
|---|---|
| `predict.py` | CLI |
| `src/extract.py` | decode → eye crop → backbone tokens |
| `src/backbone.py` | geometry, frame preprocessing, pooling |
| `src/preprocess.py` | `--preprocess` ffmpeg/NVDEC crop+resize cache |
| `src/videomaev2_backbone.py` | the VideoMAEv2 ViT |
| `src/vjepa21_backbone.py` | the V-JEPA 2.1 ViT (RoPE) |
| `src/disposition_next.py`, `src/hlgauss.py` | the head |
| `src/infer.py` | sliding-window blend, hold gate, smoothing |
| `src/postprocess.py` | `--postprocess` wave normalisation |
| `src/checkpoint.py`, `src/token_cache.py`, `src/funscript.py` | loading, caching, output |
| `src/progress.py` | timed step lines |
| `src/hub.py` | HF revision checks, cache/offline fallback |
