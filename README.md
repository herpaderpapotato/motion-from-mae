# motion_from_mae — inference

Video → funscript with a trained DispositionNext head. CUDA only (torchcodec GPU decode).
```
python predict.py --video video.mp4 --out video.funscript --vr --frame-view crop --start-time 1106.3 --duration 200
```

`--checkpoint` defaults to `herpaderpapotato/motion_from_mae`; the head records the
backbone it needs (`herpaderpapotato/motion_from_mae_extract`) and both are pulled into
the HF cache on first use. It also accepts a local `.safetensors` export or a training `.pt`.

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

Output never overwrites: if `video.funscript` exists the run writes
`video.001.funscript`, then `.002`, and so on.

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
| `src/progress.py` | timed step lines, quiet HF cache lookups |
