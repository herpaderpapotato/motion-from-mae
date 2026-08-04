# motion_from_mae — inference

Video → funscript with a trained DispositionNext head. CUDA only (torchcodec GPU decode).

```
python predict.py --video video.mp4 --out video.funscript --vr --frame-view crop \
    --start-time 1106.3 --duration 200
```

`--checkpoint` defaults to `herpaderpapotato/motion_from_mae`; the head records the
backbone it needs (`herpaderpapotato/motion_from_mae_extract`) and both are pulled into
the HF cache on first use. It also accepts a local `.safetensors` export or a training `.pt`.

Tokens are cached under `data/video_token_cache/` (`--no-token-cache` to disable,
`--token-cache-dir` to move); a re-run or an interrupted run resumes from there.

`--preprocess` bakes the eye crop, the frame-view crop and the resize into a cached
224x224 clip with ffmpeg + NVDEC (`data/video_preprocess_cache/`, `--preprocess-dir`
to move). Decoding an 8K source is the throughput ceiling (~130 frame/s); the cached
clip decodes at ~2000 frame/s, so re-runs over a window are ~5x faster. Needs a
`*_cuvid` decoder for the source codec. ffmpeg's resize is not bit-identical with the
in-process one, so predictions shift slightly (position correlation ~0.99) and the two
paths keep separate token caches.

TLDR, preprocess with ffmpeg can crunch a 1 hour 24GB 60fps 8k SBS VR video into ~3.6GB 224x224 cropped (or not) left eye view in about 23 minutes, which can then be used to generate a funscript in 2 minutes and reused in the future if the videomae or head model is updated. There's also a token cache by default which speeds things up if only the head model is updated. `--no-token-cache` to opt out on that.

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
| `src/disposition_next.py`, `src/hlgauss.py` | the head |
| `src/infer.py` | sliding-window blend, hold gate, smoothing |
| `src/postprocess.py` | `--postprocess` wave normalisation |
| `src/checkpoint.py`, `src/token_cache.py`, `src/funscript.py` | loading, caching, output |
