# motion_from_mae — inference

Video → funscript for scripters with a trained DispositionNext head.

```
# default backbone
python predict.py --video video.mp4 --out video.funscript --frame-view crop

# videomae2.1 224x224 backbone
python predict.py --video video.mp4 --out video.funscript --frame-view crop --checkpoint herpaderpapotato/motion_from_mae

# v-jepa2.1 384x384 backbone, slower/larger, still in testing 
python predict.py --video video.mp4 --out video.funscript --frame-view crop --checkpoint herpaderpapotato/motion_from_mae_alt

# command guidance
python predict.py --help
```

On a 6gb 40 minute 5k scene with a 5090, this processed at 310 frames per second. ~7 minute script generation in that scenario.


## Install instructions
1. install miniconda or a full anaconda installer
2. Follow [WINDOWS.MD](WINDOWS.MD) or [LINUX.MD](LINUX.MD) (wsl)


## Comments

- Currently licensed as Creative Commons Attribution-NonCommercial 4.0 International, because that's what videomaev2 is so aligning keeps it simple. May change in the future if an alternate backbbone is adopted as primary.
- CUDA only (torchcodec GPU decode), it's a solvable problem but for now that's what it is.
- VR scene trained not normal flat scenes.
- Intended for jumpstarting a script.
    - It's consistently frame perfect on the usual positions and acts. It can take the monotony out of that 60 second, 2 stroke per second sequence and instead give the scripter ample time to put into that hand+hand+other sequence that's more nuanced.
    - Outputs native fps funscripts (a lot of keypoints). To really make more normal funscripts, a good simplification algorithm probably needs to be added to the mix, but I'm not decided on it yet.
    - It's not trained on any community or other scripts. Current dataset is only 359 x 20 second (1200 interpolated values + frames each) sequences for train, and 40 for validation. As that expands, performance would be expected to improve.
    - During transitions the output is questionable. Sometimes it makes sense, sometimes it's garbage. It's likely a solvable problem but the intent is to assist human scripters, not replace.
- My speed test results with a 4090
    - with preprocessing 
        - ~3gb VRAM usage during ffmpeg video preprocessing. 25gb 8k @ 120fps ~ 21 minutes for 55minute video.
        - 1.5gb VRAM during token extraction. ~2 mins.
        - ~1gb VRAM during funscript prediction. ~20 seconds.
        - Total time ~25 minutes.
    - Without preprocessing, 25gb 8k @ 120fps ~ 23 minutes for 55minute video.
        - Faster, still has reusable token cache for video, but no reusable video file.


`--checkpoint` defaults to `herpaderpapotato/motion_from_mae`; the head records the
backbone it needs (`herpaderpapotato/motion_from_mae_extract`) and both are pulled into
the HF cache on first use. It also accepts a local `.safetensors` export or a training `.pt`.

Tokens are cached under `data/video_token_cache/` (`--no-token-cache` to disable,
`--token-cache-dir` to move); a re-run or an interrupted run resumes from there.

`--preprocess` makes a cached
224x224 clip with ffmpeg + NVDEC (`data/video_preprocess_cache/`, `--preprocess-dir`
to move). Decoding an 8K source is ~130 frame/s, the cached
clip decodes at ~2000 frame/s. i.e. re-runs over a window are ~5x faster. Needs a
`*_cuvid` decoder for the source codec. ffmpeg's resize isnt identical to the non-preprocessed one, so predictions can duffer slightly (position correlation ~0.99). Training data was all ffmpeg resized so maybe it'd be better, or maybe not. Too soon to say.

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
| `src/disposition_next.py`, `src/hlgauss.py` | the head |
| `src/infer.py` | sliding-window blend, hold gate, smoothing |
| `src/postprocess.py` | `--postprocess` wave normalisation |
| `src/checkpoint.py`, `src/token_cache.py`, `src/funscript.py` | loading, caching, output |


Also I used ai to help write the code and documentation (duh), because it's a lot of iteration and testing...

It's been a problem I've been coming going back to since 2023. It aint perfect, and my knowledge is still catching up in many areas (compare the loss functions I had in silver-lamp to the ones in this!).

I look at this iteration as significant, and that it does something unique, despite the unusal nature the task. Any minute now Cunningham's Law will kick in and someone that knows what they're actually doing will step in!
