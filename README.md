# motion_from_mae — inference

Video → funscript with a trained DispositionNext head. CUDA only (torchcodec GPU decode).
```
python predict.py --video video.mp4 --out video.funscript --vr --frame-view crop --start-time 1106.3 --duration 200 --compile --preprocess --token-cache --do-a-barrel-roll
```

`--checkpoint` defaults to `herpaderpapotato/motion_from_mae_alt`; the model depends on a extractor model (backbone)
and that's recorded to keep it simple stupied (`herpaderpapotato/motion_from_mae_altextract`).
They auto download because who needs the noise of manually downloading that stuff. Or if you decide
to train your own it'll work with that too because everything seems to work that way.

Hub checkpoints are checked on run (~1 s), so it's up to date with the repo, 
or you can pin to an older revision with  `--checkpoint-revision <sha|tag>` if you decide the new model is crap.
`--offline` (or `HF_HUB_OFFLINE=1`) disables the check, in case that's your want.

Tokens are cached under `data/video_token_cache/` (`--no-token-cache` to disable,
`--token-cache-dir` to move); a re-run or an interrupted run resumes from there. 
Other probed video metadata is kept in a small `_src_v1.npz` per video, keyed on
path + size + mtime.

Two backbone families are supported, I've only really been working on the **V-JEPA 2.1**.
I'm fairly certain there's a rope bug in the finetune export so I'm putting myself on blast to go follow that up.

`--crop-box x1,y1,x2,y2` replaces `--frame-view` with an explicit box in fractions of
the eye (`crop` is `0.1667,0.3333,0.8333,1`, `full` is `0,0,1,1`). The box is part of
the token- and preprocess-cache identity. Or keep using frame view since "replaces" was
a bit of a misnomer. It's more like "overrides". Either way don't use them both or it'll be confusing.

`--preprocess` makes a small 384x384 version of the source, since if you rerun the same video
against multiple models, it saves a heap of time but definitely adds up over time.

`--compile` torch.compiles the backbone blocks. It's like 15% faster token extraction so I normally use it.

There's also a token cache by default which speeds things up for reruns.
i.e. if there's a lot of head model changes happening and you "want to see how the new compares"
`--no-token-cache` to opt out on that.

Action timestamps come from the source's own per-frame presentation times (one ffprobe
index read, ~3.5 s for 179k frames, cached per source file), because vfr was messing up some prediction timings.
Some videos reported 60000/1001 but then seemed to be closer to 59.9297, which added up to ~0.5 s by the end.
`--timing nominal-fps` restores the old behaviour; but I don't use it.

Output is simplified to keyframes by default (savgol lowpass → extrema seed → greedy
pchip refine within `--simplify-max-err` → device pass for `--simplify-min-amp` /
`--simplify-min-gap-ms`), and the dense per-frame track is kept beside it as
`video.raw.funscript`. `--no-simplify` writes the dense track alone. The simplified
file's `metadata.simplification` has some metadata info in it about the changes in case one day that's relevant.

Every funscript records what made it: `model_hash` (sha256 over the head's weight file
and its configs), the checkpoint id/revision, and the backbone id/revision.

Output never overwrites: if `video.funscript` exists the run writes
`video.001.funscript` (with `video.001.raw.funscript` beside it), then `.002`, and so
on. In folder mode a video that already has a funscript is skipped; `--force` processes
it anyway into a new numbered pair, and `--overwrite` replaces the existing pair.

Resulting funscripts should only be used to facilitate funscript creation. Any attempts to use the direct outputs is both unsupported and potentially a safety risk. That's the token disclaimer.

| module | what |
|---|---|
| `predict.py` | CLI |
| `src/extract.py` | decode → eye crop → backbone tokens |
| `src/backbone.py` | geometry, frame preprocessing, pooling |
| `src/preprocess.py` | `--preprocess` ffmpeg/NVDEC crop+resize cache |
| `src/videomaev2_backbone.py` | the VideoMAEv2 ViT |
| `src/vjepa21_backbone.py` | the V-JEPA 2.1 ViT (RoPE) |
| `src/disposition_next.py`, `src/hlgauss.py` | the head |
| `src/infer.py` | sliding-window blend, hold gate, confidence axes |
| `src/simplify.py` | per-frame track → keyframes |
| `src/postprocess.py` | `--postprocess` wave normalisation |
| `src/checkpoint.py`, `src/token_cache.py`, `src/funscript.py` | loading, caching, output |
| `src/progress.py` | timed step lines |
| `src/hub.py` | HF revision checks, cache/offline fallback |

I read some real badly written AI (emojis/hype/headline/all the turns of phrase), and then I thought of this readme.md and felt like I was part of the problem (it wasn't that bad but still...), so I rewrote/culled bits. Apologies if it makes less sense now or reads worse.