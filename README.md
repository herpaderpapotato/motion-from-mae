# motion-from-mae
Extracting human motion data from scenes

DispositionNext (DNX): a VideoMAEv2-backboned video → funscript motion model.
This repo is a legacy-free extraction of the DNX system from `motionhelp` (YOLO/pose, optical flow, the DispositionTCN baseline).

## What's here

- `src/models/` — `DispositionNext` head, vendored VideoMAEv2 ViT backbone.
- `src/data/` — token extraction/caching, on-the-fly video dataset, funscript
  I/O, wave postprocess, scene curation.
- `src/training/` — DNX losses, timing/amplitude/hold metrics.
- `scripts/` — extraction, training (Phase 1 frozen-head and Phase 2a/2b LoRA),
  evaluation, prediction CLI + tkinter job-queue GUI, dataset prep.

## Setup

`conda-win-64.lock.txt` is an **explicit** lockfile (exact package URLs, no
solving) capturing the source env's win-64 / CUDA 12.9 conda-forge build of
the torch stack (`pytorch`/`torchvision`/`torchaudio`/`torchcodec`, all
`cuda128_*`/`cuda129_*` builds, not `cpu_*`). A plain version-pinned
`environment.yml` was tried first and doesn't work: an unpinned solve on this
same channel config silently resolves to the CPU builds, and pinning the
source env's exact build strings for a *fresh* solve hits an unsatisfiable
`pybind11-abi` conflict (conda-forge/defaults repodata has been patched since
that env was built). The explicit lockfile sidesteps solving entirely, so
that drift can't bite here — but it does mean this only reproduces on win-64
with a CUDA 12.9-capable driver; there's no cross-platform equivalent.

```bat
conda create -p .conda --file conda-win-64.lock.txt
conda activate .\.conda
pip install -r requirements-pip.txt
```

Copy `.env.sample` to `.env` and fill in the xbvr database URL if using
`prepare_videos.py` / the predict-job GUI's scene browser.

## Quickstart

```bat
:: extract VideoMAEv2 tokens for labelled scenes
python scripts\extract_videomae.py --backbone data\models\backbones\VideoMAEv2-Base --slug videomaev2-b --device cuda:0

:: train the DNX head on frozen tokens (see configs/dnx_head.sample.yaml -- a
:: config file declares the slug/frame-mode/run-name set once; flags still win)
python scripts\train_disposition_next.py --config configs\dnx_head.yaml
python scripts\train_disposition_next.py --backbone-slug videomaev2-b --run-name my_run

:: continue a head run that was still improving (pass a larger --epochs)
python scripts\train_disposition_next.py --config configs\dnx_head.yaml ^
    --resume-from data\models\checkpoints_dnx\<run>\dnx_epoch1000.pt --epochs 2000

:: predict a funscript from a video
python scripts\predict_disposition.py --video path\to\video.mp4 --vr --sbs-crop left ^
    --checkpoint data\models\checkpoints_dnx\<run>\best_disposition_next.pt

:: evaluate against the timing benchmark
python scripts\evaluate_timing.py --benchmark data\benchmarks\timing_v2.json ^
    --checkpoint data\models\checkpoints_dnx\<run>\best_disposition_next.pt --run-name eval
```

`data/processed/`, `data/preprocessed/`, and `data/splits/` start empty — this
repo builds a fresh self-labelled dataset.
