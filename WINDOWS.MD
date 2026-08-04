# Windows setup (conda)


## Warning

This particularly MD file was AI written based off of installing + testing on an existing troublesome windows machine. That said, it eventually managed to work through the amount of jank I had on that machine and get it working so it should work for most people.

## Guide

Verified on Windows 11, NVIDIA driver 610.47, RTX 5090, miniconda 24.11.0, Python 3.12.

`predict.py` is CUDA-only: torchcodec decodes on the GPU via NVDEC, so an NVIDIA GPU and a
*shared* FFmpeg build are both hard requirements.

Run these from an **Anaconda Prompt** (or from PowerShell after a one-time
`conda init powershell`, then open a new shell).

## Install

**1. Get a shared FFmpeg build.** torchcodec loads `avutil-*.dll` and friends at import
time and supports FFmpeg 4 through 8. The default `winget install ffmpeg` (`Gyan.FFmpeg`)
is a *static* build with no DLLs and will not work — you need a **gpl-shared** one:

```powershell
winget install --id BtbN.FFmpeg.GPL.Shared.7.1
```

(Or download `ffmpeg-n7.1-latest-win64-gpl-shared-7.1.zip` from
[BtbN/FFmpeg-Builds](https://github.com/BtbN/FFmpeg-Builds/releases/tag/latest) and unzip
it anywhere.)

**2. Create the environment.**

```powershell
conda create -n motion-from-mae python=3.12 -y
conda activate motion-from-mae
```

**3. Copy FFmpeg into the environment.** `%CONDA_PREFIX%\Library\bin` is the one directory
torch always registers for DLL loading, so putting the FFmpeg DLLs there makes torchcodec
find them regardless of what `PATH` looks like. It also puts `ffmpeg.exe` / `ffprobe.exe`
on `PATH` whenever the env is active, which `predict.py` needs.

`$src` is the folder containing `bin\ffmpeg.exe`; for the winget install it is:

```powershell
$src = (Get-ChildItem -Directory "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\BtbN.FFmpeg.GPL.Shared.7.1_*\ffmpeg-*")[0].FullName
Copy-Item "$src\bin\*" "$env:CONDA_PREFIX\Library\bin"
```

**4. Install torch, torchcodec and the requirements.**

```powershell
pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.13.0 torchcodec==0.15.0
pip install -r requirements.txt
```

Both must come from the `cu130` index. It is the only CUDA index carrying Windows
torchcodec wheels that also supports Blackwell — `cu126` has Windows wheels but no sm_120
kernels, and `cu128` has no Windows torchcodec wheel at all. The PyPI `torch` is CPU-only
and the PyPI `torchcodec` is built against a different libtorch; neither works.

## Verify

```powershell
where python
where ffprobe
python -c "import sys, torch, torchcodec; print(sys.executable); print(torch.__version__, torchcodec.__version__, torch.cuda.is_available())"
```

The first `where` hit must be inside the env, and both versions must end in `+cu130`:

```
2.13.0+cu130 0.15.0+cu130 True
```

## Run

```powershell
cd <repo>
python predict.py --video video.mp4 --out data\smoketest.funscript --vr --frame-view crop --start-time 60 --duration 30
```

Then the full run:

```powershell
python predict.py --video video.mp4 --out video.funscript --vr --frame-view crop
```

First run downloads the head (`herpaderpapotato/motion_from_mae`) and the backbone it names
(`herpaderpapotato/motion_from_mae_extract`) into `%USERPROFILE%\.cache\huggingface` (or
`$env:HF_HOME`). The "unauthenticated requests to the HF Hub" warning is harmless; set
`HF_TOKEN` to silence it.

`predict.py` creates `data\` in the current working directory, so run it from the repo root.
Caches land in `data\video_token_cache\` and, with `--preprocess`,
`data\video_preprocess_cache\`. Output never overwrites: a second run writes
`video.001.funscript`, then `.002`.

`--preprocess` needs a `*_cuvid` decoder for your source codec and ideally NVENC; the BtbN
gpl-shared build has both. Without them it raises `PreprocessUnavailable` and the default
path is unaffected.

## Troubleshooting

**`Could not load libtorchcodec`** — almost always FFmpeg. Check that the DLLs really landed
in the env, and that the wheel is the `cu130` one:

```powershell
Get-ChildItem "$env:CONDA_PREFIX\Library\bin\avutil-*.dll"
python -c "import torchcodec; print(torchcodec.__version__)"   # must end in +cu130
```

The traceback lists a failure for every FFmpeg major version; only the one matching your
install is real.

**`parser.error: CUDA is required`** / **`torch.cuda.is_available() is False`** — CPU-only
torch got installed. `python -c "import torch; print(torch.__version__)"` must end in
`+cu130`; reinstall from the index above.

**`no kernel image is available for execution on the device`** — the wheel has no kernels for
your card, most likely a `cu126` build on a 50-series GPU. Reinstall from `cu130`.

**`ModuleNotFoundError: No module named 'torch'` right after activating** — `python` resolved
to another Python that sits ahead of the env on `PATH` (`where python` will show it). Either
remove that entry from `PATH`, or add
`%CONDA_PREFIX%\etc\conda\activate.d\motion_from_mae.bat`:

```bat
@echo off
set "_MFM_OLD_PATH=%PATH%"
set "PATH=%CONDA_PREFIX%;%CONDA_PREFIX%\Library\bin;%CONDA_PREFIX%\Scripts;%PATH%"
```

with the matching `deactivate.d\motion_from_mae.bat`:

```bat
@echo off
set "PATH=%_MFM_OLD_PATH%"
set "_MFM_OLD_PATH="
```

**`pip install` uninstalled a package from another project** — conda envs do not shield you
from `%APPDATA%\Python\Python3xx\site-packages`, and pip will uninstall packages it finds
there. If you have a `pip install --user` history, set `$env:PYTHONNOUSERSITE = "1"` before
installing.

**`ffprobe` warning about `r_frame_rate`** — `ffprobe` is not on `PATH`, so the run falls back
to the container's average frame rate, which drifts over a long file and misaligns the
funscript timestamps. Re-check step 3.

**CUDA out of memory** — decode batch size is derived from the source resolution
(`EyeDecoder._auto_decode_batch`, budgeted to ~2 GB). Free VRAM, or pick another GPU with
`$env:CUDA_VISIBLE_DEVICES` (`--device` only accepts `cuda`, not `cuda:1`).

Note on that last line.... LOL ai can't read code and interprets this to mean that it can't handle cuda:1...
```python
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        parser.error("CUDA is required (torchcodec GPU decode)")
```
It can! Unless there's a regression elsewhere that I haven't seen/tested yet.


## Measured on a windows machine

5400x2700 HEVC SBS, 59.94 fps, 137,378 frames (~38 min), 6.4 GB, on an RTX 5090. Wall clock
for a 120 s window (7,193 frames), including ~8 s of model load:

| | |
|---|---|
| token extraction | ~300 frame/s (~7.5 min extrapolated for the full file) |
| head prediction | 0.43 s per 120 s window |
| 120 s window, no `--preprocess` | 32.4 s |
| 120 s window, `--preprocess`, cold | 30.6 s |
| 120 s window, `--preprocess`, clip cached | 11.1 s |
| token cache | ~74 MB per 120 s (~1.2 GB extrapolated for the full file) |
| preprocess clip | ~94 MB per 120 s |
| conda env on disk | ~3.4 GB |

VRAM was not measured here; the Linux figures (~1.5 GB extraction, ~1 GB prediction) are the
closest reference.

`--preprocess` roughly breaks even on the first pass and pays back on re-runs over the same
window. Its output differs slightly from the in-process path (ffmpeg's resize is not
bit-identical), which is why the two keep separate token caches: over the same 7,193-frame
window, position correlation 0.9993, worst-case difference 8 on the 0-100 scale.

The README's 8K numbers are for a larger source; extraction is decode-bound, so throughput
tracks source resolution.

## Versions verified here

```
torch==2.13.0+cu130    torchcodec==0.15.0+cu130   FFmpeg 7.1.5 win64-gpl-shared
numpy==2.5.1           scipy==1.18.0              h5py==3.16.0
tqdm==4.70.0           safetensors==0.8.0         huggingface_hub==1.26.0
```
