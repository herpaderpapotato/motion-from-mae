# Linux setup (conda)

Verified on Ubuntu 24.04 (WSL2), NVIDIA driver 591.44 / CUDA 13.1, RTX 4090,
conda 25.5.1, Python 3.12.

`predict.py` is CUDA-only: torchcodec decodes on the GPU via NVDEC, so a working
NVIDIA driver and an FFmpeg install that torchcodec can load are both hard
requirements.

## 1. System packages

torchcodec does **not** bundle FFmpeg — it dlopens the system shared libraries and
supports major versions 4 through 8. Ubuntu 24.04's `ffmpeg` (6.1, `libavutil.so.58`)
works as-is.

```bash
sudo apt install ffmpeg
```

`ffprobe` also has to be on `PATH` — `src/extract.py` shells out to it for the exact
`r_frame_rate`, and `--preprocess` uses `ffmpeg` directly.

For `--preprocess`, the FFmpeg build additionally needs a `*_cuvid` decoder for your
source codec and, ideally, an NVENC encoder. Ubuntu's build has both:

```bash
ffmpeg -hide_banner -decoders | grep cuvid    # h264_cuvid, hevc_cuvid, av1_cuvid, ...
ffmpeg -hide_banner -encoders | grep nvenc    # h264_nvenc, hevc_nvenc, av1_nvenc
```

Without them `--preprocess` raises `PreprocessUnavailable`; the default (no
`--preprocess`) path is unaffected.

## 2. Create the environment

```bash
conda create -n motion-from-mae python=3.12 -y
conda activate motion-from-mae
```

## 3. Install torch + torchcodec

Install both from the same PyTorch CUDA index so the versions match — `requirements.txt`
is unpinned, and a plain `pip install torch torchcodec` from PyPI can land on an
incompatible pair.

```bash
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0 torchcodec==0.11.1
```

cu128 wheels run fine against a newer driver (CUDA is forward-compatible on the driver
side). Check the [torchcodec compatibility table](https://github.com/pytorch/torchcodec#installing-torchcodec)
if you pick different versions.

## 4. Install the remaining requirements

```bash
pip install -r requirements.txt
```

(`torch` and `torchcodec` are already satisfied by step 3, so pip leaves them alone.)

## 5. Fix the NPP dependency

**This step is required on Ubuntu 24.04.** `libtorchcodec_core6.so` links against
`libnppicc.so.12` (NVIDIA Performance Primitives, used for the NVDEC NV12→RGB
conversion) and carries no `RPATH`, so the loader falls back to the distro copy at
`/lib/x86_64-linux-gnu/libnppicc.so.12`. That copy is from CUDA 12.0 and is missing the
symbols torchcodec wants, which surfaces as an unhelpful import error:

```
RuntimeError: Could not load libtorchcodec. ...
  libtorchcodec_core6.so: undefined symbol:
  nppiNV12ToRGB_8u_ColorTwist32f_P2C3R_Ctx, version libnppicc.so.12
```

Note the error lists a failure for *every* FFmpeg major version; only the one matching
your installed FFmpeg (version 6 here) is the real one — the rest are just
"library not found".

Install a current NPP into the env and put it ahead of the system copy on
`LD_LIBRARY_PATH`:

```bash
pip install nvidia-npp-cu12==12.3.3.100

mkdir -p "$CONDA_PREFIX/etc/conda/activate.d" "$CONDA_PREFIX/etc/conda/deactivate.d"

cat > "$CONDA_PREFIX/etc/conda/activate.d/torchcodec_npp.sh" <<'EOF'
export _TORCHCODEC_NPP_OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/npp/lib:${LD_LIBRARY_PATH:-}"
EOF

cat > "$CONDA_PREFIX/etc/conda/deactivate.d/torchcodec_npp.sh" <<'EOF'
export LD_LIBRARY_PATH="${_TORCHCODEC_NPP_OLD_LD_LIBRARY_PATH:-}"
unset _TORCHCODEC_NPP_OLD_LD_LIBRARY_PATH
EOF
```

Re-activate the environment so the hook runs:

```bash
conda deactivate && conda activate motion-from-mae
```

## 6. Verify

```bash
python - <<'EOF'
import torch
from torchcodec.decoders import VideoDecoder
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0))
EOF
```

`torch.cuda.get_device_name(0)` is worth checking on a multi-GPU box: torch enumerates
in PCI bus order, which need not match `nvidia-smi`'s ordering. Pin a specific card with
`CUDA_VISIBLE_DEVICES=1` if the default is the wrong one — `--device` only accepts
`cuda`, not `cuda:1`.

## 7. Run

Smoke-test on a short window before committing to a full video:

```bash
conda activate motion-from-mae
cd /path/to/motion-from-mae

python predict.py --video video.mp4 --out data/smoketest.funscript \
  --vr --frame-view crop --start-time 60 --duration 30
```

Then the full run:

```bash
python predict.py --video video.mp4 --out video.funscript --vr --frame-view crop
```

First run downloads the head (`herpaderpapotato/motion_from_mae`) and the backbone it
names (`herpaderpapotato/motion_from_mae_extract`) into the HF cache
(`~/.cache/huggingface`, or `$HF_HOME`). The "unauthenticated requests to the HF Hub"
warning is harmless; set `HF_TOKEN` to silence it.

`predict.py` creates `data/` in the current working directory, so run it from the repo
root. Caches land in `data/video_token_cache/` and, with `--preprocess`,
`data/video_preprocess_cache/`. Output never overwrites: a second run writes
`video.001.funscript`, then `.002`.

## Measured on this machine

6144x3072 HEVC SBS, 59.94 fps, 39 min, 8.9 GB, on the RTX 4090:

| | |
|---|---|
| token extraction | ~200 frame/s (~12 min for the full file) |
| head prediction | ~0.5 s per 30 s window |
| VRAM | ~1.5 GB extraction, ~1 GB prediction |
| token cache | ~19 MB per 30 s of video (~1.5 GB for the full file) |

The README's 8K numbers are for a larger source; extraction is decode-bound, so
throughput tracks source resolution.

`--preprocess` costs an ffmpeg transcode pass up front and pays it back on re-runs over
the same window. Verified working here; it produces slightly different numbers than the
in-process path (ffmpeg's resize is not bit-identical — position correlation ~0.99),
which is why the two keep separate token caches.

## Troubleshooting

**`Could not load libtorchcodec`** — see step 5. If NPP is not the culprit, check that
`ffmpeg -version` reports 4–8 and that `libavutil.so.*` is in the loader path.

**`parser.error: CUDA is required`** or **`torch.cuda.is_available() is False`** — the
CPU-only torch wheel got installed. Confirm with `python -c "import torch;
print(torch.__version__)"`; it must end in `+cu128`, not a bare version. Reinstall from
the `--index-url` in step 3.

**`ffprobe` warning about `r_frame_rate`** — `ffprobe` is missing from `PATH`. The run
continues on the container's average frame rate, which drifts over a long file and
misaligns the funscript timestamps. Install `ffmpeg` rather than ignoring it.

**CUDA out of memory** — decode batch size is derived from the source resolution
(`EyeDecoder._auto_decode_batch`, budgeted to ~2 GB). If another process is holding
VRAM, free it or point at a different GPU with `CUDA_VISIBLE_DEVICES`.

## Recreating the environment

```bash
conda create -n motion-from-mae python=3.12 -y
conda activate motion-from-mae
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0 torchcodec==0.11.1
pip install -r requirements.txt
pip install nvidia-npp-cu12==12.3.3.100
# then the activate.d hook from step 5
```

Versions installed and verified here:

```
torch==2.11.0+cu128    torchcodec==0.11.1+cu128    nvidia-npp-cu12==12.3.3.100
numpy==2.5.1           scipy==1.18.0               h5py==3.16.0
tqdm==4.70.0           safetensors==0.8.0          huggingface_hub==1.26.0
```
