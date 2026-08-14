"""Load a DNX head from an HF repo id, a .safetensors export, or a training .pt."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from src.backbone import pooling_from_num_tokens
from src.disposition_next import DispositionNext, extract_dnx_config

DEFAULT_CHECKPOINT = "herpaderpapotato/motion_from_mae_alt"
SAFETENSORS_FORMAT = "dnx_inference_v1"


def resolve_checkpoint(checkpoint: str | Path) -> Path:
    """A local path as-is, or an HF repo id downloaded to the hub cache."""
    path = Path(checkpoint)
    if path.exists():
        return path

    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(str(checkpoint), "model.safetensors"))


def load_checkpoint(path: Path, device: torch.device) -> dict:
    """Checkpoint dict with model_state_dict / ema_state_dict / model_config /
    data_config, from either container format."""
    if path.suffix != ".safetensors":
        return torch.load(path, map_location=device, weights_only=False)

    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device=str(device)) as f:
        metadata = f.metadata() or {}
        if metadata.get("format") != SAFETENSORS_FORMAT:
            raise SystemExit(f"{path}: not a {SAFETENSORS_FORMAT} file (format={metadata.get('format')!r})")
        ckpt = {k: json.loads(v) for k, v in metadata.items() if k not in ("format", "kind")}
        state = {k[len("model."):]: f.get_tensor(k) for k in f.keys() if k.startswith("model.")}
    # The export ties both keys to one stored copy of the weights.
    ckpt["model_state_dict"] = state
    ckpt["ema_state_dict"] = state
    return ckpt


def load_dnx_model(
    checkpoint: str | Path, device: torch.device, use_ema: bool = True,
) -> tuple[DispositionNext, dict, dict]:
    ckpt = load_checkpoint(resolve_checkpoint(checkpoint), device)
    model_config = ckpt["model_config"]
    model = DispositionNext(**extract_dnx_config(model_config))
    has_ema = "ema_state_dict" in ckpt
    model.load_state_dict(ckpt["ema_state_dict"] if (use_ema and has_ema) else ckpt["model_state_dict"])
    model.eval().to(device)
    print(
        f"Loaded DispositionNext ({'EMA' if (use_ema and has_ema) else 'raw'} weights): "
        f"epoch={ckpt.get('epoch', '?')} val_peak_f1_2={ckpt.get('val_peak_f1_2')}"
    )
    return model, model_config, ckpt.get("data_config", {})


def resolve_pooling_for_head(model: DispositionNext) -> str:
    """The head's own projector width decides the pooling layout -- it is what
    actually rejects a mismatched token pack."""
    return pooling_from_num_tokens(model.n_pool_tokens)
