"""Load a DNX head from an HF repo id, a .safetensors export, or a training .pt."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from src.backbone import pooling_from_num_tokens
from src.disposition_next import DispositionNext, extract_dnx_config, upgrade_dnx_state

DEFAULT_CHECKPOINT = "herpaderpapotato/motion_from_mae_alt"
SAFETENSORS_FORMAT = "dnx_inference_v1"


def resolve_checkpoint(checkpoint: str | Path, revision: str | None = None) -> Path:
    """A local path as-is, or an HF repo id resolved against the hub cache.

    Without `revision` the hub's current commit is used, so a re-published head
    is picked up rather than served stale from the cache.
    """
    path = Path(checkpoint)
    if path.exists():
        return path

    from huggingface_hub import hf_hub_download

    from src.hub import resolve

    return resolve(
        lambda **kw: hf_hub_download(str(checkpoint), "model.safetensors", **kw),
        str(checkpoint), f"head {checkpoint}", revision=revision,
    )


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


def model_hash(path: Path, model_config: dict, data_config: dict, use_ema: bool) -> str:
    """sha256 over the weight file and the configs it was built with.

    Written into every funscript: it answers "which model produced this" for a
    local export, a training .pt and a hub revision alike.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    h.update(json.dumps({"model_config": model_config, "data_config": data_config,
                         "weights": "ema" if use_ema else "raw"},
                        sort_keys=True, default=str).encode("utf-8"))
    return h.hexdigest()


def load_dnx_model(
    checkpoint: str | Path, device: torch.device, use_ema: bool = True,
    revision: str | None = None,
) -> tuple[DispositionNext, dict, dict, dict]:
    """Returns (model, model_config, data_config, provenance) -- the last a
    small dict of what produced the weights, for the output's metadata."""
    from src.hub import hub_revision

    path = resolve_checkpoint(checkpoint, revision)
    ckpt = load_checkpoint(path, device)
    model_config = ckpt["model_config"]
    model = DispositionNext(**extract_dnx_config(model_config))
    has_ema = "ema_state_dict" in ckpt
    state = ckpt["ema_state_dict"] if (use_ema and has_ema) else ckpt["model_state_dict"]
    model.load_state_dict(upgrade_dnx_state(state, model))
    model.eval().to(device)
    f1 = ckpt.get("val_peak_f1_2")
    revision = hub_revision(path)
    weights = "ema" if (use_ema and has_ema) else "raw"
    data_config = ckpt.get("data_config", {})
    provenance = {
        "checkpoint": str(checkpoint),
        "checkpoint_revision": revision,
        "weights": weights,
        "epoch": ckpt.get("epoch"),
        "model_hash": model_hash(path, model_config, data_config, use_ema and has_ema),
    }
    print(
        f"Head: {checkpoint}" + (f" @ {revision[:7]}" if revision else "")
        + f", {'EMA' if weights == 'ema' else 'raw'} weights, epoch {ckpt.get('epoch', '?')}"
        + (f", val peak F1@2 {f1:.4f}" if isinstance(f1, float) else "")
        + f", sha {provenance['model_hash'][:12]}"
    )
    return model, model_config, data_config, provenance


def resolve_interleave_for_head(model: DispositionNext, data_cfg: dict | None = None) -> bool:
    """interleave_input from the head's frames_per_slot, cross-checked against data_config."""
    derived = model.frames_per_slot == 1
    declared = (data_cfg or {}).get("interleave_input")
    if declared is not None and bool(declared) != derived:
        raise SystemExit(
            f"checkpoint is inconsistent: data_config records interleave_input={declared} but the head "
            f"predicts {model.frames_per_slot} frame(s) per slot."
        )
    return derived


def resolve_pooling_for_head(model: DispositionNext) -> str:
    """The head's own projector width decides the pooling layout -- it is what
    actually rejects a mismatched token pack."""
    return pooling_from_num_tokens(model.n_pool_tokens)
