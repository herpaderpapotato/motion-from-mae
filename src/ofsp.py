"""Create or extend an OpenFunscripter project (.ofsp) with a predicted track.

An .ofsp is CBOR of OFS's project states. ProjectState.binaryFunscriptData is a
bitsery buffer (little-endian) holding the scripts; see OFS_Project.h and
Funscript.h/FunscriptAction.h `serialize` for the layout mirrored here.
"""

from __future__ import annotations

import os
import shutil
import struct
import time
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# CBOR, restricted to what nlohmann::json::to_cbor emits
# --------------------------------------------------------------------------- #


def _cbor_head(major: int, n: int) -> bytes:
    if n < 24:
        return bytes([(major << 5) | n])
    if n <= 0xFF:
        return bytes([(major << 5) | 24, n])
    if n <= 0xFFFF:
        return bytes([(major << 5) | 25]) + struct.pack(">H", n)
    if n <= 0xFFFFFFFF:
        return bytes([(major << 5) | 26]) + struct.pack(">I", n)
    return bytes([(major << 5) | 27]) + struct.pack(">Q", n)


def cbor_dumps(v) -> bytes:
    if v is None:
        return b"\xf6"
    if isinstance(v, bool):
        return b"\xf5" if v else b"\xf4"
    if isinstance(v, int):
        return _cbor_head(0, v) if v >= 0 else _cbor_head(1, -1 - v)
    if isinstance(v, float):
        # nlohmann writes float32 when it round-trips exactly
        f32 = struct.pack(">f", v)
        if struct.unpack(">f", f32)[0] == v or v != v:
            return b"\xfa" + f32
        return b"\xfb" + struct.pack(">d", v)
    if isinstance(v, (bytes, bytearray)):
        return _cbor_head(2, len(v)) + bytes(v)
    if isinstance(v, str):
        b = v.encode("utf-8")
        return _cbor_head(3, len(b)) + b
    if isinstance(v, (list, tuple)):
        return _cbor_head(4, len(v)) + b"".join(cbor_dumps(x) for x in v)
    if isinstance(v, dict):
        out = [_cbor_head(5, len(v))]
        for k in sorted(v):
            out.append(cbor_dumps(str(k)))
            out.append(cbor_dumps(v[k]))
        return b"".join(out)
    raise TypeError(f"cannot CBOR-encode {type(v).__name__}")


def cbor_loads(buf: bytes):
    value, i = _cbor_item(buf, 0)
    if i != len(buf):
        raise ValueError(f"trailing {len(buf) - i} bytes after CBOR item")
    return value


def _cbor_item(buf: bytes, i: int):
    ib = buf[i]
    major, ai = ib >> 5, ib & 0x1F
    i += 1
    if major == 7:
        if ai == 20:
            return False, i
        if ai == 21:
            return True, i
        if ai in (22, 23):
            return None, i
        if ai == 25:
            return float(np.frombuffer(buf[i:i + 2], ">f2")[0]), i + 2
        if ai == 26:
            return struct.unpack(">f", buf[i:i + 4])[0], i + 4
        if ai == 27:
            return struct.unpack(">d", buf[i:i + 8])[0], i + 8
        raise ValueError(f"unsupported CBOR simple value {ai}")
    if ai < 24:
        n = ai
    elif ai in (24, 25, 26, 27):
        size = 1 << (ai - 24)
        n = int.from_bytes(buf[i:i + size], "big")
        i += size
    else:
        raise ValueError("indefinite-length CBOR is not supported")
    if major == 0:
        return n, i
    if major == 1:
        return -1 - n, i
    if major == 2:
        return bytes(buf[i:i + n]), i + n
    if major == 3:
        return buf[i:i + n].decode("utf-8"), i + n
    if major == 4:
        out = []
        for _ in range(n):
            x, i = _cbor_item(buf, i)
            out.append(x)
        return out, i
    if major == 5:
        out = {}
        for _ in range(n):
            k, i = _cbor_item(buf, i)
            out[k], i = _cbor_item(buf, i)
        return out, i
    # major 6: a tag; keep the tagged value
    return _cbor_item(buf, i)


# --------------------------------------------------------------------------- #
# bitsery (DefaultConfig, no bit packing)
# --------------------------------------------------------------------------- #


def _write_size(n: int) -> bytes:
    if n < 0x80:
        return bytes([n])
    if n < 0x4000:
        return bytes([(n >> 8) | 0x80, n & 0xFF])
    if n >= 0x40000000:
        raise ValueError(f"bitsery size {n} too large")
    return bytes([(n >> 24) | 0xC0, (n >> 16) & 0xFF]) + struct.pack("<H", n & 0xFFFF)


def _read_size(buf: bytes, i: int) -> tuple[int, int]:
    hb = buf[i]
    if hb < 0x80:
        return hb, i + 1
    lb = buf[i + 1]
    if hb & 0x40:
        lw = struct.unpack_from("<H", buf, i + 2)[0]
        return ((((hb & 0x3F) << 8) | lb) << 16) | lw, i + 4
    return ((hb & 0x7F) << 8) | lb, i + 2


def _growable(body: bytes) -> bytes:
    return struct.pack("<I", len(body) + 4) + body


def _text(s: str) -> bytes:
    b = s.encode("utf-8")
    return _write_size(len(b)) + b


def encode_script(actions: list[tuple[float, int]], rel_path: str, title: str) -> bytes:
    """One Funscript object: Growable{actions, currentPathRelative, title, Enabled}."""
    act = b"".join(_growable(struct.pack("<fhBB", t, p, 0, 0)) for t, p in actions)
    return _growable(_write_size(len(actions)) + act + _text(rel_path) + _text(title) + b"\x01")


def decode_scripts(blob: bytes) -> list[tuple[bytes, str]]:
    """Each script's raw Growable bytes, kept verbatim, with its relative path."""
    if not blob:
        return []
    total = struct.unpack_from("<I", blob, 0)[0]
    n, i = _read_size(blob, 4)
    scripts = []
    for _ in range(n):
        ptr_id, i = _read_size(blob, i)
        if ptr_id == 0:
            raise ValueError("null script pointer in binaryFunscriptData")
        size = struct.unpack_from("<I", blob, i)[0]
        raw = bytes(blob[i:i + size])
        n_act, j = _read_size(raw, 4)
        j += 12 * n_act
        plen, j = _read_size(raw, j)
        scripts.append((raw, raw[j:j + plen].decode("utf-8")))
        i += size
    if i > total:
        raise ValueError("binaryFunscriptData overran its own length")
    return scripts


def encode_scripts(raws: list[bytes]) -> bytes:
    # PointerLinkingContext numbers each distinct shared_ptr 1, 2, 3, ...
    body = _write_size(len(raws)) + b"".join(_write_size(k + 1) + r for k, r in enumerate(raws))
    return _growable(body)


# --------------------------------------------------------------------------- #


def _actions_from_funscript(funscript: dict) -> list[tuple[float, int]]:
    """Sorted, one action per float32 time: OFS keys its action set on atS alone."""
    by_time: dict[float, int] = {}
    for a in funscript["actions"]:
        t = float(np.float32(a["at"] / 1000.0))
        if t >= 0.0:
            by_time[t] = max(0, min(100, int(a["pos"])))
    return sorted(by_time.items())


def _relpath(path: Path, start: Path) -> str:
    try:
        return os.path.relpath(path.resolve(), start.resolve())
    except ValueError:  # different drive on Windows
        raise ValueError(f"{path} is not reachable by a relative path from {start}; "
                         "OFS stores project paths relative to the .ofsp") from None


def new_project_state(media_path: Path, ofsp_dir: Path, funscript: dict) -> dict:
    meta = funscript.get("metadata", {})
    actions = funscript["actions"]
    return {
        "metadata": {
            "type": "basic", "title": media_path.stem, "creator": str(meta.get("creator", "")),
            "script_url": "", "video_url": "", "tags": [], "performers": [],
            "description": "", "license": "", "notes": "",
            "duration": float(actions[-1]["at"] / 1000.0) if actions else 0.0,
            "topic_url": "", "topic_tags": [], "topic_creator": "", "topic_date": "",
        },
        "relativeMediaPath": _relpath(media_path, ofsp_dir),
        "activeTimer": 0.0,
        "lastPlayerPosition": 0.0,
        "activeScriptIdx": 0,
        "nudgeMetadata": False,
        "binaryFunscriptData": b"",
    }


def add_to_project(ofsp_path: Path, media_path: Path, script_path: Path, funscript: dict) -> str:
    """Add `funscript` (already written at `script_path`) to `ofsp_path` as a track.

    An existing project is copied to <name>.ofsp.<timestamp>.backup first; a
    track with the same relative path is replaced, anything else is appended.
    The added track becomes the active one. Returns a one-line status.
    """
    ofsp_dir = ofsp_path.parent
    rel = _relpath(script_path, ofsp_dir)
    raw = encode_script(_actions_from_funscript(funscript), rel, Path(rel).stem)

    backup = None
    if ofsp_path.exists():
        root = cbor_loads(ofsp_path.read_bytes())
        entry = root.get("ProjectState") if isinstance(root, dict) else None
        if not isinstance(entry, dict) or not isinstance(entry.get("State"), dict):
            raise ValueError(f"{ofsp_path} has no ProjectState")
        state = entry["State"]
        scripts = decode_scripts(state.get("binaryFunscriptData") or b"")
        stamp = time.strftime("%Y%m%d-%H%M%S")
        backup = ofsp_path.with_name(f"{ofsp_path.name}.{stamp}.backup")
        n = 1
        while backup.exists():
            backup = ofsp_path.with_name(f"{ofsp_path.name}.{stamp}-{n}.backup")
            n += 1
        shutil.copy2(ofsp_path, backup)
    else:
        state = new_project_state(media_path, ofsp_dir, funscript)
        root = {"ProjectState": {"TypeName": "ProjectState", "State": state}}
        scripts = []

    paths = [p for _, p in scripts]
    raws = [r for r, _ in scripts]
    replaced = rel in paths
    idx = paths.index(rel) if replaced else len(raws)
    if replaced:
        raws[idx] = raw
    else:
        raws.append(raw)
    state["binaryFunscriptData"] = encode_scripts(raws)
    state["activeScriptIdx"] = idx

    tmp = ofsp_path.with_name(ofsp_path.name + ".partial")
    tmp.write_bytes(cbor_dumps(root))
    tmp.replace(ofsp_path)

    status = f"ofsp -> {ofsp_path.name} ({'replaced' if replaced else 'added'} track {idx + 1}/{len(raws)}"
    return status + (f", backup {backup.name})" if backup else ", new)")
