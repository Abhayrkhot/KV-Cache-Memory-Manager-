"""Bundled model configs and resolution of user-supplied model references."""

from __future__ import annotations

import json
from functools import cache
from importlib import resources
from pathlib import Path
from typing import Any

from pagedkv.spec import KVSpec, from_hf_config

_PKG = "pagedkv.configs"


@cache
def preset_names() -> tuple[str, ...]:
    files = resources.files(_PKG).iterdir()
    return tuple(
        sorted(
            f.name[:-5] for f in files if f.name.endswith(".json") and not f.name.startswith("_")
        )
    )


@cache
def _params_index() -> dict[str, int]:
    path = resources.files(_PKG) / "_params.json"
    if not path.is_file():
        return {}
    return {k: int(v) for k, v in json.loads(path.read_text()).items()}


def load_preset(name: str) -> dict[str, Any]:
    if name not in preset_names():
        raise KeyError(f"unknown preset {name!r}; available: {', '.join(preset_names())}")
    data: dict[str, Any] = json.loads((resources.files(_PKG) / f"{name}.json").read_text())
    return data


def preset_params(name: str) -> int | None:
    """Exact parameter count measured from the architecture (see scripts/validate_estimator.py)."""
    return _params_index().get(name)


def load_config(ref: str) -> tuple[str, dict[str, Any]]:
    """Resolve a preset name, a path to ``config.json``, or a Hugging Face repo id."""
    if ref in preset_names():
        return ref, load_preset(ref)
    path = Path(ref).expanduser()
    if path.is_dir():
        path = path / "config.json"
    if path.is_file():
        cfg: dict[str, Any] = json.loads(path.read_text())
        return path.parent.name, cfg
    if "/" in ref:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as err:
            raise ValueError(f"{ref!r} looks like a Hub repo id; install huggingface_hub") from err
        cfg = json.loads(Path(hf_hub_download(ref, "config.json")).read_text())
        return ref, cfg
    raise ValueError(
        f"{ref!r} is not a preset, a config.json path, or a Hub repo id. "
        f"Presets: {', '.join(preset_names())}"
    )


def load_spec(ref: str) -> KVSpec:
    name, cfg = load_config(ref)
    return from_hf_config(cfg, name)
