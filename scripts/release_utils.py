"""Shared primitives for OmniMIRA release verification scripts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import yaml


STATE_WRAPPERS = ("model", "state_dict", "model_state_dict")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"manifest is not valid JSON: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError("manifest root must be a mapping")
    checkpoints = value.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        raise ValueError("manifest checkpoints must be a non-empty list")
    for index, record in enumerate(checkpoints):
        if not isinstance(record, Mapping):
            raise ValueError(f"manifest checkpoint record {index} must be a mapping")
    return dict(value)


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"{path} is not valid YAML: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} root must be a mapping")
    return dict(value)


def extract_checkpoint_state(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint root is not a mapping")
    for key in STATE_WRAPPERS:
        if key not in checkpoint:
            continue
        candidate = checkpoint[key]
        if not isinstance(candidate, Mapping):
            raise ValueError(f"checkpoint {key} is not a state mapping")
        if not candidate or any(not isinstance(value, torch.Tensor) for value in candidate.values()):
            raise ValueError(f"checkpoint {key} contains non-tensor or empty state")
        return candidate
    if checkpoint and all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        return checkpoint
    raise ValueError("checkpoint has no tensor model state mapping")


def resolve_path(value: str | Path, *, base: str | Path | None = None) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(base).expanduser() / path if base is not None else Path.cwd() / path
    return path.resolve()


def safe_filename(value: Any) -> str:
    if not isinstance(value, str) or not value or value in {".", ".."}:
        raise ValueError("checkpoint filename must be a non-empty basename")
    if "/" in value or "\\" in value or Path(value).name != value:
        raise ValueError("checkpoint filename must not contain a path")
    return value
