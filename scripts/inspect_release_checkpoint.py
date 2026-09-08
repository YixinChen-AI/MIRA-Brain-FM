#!/usr/bin/env python3
"""Inventory release checkpoint files without constructing a model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from release_utils import extract_checkpoint_state, resolve_path, sha256_file


def inspect(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = extract_checkpoint_state(checkpoint)
    non_tensors = sorted(key for key, value in state.items() if not isinstance(value, torch.Tensor))
    if non_tensors:
        raise ValueError(f"non-tensor model state entries: {non_tensors}")
    state_keys = sorted(state)
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "state_keys": state_keys,
        "tensor_shapes": {key: list(state[key].shape) for key in state_keys},
        "tensor_dtypes": {key: str(state[key].dtype) for key in state_keys},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    records = [inspect(resolve_path(path)) for path in args.checkpoints]
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
