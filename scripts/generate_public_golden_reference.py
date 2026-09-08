"""Generate deterministic frozen ROI references from a public OmniMIRA checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import sys
from pathlib import Path

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from omnimira.atlas import ATLAS_LABEL_HASHES, ATLAS_ORDER, atlas_hashes
from omnimira.inference import MODEL_VERSION, extract_roi_features, load_omnimira
from omnimira.schema import MODEL_SHAPE, file_sha256


MODALITIES = ("t1", "av45", "fdg", "ct")


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _set_deterministic(seed: int) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--atlas-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=REPO / "configs" / "omnimira_3atlas.yaml")
    parser.add_argument("--seed", type=int, default=20260715)
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    input_dir = args.input_dir.expanduser().resolve()
    config = args.config.expanduser().resolve()
    if not checkpoint.is_file() or not input_dir.is_dir() or not config.is_file():
        parser.error("checkpoint, input directory and config must exist")
    os.environ["OMNIMIRA_ATLAS_DIR"] = str(args.atlas_dir.expanduser().resolve())
    _set_deterministic(args.seed)
    try:
        model = load_omnimira(checkpoint, config, device="cpu")
        arrays: dict[str, np.ndarray] = {}
        inputs: dict[str, dict[str, object]] = {}
        for modality in MODALITIES:
            path = input_dir / f"{modality}.npy"
            if not path.is_file():
                raise FileNotFoundError(f"missing reference input: {path.name}")
            volume = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
            if tuple(volume.shape) != MODEL_SHAPE:
                raise ValueError(f"{path.name} has shape {volume.shape}; expected {MODEL_SHAPE}")
            features = extract_roi_features(model, volume, modality=modality)
            inputs[modality] = {
                "filename": path.name,
                "sha256": file_sha256(path),
                "shape": list(volume.shape),
                "dtype": str(volume.dtype),
            }
            for atlas in ATLAS_ORDER:
                arrays[f"{modality}__{atlas}"] = np.ascontiguousarray(features[atlas])
    except (FileNotFoundError, ValueError) as exc:
        print(f"golden reference generation failed: {exc}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **{key: arrays[key] for key in sorted(arrays)})
    manifest = {
        "schema_version": 1,
        "model_version": MODEL_VERSION,
        "seed": args.seed,
        "checkpoint": {
            "filename": checkpoint.name,
            "sha256": file_sha256(checkpoint),
            "bytes": checkpoint.stat().st_size,
        },
        "config": {
            "path": "configs/omnimira_3atlas.yaml",
            "sha256": file_sha256(config),
        },
        "inputs": inputs,
        "atlases": {
            name: {
                "sha256": atlas_hashes()[name],
                "label_sha256": ATLAS_LABEL_HASHES[name],
            }
            for name in ATLAS_ORDER
        },
        "arrays": {
            key: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": _array_sha256(value),
            }
            for key, value in sorted(arrays.items())
        },
        "output": {
            "filename": args.output.name,
            "sha256": file_sha256(args.output),
            "bytes": args.output.stat().st_size,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": "cpu",
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
