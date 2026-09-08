#!/usr/bin/env python3
"""Build a private OmniMIRA V4 normalized cache from MNI-space NIfTI sources."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from omnimira.io import load_volume
from omnimira.schema import MODEL_SHAPE, input_contract_sha256, modality_contract


OPTIONAL_FIELDS = ("cohort", "age", "sex", "time_id", "roi_volumes", "roi_distances")


def sidecar_metadata(modality: str) -> dict:
    stream = modality_contract(modality)
    return {
        "schema_version": "2.0",
        "input_contract_sha256": input_contract_sha256(),
        "template_sha256": stream.template_sha256,
        "brain_mask_sha256": stream.brain_mask_sha256,
        "space": stream.space,
        "shape": list(MODEL_SHAPE),
        "affine": stream.affine.tolist(),
        "qform_code": stream.qform_code,
        "sform_code": stream.sform_code,
        "orientation": stream.orientation,
        "modality": modality,
        "normalization": stream.normalization,
        "preprocessing_state": "normalized_v2",
    }


def _source_record(entry: dict, index: int) -> tuple[Path, str, str]:
    source = Path(str(entry.get("path", ""))).expanduser()
    subject_id = entry.get("subject_id")
    modality = entry.get("modality")
    if not source.is_file():
        raise ValueError(f"entry {index} source path does not exist")
    if not source.name.endswith((".nii", ".nii.gz")):
        raise ValueError(f"entry {index} requires a NIfTI source, not {source.suffix or 'an extensionless file'}")
    if subject_id is None or not str(subject_id).strip():
        raise ValueError(f"entry {index} is missing subject_id")
    if modality not in ("t1", "av45", "fdg", "ct"):
        raise ValueError(f"entry {index} has unsupported modality {modality!r}")
    return source, str(subject_id), str(modality)


def _cache_filename(source: Path, subject_id: str, modality: str, index: int) -> str:
    identity = f"{index}\0{subject_id}\0{modality}\0{source}".encode("utf-8")
    return f"{index:06d}_{hashlib.sha256(identity).hexdigest()[:16]}.npy"


def prepare_entries(entries: list[dict], cache_dir: Path, *, overwrite: bool = False) -> list[dict]:
    """Write V4-normalized cache files and return a training-ready private manifest."""
    if not isinstance(entries, list):
        raise ValueError("source manifest must be a JSON list")
    cache_dir = Path(cache_dir).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    output = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"entry {index} must be an object")
        source, subject_id, modality = _source_record(entry, index)
        destination = cache_dir / _cache_filename(source, subject_id, modality, index)
        sidecar = destination.with_suffix(".npy.json")
        if overwrite or not (destination.is_file() and sidecar.is_file()):
            volume = load_volume(source, modality)
            np.save(destination, np.asarray(volume, dtype=np.float32), allow_pickle=False)
            sidecar.write_text(
                json.dumps(sidecar_metadata(modality), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        record = {"path": str(destination.resolve()), "subject_id": subject_id, "modality": modality}
        for key in OPTIONAL_FIELDS:
            if entry.get(key) is not None:
                record[key] = entry[key]
        output.append(record)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="private raw-NIfTI manifest")
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="private normalized manifest")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        entries = json.loads(args.input.read_text(encoding="utf-8"))
        output = prepare_entries(entries, args.cache_dir, overwrite=args.overwrite)
    except (OSError, ValueError, json.JSONDecodeError, KeyError) as exc:
        raise SystemExit(f"cache preparation failed: {exc}") from exc
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cache_entries": len(output), "manifest": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
