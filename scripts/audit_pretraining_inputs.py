#!/usr/bin/env python3
"""Audit private pretraining input metadata without exposing input paths."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from omnimira.io import _read_sidecar, _validate_sidecar
from omnimira.schema import InputContractError


def audit_entries(entries: list[dict]) -> dict[str, int]:
    """Return aggregate V4 sidecar compliance counts; never retain paths."""
    report = {
        "entries_checked": 0,
        "nifti_entries": 0,
        "numpy_entries": 0,
        "missing_sidecar": 0,
        "invalid_sidecar": 0,
        "unsupported_path": 0,
    }
    for entry in entries:
        report["entries_checked"] += 1
        path = Path(str(entry["path"])).expanduser()
        modality = str(entry["modality"])
        if path.suffix == ".npy":
            report["numpy_entries"] += 1
            if not path.with_suffix(".npy.json").is_file():
                report["missing_sidecar"] += 1
                continue
            try:
                _validate_sidecar(_read_sidecar(path), modality)
            except InputContractError:
                report["invalid_sidecar"] += 1
        elif path.name.endswith((".nii", ".nii.gz")):
            report["nifti_entries"] += 1
        else:
            report["unsupported_path"] += 1
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    try:
        entries = json.loads(args.manifest.read_text(encoding="utf-8"))
        if not isinstance(entries, list):
            raise ValueError("manifest must be a JSON list")
        report = audit_entries(entries)
    except (OSError, ValueError, json.JSONDecodeError, KeyError) as exc:
        raise SystemExit(f"input audit failed: {exc}") from exc
    print(json.dumps(report, sort_keys=True))
    if args.strict and any(report[key] for key in (
        "missing_sidecar", "invalid_sidecar", "unsupported_path"
    )):
        raise SystemExit("input audit failed: V4 input contract violations detected")


if __name__ == "__main__":
    main()
