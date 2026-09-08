#!/usr/bin/env python3
"""Validate a manifest before launching formal OmniMIRA public pretraining."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[1]


def validate_entries(entries, config, *, check_paths=True):
    """Return a summary or raise ValueError for a manifest that cannot train."""
    if not isinstance(entries, list) or not entries:
        raise ValueError("manifest must be a non-empty JSON list")
    modalities = tuple(config["data"]["modality_order"])
    sampler = config["sampler"]
    counts = Counter()
    participants = set()
    errors = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            errors.append(f"entry {index} must be an object")
            continue
        modality = entry.get("modality")
        if modality not in modalities:
            errors.append(f"entry {index} has unsupported modality {modality!r}")
            continue
        counts[modality] += 1
        subject = entry.get("subject_id")
        if subject is None or str(subject).strip() == "":
            errors.append(f"entry {index} is missing subject_id")
        else:
            participants.add(str(subject))
        path = entry.get("path")
        if not isinstance(path, str) or not path:
            errors.append(f"entry {index} is missing path")
        elif check_paths and not Path(path).expanduser().is_file():
            errors.append(f"entry {index} path does not exist: {path}")
    for modality in modalities:
        if counts[modality] < sampler["n_per_modality"]:
            errors.append(
                f"{modality} has {counts[modality]} scans; need at least "
                f"{sampler['n_per_modality']}"
            )
    if len(participants) < sampler["min_unique_subjects"]:
        errors.append(
            f"manifest has {len(participants)} unique participants; need at least "
            f"{sampler['min_unique_subjects']}"
        )
    if errors:
        raise ValueError("; ".join(errors))
    return {
        "scans": len(entries),
        "modalities": {name: counts[name] for name in modalities},
        "unique_participants": len(participants),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--config", type=Path, default=REPO / "configs" / "omnimira_3atlas.yaml"
    )
    parser.add_argument(
        "--skip-path-check", action="store_true",
        help="validate manifest structure and sampling constraints without file I/O",
    )
    args = parser.parse_args()
    try:
        entries = json.loads(args.manifest.read_text(encoding="utf-8"))
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        summary = validate_entries(entries, config, check_paths=not args.skip_path_check)
    except (OSError, ValueError, json.JSONDecodeError, KeyError) as exc:
        raise SystemExit(f"manifest validation failed: {exc}") from exc
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
