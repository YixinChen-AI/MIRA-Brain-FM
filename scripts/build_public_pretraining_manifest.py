#!/usr/bin/env python3
"""Convert a private source-data manifest to the OmniMIRA V4 training schema."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[1]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate_pretraining_manifest import validate_entries


OPTIONAL_FIELDS = (
    "cohort",
    "age",
    "sex",
    "time_id",
    "roi_volumes",
    "roi_distances",
)


def _first_present(entry: dict, keys: tuple[str, ...]):
    for key in keys:
        value = entry.get(key)
        if value is not None and str(value).strip() != "":
            return value
    return None


def normalize_entries(entries: list[dict]) -> list[dict]:
    """Rename source-manifest fields without changing the underlying data split."""
    if not isinstance(entries, list):
        raise ValueError("manifest must be a JSON list")

    normalized = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"entry {index} must be an object")
        source_path = _first_present(entry, ("path", "npy_path"))
        if source_path is None:
            raise ValueError(f"entry {index} is missing a path")
        subject_id = _first_present(entry, ("subject_id", "subject"))
        if subject_id is None:
            raise ValueError(f"entry {index} is missing a subject identifier")
        modality = _first_present(entry, ("modality",))
        if modality is None:
            raise ValueError(f"entry {index} is missing modality")

        record = {
            "path": str(Path(str(source_path)).expanduser().resolve()),
            "subject_id": str(subject_id),
            "modality": str(modality),
        }
        for key in OPTIONAL_FIELDS:
            if entry.get(key) is not None:
                record[key] = entry[key]
        normalized.append(record)
    return normalized


def validate_normalized_entries(entries: list[dict], config: dict) -> dict:
    """Apply the V4 sampler contract to a normalized manifest."""
    return validate_entries(entries, config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--config", type=Path, default=REPO / "configs" / "omnimira_3atlas.yaml"
    )
    args = parser.parse_args()
    try:
        source = json.loads(args.input.read_text(encoding="utf-8"))
        entries = normalize_entries(source)
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        summary = validate_normalized_entries(entries, config)
    except (OSError, ValueError, json.JSONDecodeError, KeyError) as exc:
        raise SystemExit(f"manifest normalization failed: {exc}") from exc

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
