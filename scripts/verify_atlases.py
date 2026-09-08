"""Verify atlas binaries supplied for OmniMIRA feature extraction."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from omnimira.atlas import (
    ATLAS_LABEL_HASHES,
    ATLAS_ORDER,
    ATLAS_RESOURCES,
    atlas_label_path,
    atlas_path,
)
from omnimira.schema import file_sha256


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify OmniMIRA atlas files against the fixed public contract."
    )
    parser.add_argument(
        "--atlas-dir",
        type=Path,
        help="directory containing the three atlas files; overrides OMNIMIRA_ATLAS_DIR",
    )
    args = parser.parse_args()
    if args.atlas_dir is not None:
        os.environ["OMNIMIRA_ATLAS_DIR"] = str(args.atlas_dir.expanduser())

    try:
        report = {}
        for name in ATLAS_ORDER:
            path = atlas_path(name)
            label_path = atlas_label_path(name)
            report[name] = {
                "path": str(path),
                "sha256": file_sha256(path),
                "label_path": str(label_path),
                "label_sha256": ATLAS_LABEL_HASHES[name],
                "n_rois": ATLAS_RESOURCES[name][2],
            }
    except (FileNotFoundError, ValueError) as exc:
        print(f"atlas verification failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
