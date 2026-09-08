"""Export a clean, standalone OmniMIRA public repository tree."""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1]
EXCLUDED_NAMES = {".git", ".pytest_cache", "build", "omnimira.egg-info", "__pycache__"}
EXCLUDED_FILES = {
    Path("CITATION.cff"),
    Path("configs/release_contract.yaml"),
    Path("scripts/audit_release.py"),
    Path("scripts/generate_golden_reference.py"),
    Path("scripts/verify_release_contract.py"),
    Path("checkpoints/release_checkpoint_inventory.json"),
    Path("docs/provenance/manuscript_v4_main.tex"),
    Path("tests/test_release_integration.py"),
    Path("tests/test_release_metadata.py"),
    Path("tests/test_manuscript_contract.py"),
    Path("tests/test_demo.py"),
    Path("tests/test_example_smoke.py"),
}
PUBLIC_REWRITES = {
    Path("checkpoints/README.md"): (
        "docs/provenance/manuscript_v4_main.tex",
        "pending committed V4 manuscript revision",
    ),
}


def _excluded(relative: Path) -> bool:
    if any(part in EXCLUDED_NAMES for part in relative.parts):
        return True
    if relative in EXCLUDED_FILES:
        return True
    if relative.parts[:1] == ("demo",):
        return True
    if relative.parts[:2] == ("assets", "atlases"):
        return relative.name != "atlas_label_sources.json"
    if relative.parts[:2] == ("assets", "templates") and "_las_" in relative.name:
        return True
    if relative.parts[:2] == ("tests", "fixtures"):
        return True
    return False


def export_tree(destination: Path) -> int:
    destination = destination.expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"output directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    copied = 0
    for source in SOURCE.rglob("*"):
        relative = source.relative_to(SOURCE)
        if _excluded(relative):
            continue
        target = destination / relative
        if source.is_dir():
            target.mkdir(exist_ok=True)
        elif source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            rewrite = PUBLIC_REWRITES.get(relative)
            if rewrite is None:
                shutil.copy2(source, target)
            else:
                text = source.read_text(encoding="utf-8")
                target.write_text(text.replace(*rewrite), encoding="utf-8")
            copied += 1
    print(f"exported {copied} files to {destination}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        return export_tree(args.output)
    except (OSError, ValueError) as exc:
        print(f"public tree export failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
