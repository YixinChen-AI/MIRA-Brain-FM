from __future__ import annotations

import subprocess
import sys
import os
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "public_release_contract.yaml"
AUDIT = ROOT / "scripts" / "audit_public_release.py"
EXPORT = ROOT / "scripts" / "export_public_tree.py"
GOLDEN = ROOT / "scripts" / "generate_public_golden_reference.py"


def test_public_release_contract_contains_only_new_release_requirements() -> None:
    text = CONTRACT.read_text(encoding="utf-8")
    contract = yaml.safe_load(text)

    assert contract["status"] == "public_retraining_pending"
    assert contract["model_config"] == "configs/omnimira_3atlas.yaml"
    assert contract["checkpoint_manifest"] == "checkpoints/manifest.json"
    assert contract["publication_gates"] == [
        "public_checkpoint_training",
        "checkpoint_redistribution",
        "atlas_redistribution",
        "self_contained_result_evidence",
        "committed_manuscript_revision",
    ]
    assert "ep150" not in text
    assert "/Volumes/" not in text
    assert "/Users/" not in text
    assert "/share/home/" not in text


def test_public_wheel_metadata_does_not_bundle_atlas_binaries_or_label_tables() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "assets/atlases/*.nii" not in pyproject
    assert "assets/atlases/*.txt" not in pyproject
    assert "assets/atlases/*.json" in pyproject


def test_public_audit_passes_metadata_mode_and_blocks_publication_before_retraining() -> None:
    metadata = subprocess.run(
        [sys.executable, str(AUDIT)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert metadata.returncode == 0, metadata.stderr
    assert "metadata-only audit passed" in metadata.stdout

    publication = subprocess.run(
        [sys.executable, str(AUDIT), "--publication"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert publication.returncode == 1
    assert "publication gate remains open: public_checkpoint_training" in publication.stderr


def test_public_tree_export_excludes_internal_evidence_and_third_party_atlas_assets(tmp_path) -> None:
    destination = tmp_path / "OmniMIRA"
    result = subprocess.run(
        [sys.executable, str(EXPORT), "--output", str(destination)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (destination / "README.md").is_file()
    assert (destination / "configs" / "public_release_contract.yaml").is_file()
    assert not (destination / "configs" / "release_contract.yaml").exists()
    assert not (destination / "checkpoints" / "release_checkpoint_inventory.json").exists()
    assert not (destination / "assets" / "atlases" / "AAL3v1_1mm.nii").exists()
    assert not (destination / "assets" / "atlases" / "AAL3v1_labels.txt").exists()
    assert not (destination / "assets" / "templates" / "mni152_las_model_grid.nii.gz").exists()
    assert not (destination / "docs" / "provenance" / "manuscript_v4_main.tex").exists()
    assert not (destination / "CITATION.cff").exists()
    assert not (destination / "demo").exists()
    assert not (destination / "tests" / "test_demo.py").exists()
    assert not (destination / "tests" / "test_example_smoke.py").exists()
    assert (destination / "scripts" / "audit_public_release.py").is_file()
    assert (destination / "scripts" / "generate_public_golden_reference.py").is_file()
    assert not (destination / "scripts" / "generate_golden_reference.py").exists()
    assert "pending committed V4 manuscript revision" in (
        destination / "checkpoints" / "README.md"
    ).read_text(encoding="utf-8")

    for path in destination.rglob("*"):
        if (
            path.is_file()
            and "tests" not in path.relative_to(destination).parts
            and path.suffix in {".py", ".md", ".yaml", ".json", ".txt"}
        ):
            text = path.read_text(encoding="utf-8")
            assert "/Volumes/" not in text
            assert "/Users/" not in text
            assert "/share/home/" not in text


def test_exported_public_tree_skips_atlas_tests_without_external_atlases(tmp_path) -> None:
    destination = tmp_path / "OmniMIRA"
    exported = subprocess.run(
        [sys.executable, str(EXPORT), "--output", str(destination)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert exported.returncode == 0, exported.stderr

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_atlas.py", "-q"],
        cwd=destination,
        env={**os.environ, "PYTHONPATH": ".", "OMNIMIRA_ATLAS_DIR": ""},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "skipped" in result.stdout


def test_exported_public_tree_rejects_invalid_manifest_before_atlas_resolution(tmp_path) -> None:
    destination = tmp_path / "OmniMIRA"
    exported = subprocess.run(
        [sys.executable, str(EXPORT), "--output", str(destination)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert exported.returncode == 0, exported.stderr

    invalid_manifest = tmp_path / "invalid.json"
    invalid_manifest.write_text("[]", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "scripts/pretrain.py",
            "--manifest",
            str(invalid_manifest),
            "--device",
            "cpu",
        ],
        cwd=destination,
        env={**os.environ, "PYTHONPATH": "."},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "manifest validation failed" in result.stderr


def test_public_golden_generator_records_current_checkpoint_without_local_paths(
    tmp_path, normalized_npy_factory
) -> None:
    from omnimira.inference import _build_omnimira_for_tests

    checkpoint = tmp_path / "omnimira_public_v1.pt"
    model = _build_omnimira_for_tests(ROOT / "configs" / "omnimira_3atlas.yaml")
    torch.save({"model_version": "omnimira-public-v1", "model": model.state_dict()}, checkpoint)

    inputs = tmp_path / "inputs"
    inputs.mkdir()
    for index, modality in enumerate(("t1", "av45", "fdg", "ct")):
        source = normalized_npy_factory(tmp_path, modality, index)
        source.rename(inputs / f"{modality}.npy")
    output = tmp_path / "golden_summaries.npz"
    manifest = tmp_path / "golden_manifest.json"

    result = subprocess.run(
        [
            sys.executable,
            str(GOLDEN),
            "--checkpoint",
            str(checkpoint),
            "--input-dir",
            str(inputs),
            "--atlas-dir",
            str(ROOT / "assets" / "atlases"),
            "--output",
            str(output),
            "--manifest",
            str(manifest),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    record = json.loads(manifest.read_text(encoding="utf-8"))
    assert record["checkpoint"]["sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert "path_at_generation" not in json.dumps(record)
    with np.load(output, allow_pickle=False) as values:
        assert set(values.files) == {
            f"{modality}__{atlas}"
            for modality in ("t1", "av45", "fdg", "ct")
            for atlas in ("aal3", "ho69", "yeo7")
        }

    spec = importlib.util.spec_from_file_location("audit_public_release", AUDIT)
    audit = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(audit)
    assert audit._golden_evidence_errors(
        manifest,
        output,
        {"sha256": record["checkpoint"]["sha256"]},
    ) == []
