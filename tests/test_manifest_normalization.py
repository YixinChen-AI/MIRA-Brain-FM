from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts" / "build_public_pretraining_manifest.py"
    spec = importlib.util.spec_from_file_location("manifest_normalization", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_normalize_manifest_converts_legacy_path_and_subject_keys(tmp_path):
    source = tmp_path / "scan.npy"
    source.touch()
    entries = [
        {
            "npy_path": str(source),
            "subject": "ADNI_001",
            "cohort": "adni",
            "modality": "t1",
            "age": 71.2,
            "roi_volumes": {"aal3": "targets.npy"},
        }
    ]

    normalized = _module().normalize_entries(entries)

    assert normalized == [
        {
            "path": str(source.resolve()),
            "subject_id": "ADNI_001",
            "cohort": "adni",
            "modality": "t1",
            "age": 71.2,
            "roi_volumes": {"aal3": "targets.npy"},
        }
    ]


def test_normalize_manifest_rejects_missing_legacy_required_fields():
    import pytest

    with pytest.raises(ValueError, match="entry 0 is missing a path"):
        _module().normalize_entries([{"subject": "sub-1", "modality": "t1"}])
    with pytest.raises(ValueError, match="entry 0 is missing a subject identifier"):
        _module().normalize_entries([{"npy_path": "scan.npy", "modality": "t1"}])


def test_normalized_manifest_satisfies_public_sampler_contract(tmp_path):
    module = _module()
    entries = []
    for modality in ("t1", "av45", "fdg", "ct"):
        for index in range(16):
            path = tmp_path / f"{modality}-{index}.npy"
            path.touch()
            entries.append({"npy_path": str(path), "subject": f"sub-{modality}-{index}", "modality": modality})
    normalized = module.normalize_entries(entries)
    config = yaml.safe_load((ROOT / "configs" / "omnimira_3atlas.yaml").read_text())
    assert module.validate_normalized_entries(normalized, config)["scans"] == 64
