import importlib.util
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_public_release_contains_a_pretraining_manifest_validator():
    validator = ROOT / "scripts" / "validate_pretraining_manifest.py"
    assert validator.is_file(), "public retraining needs a manifest validation gate"


def _module():
    path = ROOT / "scripts" / "validate_pretraining_manifest.py"
    spec = importlib.util.spec_from_file_location("manifest_validator", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _entries(tmp_path, *, participants=32, per_modality=16):
    entries = []
    modalities = ("t1", "av45", "fdg", "ct")
    for modality_index, modality in enumerate(modalities):
        for index in range(per_modality):
            path = tmp_path / f"{modality}_{index}.npy"
            path.touch()
            entries.append({
                "path": str(path),
                "modality": modality,
                "subject_id": f"sub-{(modality_index * per_modality + index) % participants:03d}",
            })
    return entries


def test_validator_accepts_the_v4_sampling_requirements(tmp_path):
    config = yaml.safe_load((ROOT / "configs" / "omnimira_3atlas.yaml").read_text())
    summary = _module().validate_entries(_entries(tmp_path), config)
    assert summary == {
        "scans": 64,
        "modalities": {"t1": 16, "av45": 16, "fdg": 16, "ct": 16},
        "unique_participants": 32,
    }


def test_validator_rejects_manifest_that_cannot_form_a_v4_batch(tmp_path):
    config = yaml.safe_load((ROOT / "configs" / "omnimira_3atlas.yaml").read_text())
    entries = _entries(tmp_path, participants=1, per_modality=15)
    with pytest.raises(ValueError, match="t1 has 15 scans.*unique participants"):
        _module().validate_entries(entries, config)
