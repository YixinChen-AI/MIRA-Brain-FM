from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from omnimira.schema import modality_contract, input_contract, input_contract_sha256


ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts" / "audit_pretraining_inputs.py"
    spec = importlib.util.spec_from_file_location("input_audit", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sidecar(modality: str) -> dict:
    stream = modality_contract(modality)
    return {
        "schema_version": "2.0",
        "input_contract_sha256": input_contract_sha256(),
        "template_sha256": stream.template_sha256,
        "brain_mask_sha256": stream.brain_mask_sha256,
        "space": stream.space,
        "shape": [96, 112, 96],
        "affine": stream.affine.tolist(),
        "qform_code": stream.qform_code,
        "sform_code": stream.sform_code,
        "orientation": stream.orientation,
        "modality": modality,
        "normalization": stream.normalization,
        "preprocessing_state": "normalized_v2",
    }


def test_input_audit_reports_aggregate_sidecar_contract_failures(tmp_path):
    valid = tmp_path / "valid.npy"
    valid.touch()
    valid.with_suffix(".npy.json").write_text(json.dumps(_sidecar("t1")))
    missing = tmp_path / "missing.npy"
    missing.touch()
    wrong = tmp_path / "wrong.npy"
    wrong.touch()
    wrong.with_suffix(".npy.json").write_text(json.dumps({"modality": "t1"}))

    report = _module().audit_entries([
        {"path": str(valid), "modality": "t1"},
        {"path": str(missing), "modality": "t1"},
        {"path": str(wrong), "modality": "t1"},
        {"path": str(tmp_path / "raw.nii.gz"), "modality": "fdg"},
    ])

    assert report == {
        "entries_checked": 4,
        "nifti_entries": 1,
        "numpy_entries": 3,
        "missing_sidecar": 1,
        "invalid_sidecar": 1,
        "unsupported_path": 0,
    }
