from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from omnimira.io import load_volume
from omnimira.schema import MODEL_SHAPE, modality_contract


ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts" / "prepare_pretraining_cache.py"
    spec = importlib.util.spec_from_file_location("cache_builder", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_cache_builder_writes_v4_normalized_numpy_and_sidecar(tmp_path):
    stream = modality_contract("t1")
    source = tmp_path / "source.nii.gz"
    data = np.linspace(1, 100, np.prod(MODEL_SHAPE), dtype=np.float32).reshape(MODEL_SHAPE)
    image = nib.Nifti1Image(data, stream.affine)
    image.set_qform(stream.affine, code=1)
    image.set_sform(stream.affine, code=1)
    nib.save(image, source)

    output = _module().prepare_entries([
        {"path": str(source), "subject_id": "sub-01", "modality": "t1", "cohort": "toy"}
    ], tmp_path / "cache")

    assert len(output) == 1
    record = output[0]
    assert record["subject_id"] == "sub-01"
    assert record["cohort"] == "toy"
    cached = Path(record["path"])
    assert cached.is_file()
    assert cached.with_suffix(".npy.json").is_file()
    assert load_volume(cached, "t1").shape == MODEL_SHAPE


def test_cache_builder_rejects_numpy_sources_without_spatial_metadata(tmp_path):
    source = tmp_path / "legacy.npy"
    np.save(source, np.zeros(MODEL_SHAPE, dtype=np.float32))
    with pytest.raises(ValueError, match="NIfTI source"):
        _module().prepare_entries([
            {"path": str(source), "subject_id": "sub-01", "modality": "t1"}
        ], tmp_path / "cache")
