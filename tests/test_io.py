from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from omnimira.io import load_volume
from omnimira.schema import (
    InputContractError,
    MODEL_SHAPE,
    file_sha256,
    input_contract,
    modality_contract,
    verified_resource,
)


def _save_nifti(path: Path, data: np.ndarray, affine: np.ndarray) -> Path:
    image = nib.Nifti1Image(data.astype(np.float32), affine)
    image.set_qform(affine, code=1)
    image.set_sform(affine, code=1)
    nib.save(image, path)
    return path


def test_all_modalities_share_one_ras_model_grid():
    streams = [modality_contract(name) for name in ("t1", "av45", "fdg", "ct")]
    assert {stream.orientation for stream in streams} == {"RAS"}
    assert len({stream.space for stream in streams}) == 1
    assert all(np.array_equal(stream.affine, streams[0].affine) for stream in streams)


def test_world_coordinate_resampling_accepts_different_source_grid(tmp_path: Path):
    shape = (48, 56, 48)
    affine = np.array([
        [-3.7894736842, 0, 0, 90],
        [0, 3.8918918919, 0, -126],
        [0, 0, 3.7894736842, -72],
        [0, 0, 0, 1],
    ])
    grid = np.indices(shape, dtype=np.float32)
    data = 1 + grid[0] + 0.5 * grid[1] + 0.25 * grid[2]
    path = _save_nifti(tmp_path / "t1_las.nii.gz", data, affine)
    volume = load_volume(path, "t1")
    assert volume.shape == MODEL_SHAPE
    assert volume.dtype == np.float32
    assert np.isfinite(volume).all()
    assert 0 <= volume.min() <= volume.max() <= 1


def test_exact_grid_ct_normalization(tmp_path: Path):
    stream = modality_contract("ct")
    data = np.full(MODEL_SHAPE, 40.0, dtype=np.float32)
    path = _save_nifti(tmp_path / "ct.nii.gz", data, stream.affine)
    output = load_volume(path, "ct")
    mask = np.asarray(nib.load(verified_resource("model_brain_mask")).dataobj).astype(bool)
    np.testing.assert_allclose(output[mask], 0.5)
    assert np.all(output[~mask] == 0)


def test_normalized_numpy_requires_v2_sidecar(tmp_path: Path, normalized_npy_factory):
    path = normalized_npy_factory(tmp_path, "t1")
    assert load_volume(path, "t1").shape == MODEL_SHAPE
    metadata = json.loads(path.with_suffix(".npy.json").read_text())
    metadata["preprocessing_state"] = "normalized_v1"
    path.with_suffix(".npy.json").write_text(json.dumps(metadata))
    with pytest.raises(InputContractError, match="normalized_v2"):
        load_volume(path, "t1")


def test_non_three_dimensional_source_is_rejected(tmp_path: Path):
    path = tmp_path / "bad.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((8, 8, 8, 2), dtype=np.float32), np.eye(4)), path)
    with pytest.raises(InputContractError, match="three-dimensional"):
        load_volume(path, "t1")


def test_contract_resources_match_recorded_hashes():
    for name, record in input_contract()["resources"].items():
        assert file_sha256(verified_resource(name)) == record["sha256"]
