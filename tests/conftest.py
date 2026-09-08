from __future__ import annotations

import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from omnimira.schema import (
    MODEL_SHAPE,
    input_contract_sha256,
    modality_contract,
    verified_resource,
)


ATLAS_REQUIRED_TEST_FILES = {
    "test_atlas.py",
    "test_dataset.py",
    "test_inference.py",
    "test_output_schema.py",
    "test_visualization.py",
}
ATLAS_REQUIRED_TEST_NAMES = {
    "test_preflight_does_not_read_normalized_numpy_payloads",
    "test_public_golden_generator_records_current_checkpoint_without_local_paths",
}


def _atlas_resources_available() -> bool:
    from omnimira.atlas import ATLAS_RESOURCES

    root = os.environ.get("OMNIMIRA_ATLAS_DIR")
    atlas_root = Path(root).expanduser() if root else Path(__file__).resolve().parents[1] / "assets" / "atlases"
    return all(
        (atlas_root / Path(binary).name).is_file()
        and (atlas_root / Path(labels).name).is_file()
        for binary, _, _, labels in ATLAS_RESOURCES.values()
    )


def pytest_collection_modifyitems(items):
    """Keep a clone without third-party atlases testable by default."""
    if _atlas_resources_available():
        return
    marker = pytest.mark.skip(
        reason="requires verified external atlases; set OMNIMIRA_ATLAS_DIR"
    )
    for item in items:
        if (
            item.path.name in ATLAS_REQUIRED_TEST_FILES
            or item.name in ATLAS_REQUIRED_TEST_NAMES
        ):
            item.add_marker(marker)


@pytest.fixture
def normalized_npy_factory():
    def write(directory: Path, modality: str, seed: int = 0) -> Path:
        stream = modality_contract(modality)
        mask = np.asarray(
            nib.load(verified_resource(stream.brain_mask_name)).dataobj,
            dtype=np.uint8,
        ).astype(bool)
        volume = np.zeros(MODEL_SHAPE, dtype=np.float32)
        volume[mask] = np.random.default_rng(seed).random(int(mask.sum()), dtype=np.float32)
        path = directory / f"{modality}_{seed}.npy"
        np.save(path, volume)
        metadata = {
            "schema_version": "2.0",
            "input_contract_sha256": input_contract_sha256(),
            "template_sha256": stream.template_sha256,
            "brain_mask_sha256": stream.brain_mask_sha256,
            "space": stream.space,
            "shape": list(MODEL_SHAPE),
            "affine": stream.affine.tolist(),
            "qform_code": stream.qform_code,
            "sform_code": stream.sform_code,
            "orientation": stream.orientation,
            "modality": modality,
            "normalization": stream.normalization,
            "preprocessing_state": "normalized_v2",
        }
        path.with_suffix(".npy.json").write_text(json.dumps(metadata), encoding="utf-8")
        return path

    return write
