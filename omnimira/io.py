"""Strict loading and normalization for released OmniMIRA inputs."""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

from omnimira.schema import (
    InputContractError,
    MODEL_SHAPE,
    SUPPORTED_MODALITIES,
    input_contract,
    input_contract_sha256,
    modality_contract,
    verified_resource,
)


def _fail(message: str) -> None:
    raise InputContractError(message)


@lru_cache(maxsize=None)
def _brain_mask(modality: str) -> np.ndarray:
    stream = modality_contract(modality)
    image = nib.load(verified_resource(stream.brain_mask_name))
    mask = np.asarray(image.dataobj, dtype=np.uint8)
    if mask.shape != MODEL_SHAPE or not np.isin(mask, (0, 1)).all() or not mask.any():
        _fail(f"packaged brain mask is invalid for modality {modality!r}")
    return mask.astype(bool)


@lru_cache(maxsize=1)
def _pet_reference_mask() -> np.ndarray:
    contract = input_contract()["pet_reference"]
    resource_name = contract["resource"]
    image = nib.load(verified_resource(resource_name))
    reference = np.asarray(image.dataobj, dtype=np.uint8)
    if reference.shape != MODEL_SHAPE or not np.isin(reference, (0, 1)).all():
        _fail("packaged PET reference mask is not a binary model-space mask")
    reference = reference.astype(bool)
    expected_voxels = int(input_contract()["resources"][resource_name]["foreground_voxels"])
    if int(reference.sum()) != expected_voxels:
        _fail("packaged PET reference mask voxel count does not match the contract")
    if int(reference.sum()) < int(contract["minimum_voxels"]):
        _fail("packaged PET reference region contains fewer than 100 voxels")
    return reference


def _validate_nifti_geometry(image: nib.spatialimages.SpatialImage, modality: str) -> None:
    affine = np.asarray(image.affine, dtype=np.float64)
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        _fail(f"affine for {modality} must be a finite 4x4 matrix")
    try:
        axis_codes = nib.aff2axcodes(affine)
    except (TypeError, ValueError, np.linalg.LinAlgError) as exc:
        _fail(f"affine orientation for {modality} is invalid: {exc}")
    if any(code is None for code in axis_codes):
        _fail(f"affine orientation for {modality} is singular or indeterminate")
    if len(image.shape) != 3:
        _fail(f"{modality} input must be a three-dimensional NIfTI volume")


def _resample_to_model_grid(image: nib.spatialimages.SpatialImage) -> nib.Nifti1Image:
    """Resample a spatially standardized scan in world coordinates."""
    template = nib.load(verified_resource("model_grid"))
    target = (MODEL_SHAPE, np.asarray(template.affine, dtype=np.float64))
    if tuple(image.shape) == MODEL_SHAPE and np.allclose(
        image.affine, target[1], atol=1e-5, rtol=0.0
    ):
        return nib.Nifti1Image(np.asarray(image.dataobj, dtype=np.float32), target[1])
    return resample_from_to(image, target, order=1, mode="constant", cval=0.0)


def _normalize_t1(volume: np.ndarray, mask: np.ndarray) -> np.ndarray:
    constants = input_contract()["normalization"]["t1"]
    selected = volume[mask & np.isfinite(volume) & (volume > 0)]
    minimum = int(constants["minimum_voxels"])
    if selected.size < minimum:
        _fail(f"T1 requires at least {minimum} finite positive brain-mask voxels")
    p1, p99 = np.percentile(selected, constants["percentiles"])
    if not np.isfinite((p1, p99)).all() or p99 - p1 < float(constants["minimum_range"]):
        _fail("T1 p99-p1 must be finite and at least 1e-8")
    output = np.clip(volume, p1, p99)
    output = (output - p1) / (p99 - p1)
    output[~mask] = 0.0
    return output


def _normalize_pet(volume: np.ndarray, mask: np.ndarray) -> np.ndarray:
    reference = _pet_reference_mask()
    minimum = int(input_contract()["pet_reference"]["minimum_voxels"])
    if int(reference.sum()) < minimum:
        _fail(f"PET reference region requires at least {minimum} voxels")
    values = volume[reference]
    if values.size < minimum:
        _fail(f"PET reference region requires at least {minimum} voxels")
    if not np.isfinite(values).all():
        _fail("PET reference region contains non-finite voxels")
    reference_mean = float(np.mean(values, dtype=np.float64))
    constants = input_contract()["normalization"]["pet"]
    if not np.isfinite(reference_mean) or reference_mean <= float(constants["minimum_mean"]):
        _fail("PET reference mean must be finite and greater than 1e-8")
    output = np.clip(volume / reference_mean, *constants["clip"])
    output /= float(constants["divisor"])
    output[~mask] = 0.0
    return output


def _normalize_ct(volume: np.ndarray, mask: np.ndarray) -> np.ndarray:
    constants = input_contract()["normalization"]["ct"]
    output = np.clip(volume, *constants["clip_hu"])
    output /= float(constants["divisor"])
    output[~mask] = 0.0
    return output


def _load_raw_nifti(path: Path, modality: str) -> np.ndarray:
    try:
        image = nib.load(path)
    except Exception as exc:
        _fail(f"could not read NIfTI input {path}: {exc}")
    _validate_nifti_geometry(image, modality)
    image = _resample_to_model_grid(image)
    volume = np.asarray(image.dataobj, dtype=np.float32)
    mask = _brain_mask(modality)
    if not np.isfinite(volume[mask]).all():
        _fail("non-finite voxel found inside the committed brain mask")
    if modality in ("av45", "fdg") and not np.isfinite(volume[_pet_reference_mask()]).all():
        _fail("PET reference region contains non-finite voxels")
    volume = volume.copy()
    volume[~mask & ~np.isfinite(volume)] = 0.0

    if modality == "t1":
        output = _normalize_t1(volume, mask)
    elif modality in ("av45", "fdg"):
        output = _normalize_pet(volume, mask)
    else:
        output = _normalize_ct(volume, mask)
    return np.asarray(output, dtype=np.float32)


def _read_sidecar(path: Path) -> dict[str, Any]:
    sidecar_path = path.with_suffix(".npy.json")
    if not sidecar_path.is_file():
        _fail(f"normalized NumPy input requires JSON sidecar {sidecar_path.name}")
    try:
        metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(f"NumPy sidecar is not valid JSON: {exc}")
    if not isinstance(metadata, dict):
        _fail("NumPy sidecar must be a JSON object")
    return metadata


def _validate_sidecar(metadata: dict[str, Any], modality: str) -> None:
    contract = input_contract()
    stream = modality_contract(modality)
    required = set(contract["npy_sidecar"]["required_fields"])
    actual = set(metadata)
    if actual != required:
        missing = sorted(required - actual)
        extra = sorted(actual - required)
        _fail(f"NumPy sidecar fields mismatch; missing={missing}, extra={extra}")

    expected: dict[str, Any] = {
        "schema_version": "2.0",
        "input_contract_sha256": input_contract_sha256(),
        "template_sha256": stream.template_sha256,
        "brain_mask_sha256": stream.brain_mask_sha256,
        "space": stream.space,
        "shape": list(MODEL_SHAPE),
        "qform_code": stream.qform_code,
        "sform_code": stream.sform_code,
        "orientation": stream.orientation,
        "modality": modality,
        "normalization": stream.normalization,
        "preprocessing_state": "normalized_v2",
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            if key == "preprocessing_state":
                _fail("NumPy sidecar preprocessing_state must be normalized_v2; raw NumPy is rejected")
            _fail(f"NumPy sidecar {key!r} does not match the {modality} contract")

    affine = metadata.get("affine")
    try:
        affine_array = np.asarray(affine, dtype=np.float64)
    except (TypeError, ValueError):
        _fail("NumPy sidecar affine must be a numeric 4x4 matrix")
    if affine_array.shape != (4, 4) or not np.isfinite(affine_array).all():
        _fail("NumPy sidecar affine must be a finite numeric 4x4 matrix")
    if not np.allclose(affine_array, stream.affine, atol=1e-5, rtol=0.0):
        _fail(f"NumPy sidecar affine does not match the {modality} template")


def _load_normalized_numpy(path: Path, modality: str) -> np.ndarray:
    metadata = _read_sidecar(path)
    _validate_sidecar(metadata, modality)
    try:
        volume = np.load(path, mmap_mode="r", allow_pickle=False)
    except Exception as exc:
        _fail(f"could not read NumPy input {path}: {exc}")
    if volume.dtype != np.float32:
        _fail(f"normalized NumPy dtype must be float32, got {volume.dtype}")
    if volume.shape != MODEL_SHAPE:
        _fail(f"normalized NumPy shape must be {MODEL_SHAPE}, got {volume.shape}")
    if not np.isfinite(volume).all():
        _fail("normalized NumPy input contains non-finite values")
    if np.any(volume < 0.0) or np.any(volume > 1.0):
        _fail("normalized NumPy values must be in [0,1]")
    mask = _brain_mask(modality)
    if np.any(volume[~mask] != 0.0):
        _fail("normalized NumPy values outside the committed brain mask must be zero")
    return np.array(volume, dtype=np.float32, copy=True)


def load_volume(path: str | Path, modality: str) -> np.ndarray:
    """Resample and normalize one MNI-space NIfTI or load normalized-v2 NumPy."""
    stream = modality_contract(modality)
    input_path = Path(path).expanduser()
    if not input_path.is_file():
        _fail(f"input file does not exist: {input_path}")
    verified_resource(stream.template_name)
    if input_path.suffix == ".npy":
        return _load_normalized_numpy(input_path, modality)
    if input_path.name.endswith((".nii", ".nii.gz")):
        return _load_raw_nifti(input_path, modality)
    _fail("input must be .nii, .nii.gz, or normalized_v2 .npy")
