"""Stable frozen OmniMIRA loading, extraction, and serialization API."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from omnimira.atlas import ATLAS_ORDER, atlas_hashes, fixed_patch_to_roi, roi_metadata
from omnimira.schema import (
    MODEL_SHAPE,
    SUPPORTED_MODALITIES,
    file_sha256,
    input_contract,
    input_contract_sha256,
    modality_contract,
    resource_path,
)


SCHEMA_VERSION = "1.0"
MODEL_VERSION = "omnimira-public-v1"
FEATURE_DIM = 128
MODALITY_IDS = {name: index for index, name in enumerate(SUPPORTED_MODALITIES)}
SOURCE_METADATA_FIELDS = {
    "schema_version",
    "input_contract_version",
    "input_contract_sha256",
    "input_sha256",
    "template_sha256",
    "brain_mask_sha256",
    "space",
    "shape",
    "affine",
    "qform_code",
    "sform_code",
    "orientation",
    "modality",
    "normalization",
    "preprocessing_state",
}


class _RoiFeatureSet(dict):
    """Dict-compatible arrays carrying immutable extraction provenance."""

    def __init__(self, values, *, modality: str, checkpoint_sha256: str, model_version: str):
        super().__init__(values)
        self.modality = modality
        self.checkpoint_sha256 = checkpoint_sha256
        self.model_version = model_version


def _build_model(config, device: str):
    from omnimira.models.omnimira import HeadSpec, OmniMIRA

    config_path = (resource_path("configs/omnimira_3atlas.yaml") if config is None
                   else Path(config).expanduser())
    with config_path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    head_specs = None
    if cfg.get("heads") is not None:
        head_specs = [HeadSpec(**head) for head in cfg["heads"]]
    atlas_n_rois = {
        atlas["name"]: int(atlas["n_rois"]) for atlas in cfg.get("atlases", [])
    } or None
    pool = cfg["model"].get("pool", {})
    model = OmniMIRA(
        img_size=tuple(cfg["model"]["img_size"]),
        patch_size=cfg["model"]["patch_size"],
        embed_dim=cfg["model"]["embed_dim"],
        mlp_depth=cfg["model"]["mlp_depth"],
        mlp_ratio=cfg["model"]["mlp_ratio"],
        n_modalities=cfg["model"]["n_modalities"],
        n_rois=cfg["model"]["n_rois"],
        contrastive_dim=cfg["model"]["contrastive_dim"],
        mask_ratio=cfg["model"].get("mask_ratio", 0.5),
        decoder_depth=cfg["model"].get("decoder_depth", 4),
        decoder_heads=cfg["model"].get("decoder_heads", 4),
        head_specs=head_specs,
        atlas_n_rois=atlas_n_rois,
        pool_kind=pool.get("kind", "mean"),
        pool_k=pool.get("k", 4),
    )
    return model.to(device)


def _freeze(model):
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def _build_omnimira_for_tests(config, device: str = "cpu"):
    """Private checkpoint-free construction for architecture tests only."""
    return _freeze(_build_model(config, device))


def load_omnimira(checkpoint, config=None, device: str = "cpu"):
    """Strictly load and freeze an OmniMIRA checkpoint."""
    checkpoint_path = Path(checkpoint).expanduser()
    model = _build_model(config, device)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or payload.get("model_version") != MODEL_VERSION:
        raise ValueError(f"checkpoint model_version must be {MODEL_VERSION}")
    state = payload.get("model", payload.get("state_dict", payload))
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model._omnimira_checkpoint_sha256 = file_sha256(checkpoint_path)
    model._omnimira_model_version = MODEL_VERSION
    return _freeze(model)


def _model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _normalized_tensor(volume, device: torch.device) -> torch.Tensor:
    if isinstance(volume, torch.Tensor):
        value = volume.detach()
    else:
        value = torch.as_tensor(np.asarray(volume))
    if value.ndim == 3:
        value = value.unsqueeze(0).unsqueeze(0)
    elif value.ndim == 4:
        value = value.unsqueeze(1)
    if value.ndim != 5 or tuple(value.shape[1:]) != (1, *MODEL_SHAPE):
        raise ValueError(f"normalized_volume must have shape {MODEL_SHAPE} or (B,1,{MODEL_SHAPE})")
    value = value.to(device=device, dtype=torch.float32)
    if not torch.isfinite(value).all():
        raise ValueError("normalized_volume must contain only finite values")
    if torch.any(value < 0) or torch.any(value > 1):
        raise ValueError("normalized_volume values must be in [0,1]")
    return value


@torch.no_grad()
def extract_roi_features(model, normalized_volume, modality: str = "t1") -> dict:
    """Extract fixed-order AAL3, HO69, and Yeo7 float32 ROI features."""
    if modality not in MODALITY_IDS:
        raise ValueError(f"unsupported modality {modality!r}")
    device = _model_device(model)
    volume = _normalized_tensor(normalized_volume, device)
    if volume.shape[0] != 1:
        raise ValueError("extract_roi_features accepts one scan at a time")
    modality_ids = torch.full(
        (volume.shape[0],), MODALITY_IDS[modality], dtype=torch.long, device=device
    )
    mappings = {
        name: torch.from_numpy(mapping).to(device=device, dtype=torch.long)
        for name, mapping in fixed_patch_to_roi().items()
    }
    encoded = model.encode_roi_features(volume, modality_ids, mappings)
    arrays = {
        name: np.asarray(
            encoded["atlases"][name]["roi_tokens"][0].detach().cpu().numpy(),
            dtype=np.float32,
        )
        for name in ATLAS_ORDER
    }
    return _RoiFeatureSet(
        arrays,
        modality=modality,
        checkpoint_sha256=getattr(model, "_omnimira_checkpoint_sha256", "0" * 64),
        model_version=getattr(model, "_omnimira_model_version", MODEL_VERSION),
    )


def source_metadata(input_path, modality: str) -> dict[str, Any]:
    """Construct output provenance for an input already accepted by load_volume."""
    stream = modality_contract(modality)
    path = Path(input_path).expanduser()
    return {
        "schema_version": SCHEMA_VERSION,
        "input_contract_version": input_contract()["schema_version"],
        "input_contract_sha256": input_contract_sha256(),
        "input_sha256": file_sha256(path),
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
        "preprocessing_state": input_contract()["npy_sidecar"]["preprocessing_state"],
    }


def _require_type(metadata: dict[str, Any], key: str, expected_type) -> None:
    value = metadata[key]
    if isinstance(value, bool) or not isinstance(value, expected_type):
        raise ValueError(f"source_metadata {key} has the wrong type")


def validate_source_metadata(metadata: dict[str, Any], modality: str) -> None:
    """Validate exact types and the modality-specific frozen input stream."""
    if not isinstance(metadata, dict):
        raise ValueError("source_metadata must be an object")
    actual = set(metadata)
    if actual != SOURCE_METADATA_FIELDS:
        missing = sorted(SOURCE_METADATA_FIELDS - actual)
        extra = sorted(actual - SOURCE_METADATA_FIELDS)
        raise ValueError(f"source_metadata fields mismatch; missing={missing}, extra={extra}")
    for key in (
        "schema_version", "input_contract_version", "input_contract_sha256",
        "input_sha256", "template_sha256", "brain_mask_sha256", "space",
        "orientation", "modality", "normalization", "preprocessing_state",
    ):
        _require_type(metadata, key, str)
    for key in ("qform_code", "sform_code"):
        _require_type(metadata, key, int)
    for key in ("input_contract_sha256", "input_sha256", "template_sha256", "brain_mask_sha256"):
        value = metadata[key]
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"source_metadata {key} must be a lowercase SHA256")
    if not isinstance(metadata["shape"], list) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in metadata["shape"]
    ):
        raise ValueError("source_metadata shape must contain three integers")
    try:
        affine = np.asarray(metadata["affine"], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("source_metadata affine must be numeric") from exc
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError("source_metadata affine must be a finite 4x4 matrix")
    if modality not in MODALITY_IDS:
        raise ValueError(f"source_metadata modality is unsupported: {modality!r}")
    stream = modality_contract(modality)
    expected = {
        "schema_version": SCHEMA_VERSION,
        "input_contract_version": input_contract()["schema_version"],
        "input_contract_sha256": input_contract_sha256(),
        "orientation": stream.orientation,
        "space": stream.space,
        "template_sha256": stream.template_sha256,
        "brain_mask_sha256": stream.brain_mask_sha256,
        "shape": list(MODEL_SHAPE),
        "qform_code": stream.qform_code,
        "sform_code": stream.sform_code,
        "modality": modality,
        "normalization": stream.normalization,
        "preprocessing_state": input_contract()["npy_sidecar"]["preprocessing_state"],
    }
    for key, expected_value in expected.items():
        if metadata[key] != expected_value:
            raise ValueError(f"source_metadata {key} does not match the {modality} input stream")
    if not np.allclose(affine, stream.affine, atol=1e-5, rtol=0.0):
        raise ValueError(f"source_metadata affine does not match the {modality} input stream")


def save_roi_features(features, output_path, metadata) -> None:
    """Write the exact versioned, pickle-free OmniMIRA ROI NPZ contract."""
    if not isinstance(features, _RoiFeatureSet):
        raise ValueError("features must be returned by extract_roi_features")
    validate_source_metadata(metadata, features.modality)
    if metadata["modality"] != features.modality:
        raise ValueError("source_metadata modality does not match extracted features")
    expected_shapes = {"aal3": (166, 128), "ho69": (69, 128), "yeo7": (7, 128)}
    if list(features) != list(ATLAS_ORDER):
        raise ValueError("features must use fixed aal3, ho69, yeo7 order")
    for name, shape in expected_shapes.items():
        if features[name].shape != shape or features[name].dtype != np.float32:
            raise ValueError(f"{name} features must have shape {shape} and dtype float32")
    labels = roi_metadata()
    compact_json = {"sort_keys": True, "separators": (",", ":")}
    np.savez(
        Path(output_path),
        schema_version=np.str_(SCHEMA_VERSION),
        model_version=np.str_(features.model_version),
        checkpoint_sha256=np.str_(features.checkpoint_sha256),
        modality=np.str_(features.modality),
        feature_dim=np.int32(FEATURE_DIM),
        feature_dtype=np.str_("float32"),
        atlas_hashes_json=np.str_(json.dumps(atlas_hashes(), **compact_json)),
        source_metadata_json=np.str_(json.dumps(metadata, **compact_json)),
        aal3_features=features["aal3"],
        aal3_roi_ids=labels["aal3"].ids,
        aal3_roi_names=labels["aal3"].names,
        ho69_features=features["ho69"],
        ho69_roi_ids=labels["ho69"].ids,
        ho69_roi_names=labels["ho69"].names,
        yeo7_features=features["yeo7"],
        yeo7_roi_ids=labels["yeo7"].ids,
        yeo7_roi_names=labels["yeo7"].names,
    )
