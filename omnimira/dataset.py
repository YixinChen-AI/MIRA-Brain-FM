"""Readable manifest-driven dataset for OmniMIRA pretraining.

A manifest is a JSON list of entries. Required per entry: ``path`` (an
already-MNI .nii/.nii.gz or model-space .npy) and ``modality`` (one of
t1/av45/fdg/ct). Optional chronological age, ROI-volume and spatial-distance
targets drive the supervised objectives; absent targets are masked out.
"""
from __future__ import annotations

import json
import zlib
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from omnimira.io import load_volume
from omnimira.atlas import ATLAS_ORDER, fixed_patch_to_roi, roi_metadata

MODALITY_ID = {"t1": 0, "av45": 1, "fdg": 2, "ct": 3}


def build_atlas_patch_to_roi(atlas_files: Dict[str, str],
                             img_size=(96, 112, 96),
                             patch_size: int = 8) -> Dict[str, torch.Tensor]:
    """Build the patch->ROI lookup for each atlas once (shared across the loader)."""
    if tuple(img_size) != (96, 112, 96) or patch_size != 8:
        raise ValueError("public atlas mappings require the paper-defined model grid")
    unknown = set(atlas_files) - set(ATLAS_ORDER)
    if unknown:
        raise ValueError(f"unknown public atlases: {sorted(unknown)}")
    mappings = fixed_patch_to_roi()
    return {name: torch.from_numpy(mappings[name].copy()).bool() for name in atlas_files}


def _enc(s: Union[str, int, None], default: int = 0) -> int:
    if s is None:
        return default
    if isinstance(s, int):
        return s
    # crc32 (not Python hash()) so string IDs encode deterministically across runs.
    return zlib.crc32(str(s).encode()) % (2 ** 31)


class ManifestDataset(Dataset):
    def __init__(self, manifest: Union[str, List[dict]],
                 atlas_files: Dict[str, str],
                 img_size=(96, 112, 96)):
        if isinstance(manifest, (str, Path)):
            entries = json.loads(Path(manifest).read_text())
        else:
            entries = list(manifest)
        self.entries = entries
        self.img_size = tuple(img_size)
        self.atlas_files = atlas_files

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int) -> dict:
        e = self.entries[i]
        modality = e["modality"]
        vol = load_volume(e["path"], modality=modality)
        x = torch.from_numpy(vol).float().unsqueeze(0)        # (1, D, H, W)
        has_age = "age" in e and e["age"] is not None
        has_sex = "sex" in e and e["sex"] is not None
        return {
            "x":               x,
            "modality_id":     MODALITY_ID[modality],
            "subject_id":      _enc(e.get("subject_id"), default=i),
            "time_id":         _enc(e.get("time_id"), default=0),
            "age":             float(e.get("age", 0.0) or 0.0),
            "age_valid":       bool(has_age),
            "sex":             int(e.get("sex", 0) or 0),
            "sex_valid":       bool(has_sex),
            "cohort_id":       _enc(e.get("cohort"), default=0),
            "roi_volumes":     _target_dict(e.get("roi_volumes"), vector=True),
            "roi_distances":   _target_dict(e.get("roi_distances"), vector=False),
        }


def _load_target(value):
    if isinstance(value, (str, Path)):
        return np.load(Path(value), allow_pickle=False)
    return np.asarray(value, dtype=np.float32)


def _target_dict(values, *, vector: bool) -> dict[str, torch.Tensor]:
    values = values or {}
    output = {}
    for name, metadata in roi_metadata().items():
        count = len(metadata.ids)
        shape = (count,) if vector else (count, count)
        if name not in values:
            array = np.full(shape, np.nan, dtype=np.float32)
        else:
            array = np.asarray(_load_target(values[name]), dtype=np.float32)
            if array.shape != shape:
                raise ValueError(f"{name} target must have shape {shape}, got {array.shape}")
        output[name] = torch.from_numpy(array)
    return output


def collate(batch: List[dict]) -> dict:
    out = {"x": torch.stack([b["x"] for b in batch], dim=0)}
    out["modality_id"] = torch.tensor([b["modality_id"] for b in batch], dtype=torch.long)
    out["subject_id"] = torch.tensor([b["subject_id"] for b in batch], dtype=torch.long)
    out["time_id"] = torch.tensor([b["time_id"] for b in batch], dtype=torch.long)
    out["age"] = torch.tensor([b["age"] for b in batch], dtype=torch.float32)
    out["age_valid"] = torch.tensor([b["age_valid"] for b in batch], dtype=torch.bool)
    out["sex"] = torch.tensor([b["sex"] for b in batch], dtype=torch.long)
    out["sex_valid"] = torch.tensor([b["sex_valid"] for b in batch], dtype=torch.bool)
    out["cohort_id"] = torch.tensor([b["cohort_id"] for b in batch], dtype=torch.long)
    out["roi_volumes"] = {
        name: torch.stack([b["roi_volumes"][name] for b in batch]) for name in ATLAS_ORDER
    }
    out["roi_volume_valid"] = {
        name: torch.isfinite(values).all(dim=1)
        for name, values in out["roi_volumes"].items()
    }
    out["roi_distances"] = {
        name: torch.stack([b["roi_distances"][name] for b in batch]) for name in ATLAS_ORDER
    }
    out["spatial_valid"] = {
        name: torch.isfinite(values).all(dim=(1, 2))
        for name, values in out["roi_distances"].items()
    }
    return out
