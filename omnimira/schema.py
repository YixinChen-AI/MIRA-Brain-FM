"""Immutable metadata helpers for the OmniMIRA input contract."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ERROR_PREFIX = "Input contract violation:"
MODEL_SHAPE = (96, 112, 96)
SUPPORTED_MODALITIES = ("t1", "av45", "fdg", "ct")


class InputContractError(ValueError):
    """Raised when an input does not satisfy the frozen release contract."""

    def __init__(self, message: str):
        super().__init__(f"{ERROR_PREFIX} {message}")


@dataclass(frozen=True)
class ModalityContract:
    name: str
    orientation: str
    space: str
    normalization: str
    template_name: str
    brain_mask_name: str
    template_sha256: str
    brain_mask_sha256: str
    affine: np.ndarray
    qform_code: int
    sform_code: int


def _source_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resource_path(relative_path: str) -> Path:
    """Resolve release data from a source tree or an installed wheel."""
    source = _source_root() / relative_path
    if source.is_file():
        return source

    try:
        distribution = metadata.distribution("omnimira")
    except metadata.PackageNotFoundError as exc:
        raise InputContractError(f"packaged resource is missing: {relative_path}") from exc

    suffix = f"omnimira_release/{relative_path}"
    for entry in distribution.files or ():
        normalized = str(entry).replace("\\", "/")
        if normalized.endswith(suffix):
            candidate = Path(distribution.locate_file(entry))
            if candidate.is_file():
                return candidate
    distribution_root = Path(distribution.locate_file("")).resolve()
    for parent in (distribution_root, *list(distribution_root.parents)[:4]):
        candidate = parent / suffix
        if candidate.is_file():
            return candidate
    raise InputContractError(f"packaged resource is missing: {relative_path}")


@lru_cache(maxsize=1)
def input_contract_path() -> Path:
    return resource_path("configs/input_contract.yaml")


@lru_cache(maxsize=1)
def input_contract() -> dict[str, Any]:
    data = yaml.safe_load(input_contract_path().read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != "2.0":
        raise InputContractError("packaged input contract has an unsupported schema")
    return data


@lru_cache(maxsize=1)
def input_contract_sha256() -> str:
    return hashlib.sha256(input_contract_path().read_bytes()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=None)
def verified_resource(name: str) -> Path:
    record = input_contract().get("resources", {}).get(name)
    if not isinstance(record, dict):
        raise InputContractError(f"unknown contract resource: {name}")
    path = resource_path(record["path"])
    if file_sha256(path) != record["sha256"]:
        raise InputContractError(f"packaged resource hash mismatch: {name}")
    return path


@lru_cache(maxsize=None)
def modality_contract(modality: str) -> ModalityContract:
    if modality not in SUPPORTED_MODALITIES:
        supported = ", ".join(SUPPORTED_MODALITIES)
        raise InputContractError(f"unsupported modality {modality!r}; expected one of {supported}")
    contract = input_contract()
    stream = contract["modalities"][modality]
    template = contract["resources"][stream["template"]]
    mask = contract["resources"][stream["brain_mask"]]
    return ModalityContract(
        name=modality,
        orientation=stream["orientation"],
        space=stream["space"],
        normalization=stream["normalization"],
        template_name=stream["template"],
        brain_mask_name=stream["brain_mask"],
        template_sha256=template["sha256"],
        brain_mask_sha256=mask["sha256"],
        affine=np.asarray(template["affine"], dtype=np.float64),
        qform_code=int(template["qform_code"]),
        sform_code=int(template["sform_code"]),
    )
