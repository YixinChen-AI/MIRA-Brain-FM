"""Audit the standalone public OmniMIRA release contract."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from urllib.parse import urlparse

import yaml
import numpy as np


PREFIX = "public release violation:"
REQUIRED_GATES = (
    "public_checkpoint_training",
    "checkpoint_redistribution",
    "atlas_redistribution",
    "self_contained_result_evidence",
    "committed_manuscript_revision",
)
MODALITIES = ("t1", "av45", "fdg", "ct")
ATLASES = ("aal3", "ho69", "yeo7")


def _load_mapping(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a mapping: {path}")
    return value


def _load_manifest(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("checkpoints"), list):
        raise ValueError(f"invalid checkpoint manifest: {path}")
    if len(value["checkpoints"]) != 1 or not isinstance(value["checkpoints"][0], dict):
        raise ValueError("public checkpoint manifest must contain one checkpoint record")
    return value["checkpoints"][0]


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _golden_evidence_errors(manifest_path: Path, arrays_path: Path, checkpoint: dict) -> list[str]:
    errors: list[str] = []
    try:
        record = json.loads(manifest_path.read_text(encoding="utf-8"))
        with np.load(arrays_path, allow_pickle=False) as arrays:
            expected = {f"{modality}__{atlas}" for modality in MODALITIES for atlas in ATLASES}
            if set(arrays.files) != expected:
                errors.append("golden array keys do not match the four-modality three-atlas contract")
            for key in sorted(expected & set(arrays.files)):
                metadata = (record.get("arrays") or {}).get(key, {})
                value = arrays[key]
                if list(value.shape) != metadata.get("shape") or str(value.dtype) != metadata.get("dtype"):
                    errors.append(f"golden array schema mismatch: {key}")
                if _array_sha256(value) != metadata.get("sha256"):
                    errors.append(f"golden array hash mismatch: {key}")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [f"golden evidence cannot be read: {exc}"]
    if record.get("model_version") != "omnimira-public-v1":
        errors.append("golden evidence has an unexpected model version")
    if (record.get("checkpoint") or {}).get("sha256") != checkpoint.get("sha256"):
        errors.append("golden evidence checkpoint does not match the release manifest")
    output = record.get("output") or {}
    if output.get("filename") != arrays_path.name:
        errors.append("golden evidence output filename does not match the configured array file")
    if output.get("sha256") != hashlib.sha256(arrays_path.read_bytes()).hexdigest():
        errors.append("golden evidence output hash mismatch")
    if output.get("bytes") != arrays_path.stat().st_size:
        errors.append("golden evidence output byte count mismatch")
    return errors


def _metadata_errors(root: Path, contract: dict, checkpoint: dict) -> list[str]:
    errors: list[str] = []
    if contract.get("status") not in {"public_retraining_pending", "published"}:
        errors.append("unsupported public release status")
    for key in ("model_config", "manuscript_contract", "release_provenance"):
        path = root / str(contract.get(key, ""))
        if not path.is_file():
            errors.append(f"missing contract resource: {key}")
    if tuple(contract.get("publication_gates") or ()) != REQUIRED_GATES and contract.get("status") == "public_retraining_pending":
        errors.append("pending release must declare all publication gates")
    if checkpoint.get("version") != "omnimira-public-v1":
        errors.append("unexpected public checkpoint version")
    return errors


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publication", action="store_true")
    parser.add_argument(
        "--contract", type=Path, default=root / "configs" / "public_release_contract.yaml"
    )
    args = parser.parse_args()

    try:
        contract = _load_mapping(args.contract)
        manifest_path = root / str(contract["checkpoint_manifest"])
        checkpoint = _load_manifest(manifest_path)
    except (OSError, KeyError, ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
        print(f"{PREFIX} metadata load failed: {exc}", file=sys.stderr)
        return 1

    errors = _metadata_errors(root, contract, checkpoint)
    if args.publication:
        if contract.get("status") != "published":
            errors.append("public release status is not published")
        if checkpoint.get("state") != "published":
            errors.append(f"checkpoint state {checkpoint.get('state')!r}; a published checkpoint is required")
        if urlparse(str(checkpoint.get("url") or "")).scheme != "https":
            errors.append("publication checkpoint URL must use HTTPS")
        license_text = str(checkpoint.get("license") or "").lower()
        if not license_text or "assigned" in license_text or "unresolved" in license_text:
            errors.append("checkpoint redistribution license is not publication-ready")
        for gate in contract.get("publication_gates") or ():
            errors.append(f"publication gate remains open: {gate}")
        for path in (root / value for value in (contract.get("required_result_evidence") or {}).values()):
            if not path.is_file():
                errors.append(f"required result evidence is missing: {path.name}")
        if not contract.get("publication_gates"):
            evidence = contract.get("required_result_evidence") or {}
            errors.extend(
                _golden_evidence_errors(
                    root / str(evidence.get("manifest", "")),
                    root / str(evidence.get("arrays", "")),
                    checkpoint,
                )
            )

    if errors:
        for error in errors:
            print(f"{PREFIX} {error}", file=sys.stderr)
        return 1
    if args.publication:
        print("public release audit passed")
    else:
        print("public release metadata-only audit passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
