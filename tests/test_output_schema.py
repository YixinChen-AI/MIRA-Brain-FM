from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import omnimira.atlas as atlas
import omnimira.inference as inference
from omnimira.schema import MODEL_SHAPE, input_contract, input_contract_sha256, modality_contract


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_KEYS = {
    "schema_version",
    "model_version",
    "checkpoint_sha256",
    "modality",
    "feature_dim",
    "feature_dtype",
    "atlas_hashes_json",
    "source_metadata_json",
    "aal3_features",
    "aal3_roi_ids",
    "aal3_roi_names",
    "ho69_features",
    "ho69_roi_ids",
    "ho69_roi_names",
    "yeo7_features",
    "yeo7_roi_ids",
    "yeo7_roi_names",
}


def _metadata(modality: str = "t1") -> dict:
    stream = modality_contract(modality)
    return {
        "schema_version": "1.0",
        "input_contract_version": input_contract()["schema_version"],
        "input_contract_sha256": input_contract_sha256(),
        "input_sha256": "a" * 64,
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


def _features(modality: str = "t1"):
    assert hasattr(inference, "_RoiFeatureSet")
    return inference._RoiFeatureSet(
        {
            "aal3": np.zeros((166, 128), dtype=np.float32),
            "ho69": np.zeros((69, 128), dtype=np.float32),
            "yeo7": np.zeros((7, 128), dtype=np.float32),
        },
        modality=modality,
        checkpoint_sha256="b" * 64,
        model_version="omnimira-public-v1",
    )


def test_atlas_metadata_preserves_dense_valid_label_order_and_names() -> None:
    metadata = atlas.roi_metadata()
    assert list(metadata) == ["aal3", "ho69", "yeo7"]
    assert len(metadata["aal3"].ids) == 166
    assert not {35, 36, 81, 82} & set(metadata["aal3"].ids.tolist())
    assert {167, 168, 169, 170} <= set(metadata["aal3"].ids.tolist())
    np.testing.assert_array_equal(metadata["ho69"].ids, np.arange(1, 70, dtype=np.int32))
    np.testing.assert_array_equal(metadata["yeo7"].ids, np.arange(1, 8, dtype=np.int32))
    assert len(set(metadata["ho69"].names.tolist())) == 69
    assert len(set(metadata["yeo7"].names.tolist())) == 7


def test_packaged_atlas_hashes_are_exact() -> None:
    assert atlas.atlas_hashes() == {
        "aal3": "aa44bb1767594f560e811cd917fee5833c2f843b81a8febd614db20284b11a34",
        "ho69": "9ad6acc55d44c988bf30e997737d20aa6b94453624f4669bdad39f3a7a104e2e",
        "yeo7": "5b13f9eebcfdb103455a4a424f762f936a2df1c4c185ae3726f4eff68e3cf1a7",
    }


def test_save_roi_features_writes_exact_pickle_free_contract(tmp_path: Path) -> None:
    output = tmp_path / "features.npz"
    inference.save_roi_features(_features(), output, _metadata())

    with np.load(output, allow_pickle=False) as value:
        assert set(value.files) == EXPECTED_KEYS
        assert value["schema_version"].item() == "1.0"
        assert value["model_version"].item() == "omnimira-public-v1"
        assert value["checkpoint_sha256"].item() == "b" * 64
        assert value["modality"].item() == "t1"
        assert value["feature_dim"].item() == 128
        assert value["feature_dtype"].item() == "float32"
        assert value["aal3_features"].dtype == np.float32
        assert value["aal3_roi_ids"].dtype == np.int32
        assert value["aal3_roi_names"].dtype.kind == "U"
        assert json.loads(value["atlas_hashes_json"].item()) == atlas.atlas_hashes()
        assert value["source_metadata_json"].item() == json.dumps(
            _metadata(), sort_keys=True, separators=(",", ":")
        )


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda value: value.pop("input_sha256"), "input_sha256"),
        (lambda value: value.update(qform_code="1"), "qform_code"),
        (lambda value: value.update(modality="pet"), "modality"),
        (lambda value: value.update(preprocessing_state="raw"), "preprocessing_state"),
    ],
)
def test_output_metadata_rejects_missing_wrong_type_and_unknown_enum(
    tmp_path: Path, mutate, match: str
) -> None:
    metadata = _metadata()
    mutate(metadata)
    with pytest.raises(ValueError, match=match):
        inference.save_roi_features(_features(), tmp_path / "bad.npz", metadata)


def test_output_metadata_rejects_self_consistent_wrong_stream(tmp_path: Path) -> None:
    wrong = _metadata("fdg")
    wrong["modality"] = "t1"
    with pytest.raises(ValueError, match="normalization"):
        inference.save_roi_features(_features("t1"), tmp_path / "wrong.npz", wrong)


def test_output_schema_and_label_sources_are_packaged_and_traceable() -> None:
    schema = json.loads((ROOT / "configs" / "output_schema.json").read_text(encoding="utf-8"))
    assert schema["$id"] == "https://omnimira.org/schemas/roi-features-1.0.json"
    sources = json.loads(
        (ROOT / "assets" / "atlases" / "atlas_label_sources.json").read_text(encoding="utf-8")
    )
    assert sources["aal3"]["ordering"] == (
        "occupied AAL3v1 labels in ascending raw-label order; labels 35, 36, 81, "
        "and 82 are absent, while labels 167 through 170 are retained"
    )
    assert sources["ho69"]["ordering"] == "48 cortical labels followed by 21 subcortical labels"
    assert all("path_at_derivation" not in source for source in sources["ho69"]["sources"])
    assert sources["ho69"]["sources"][0]["sha256"] == "b8eb5ba5e787a4d3b442f75041f533f31b278a83e467bd2ab22438c1cff36581"
    assert "path_at_derivation" not in sources["yeo7"]["sources"][0]
    assert sources["yeo7"]["sources"][0]["sha256"] == "f6e68af76dffac569c4e6b5d7cfe13036e5ce0a702192a3774ddae48af985e26"


def test_input_sha256_metadata_helper_hashes_exact_input(tmp_path: Path) -> None:
    source = tmp_path / "scan.npy"
    source.write_bytes(b"exact input bytes")
    metadata = inference.source_metadata(source, "ct")
    assert metadata["input_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    inference.validate_source_metadata(metadata, "ct")
