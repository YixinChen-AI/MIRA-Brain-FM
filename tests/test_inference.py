from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

import omnimira.inference as inference


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "omnimira_3atlas.yaml"
MODALITIES = ("t1", "av45", "fdg", "ct")


@pytest.fixture(scope="module")
def frozen_test_model(tmp_path_factory):
    assert hasattr(inference, "_build_omnimira_for_tests")
    source = inference._build_omnimira_for_tests(CONFIG, device="cpu")
    checkpoint = tmp_path_factory.mktemp("checkpoint") / "model.pt"
    torch.save({"model_version": inference.MODEL_VERSION, "model": source.state_dict()}, checkpoint)
    return inference.load_omnimira(checkpoint, CONFIG, device="cpu")


def test_public_loader_signature_has_no_non_strict_escape_hatch() -> None:
    signature = inspect.signature(inference.load_omnimira)
    assert list(signature.parameters) == ["checkpoint", "config", "device"]
    assert signature.parameters["device"].default == "cpu"
    assert signature.parameters["config"].default is None


def test_public_loader_rejects_incomplete_checkpoint(tmp_path: Path) -> None:
    assert hasattr(inference, "_build_omnimira_for_tests")
    model = inference._build_omnimira_for_tests(CONFIG, device="cpu")
    state = model.state_dict()
    state.pop(next(iter(state)))
    checkpoint = tmp_path / "incomplete.pt"
    torch.save({"model_version": inference.MODEL_VERSION, "model": state}, checkpoint)

    with pytest.raises(RuntimeError, match="Missing key"):
        inference.load_omnimira(checkpoint, CONFIG)


@pytest.mark.parametrize("modality", MODALITIES)
def test_extract_public_api_returns_ordered_float32_arrays(
    frozen_test_model, modality: str
) -> None:
    volume = np.zeros((96, 112, 96), dtype=np.float32)
    features = inference.extract_roi_features(
        frozen_test_model, volume, modality=modality
    )

    assert list(features) == ["aal3", "ho69", "yeo7"]
    assert features["aal3"].shape == (166, 128)
    assert features["ho69"].shape == (69, 128)
    assert features["yeo7"].shape == (7, 128)
    assert all(value.dtype == np.float32 for value in features.values())


def test_extract_rejects_unknown_modality(frozen_test_model) -> None:
    with pytest.raises(ValueError, match="unsupported modality"):
        inference.extract_roi_features(
            frozen_test_model,
            np.zeros((96, 112, 96), dtype=np.float32),
            modality="pet",
        )


def test_extract_rejects_batch_instead_of_silently_dropping_scans(
    frozen_test_model,
) -> None:
    with pytest.raises(ValueError, match="one scan at a time"):
        inference.extract_roi_features(
            frozen_test_model,
            np.zeros((2, 1, 96, 112, 96), dtype=np.float32),
            modality="t1",
        )


def test_extract_bypasses_full_forward_and_pretraining(
    frozen_test_model, monkeypatch
) -> None:
    def fail(*args, **kwargs):
        pytest.fail("pretraining path executed during feature extraction")

    monkeypatch.setattr(frozen_test_model, "forward", fail)
    monkeypatch.setattr(frozen_test_model.heads, "forward", fail)
    monkeypatch.setattr(frozen_test_model.patch_recon_head, "forward", fail)
    monkeypatch.setattr(frozen_test_model.roi_mae_decoder, "forward", fail)

    features = inference.extract_roi_features(
        frozen_test_model,
        np.zeros((96, 112, 96), dtype=np.float32),
        modality="t1",
    )
    assert features["aal3"].shape == (166, 128)


def test_cpu_extraction_is_repeatable(frozen_test_model) -> None:
    volume = np.random.default_rng(20260715).random(
        (96, 112, 96), dtype=np.float32
    )
    first = inference.extract_roi_features(frozen_test_model, volume, modality="ct")
    second = inference.extract_roi_features(frozen_test_model, volume, modality="ct")

    for atlas in first:
        np.testing.assert_allclose(first[atlas], second[atlas], rtol=1e-6, atol=1e-7)
