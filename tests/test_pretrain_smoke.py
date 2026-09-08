import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import yaml

from omnimira.schema import file_sha256


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "omnimira_3atlas.yaml"


def _pretrain_module():
    path = ROOT / "scripts" / "pretrain.py"
    spec = importlib.util.spec_from_file_location("pretrain", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_v4_training_recipe_is_committed():
    cfg = yaml.safe_load(CONFIG.read_text())
    assert cfg["sampler"] == {
        "batch_size": 64,
        "n_per_modality": 16,
        "min_unique_subjects": 32,
    }
    assert cfg["optim"] == {
        "lr": 1.5e-4, "min_lr": 1e-6, "warmup_start_lr": 1.5e-6,
        "weight_decay": 0.05, "betas": [0.9, 0.95],
        "warmup_epochs": 30, "total_epochs": 300,
    }


def test_pretraining_rejects_invalid_manifest_before_model_construction(
    tmp_path, monkeypatch
):
    manifest = tmp_path / "invalid.json"
    manifest.write_text(json.dumps([
        {"path": str(tmp_path / "missing.npy"), "modality": "t1"}
    ]))
    module = _pretrain_module()
    monkeypatch.setattr("sys.argv", [
        "pretrain.py", "--manifest", str(manifest), "--epochs", "0", "--device", "cpu",
    ])
    with pytest.raises(SystemExit, match="manifest validation failed"):
        module.main()


def test_pretraining_help_runs_from_a_source_checkout_without_pythonpath():
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "pretrain.py"), "--help"],
        capture_output=True, text=True, check=False, env=env,
    )
    assert result.returncode == 0, result.stderr
    assert "--manifest" in result.stdout


def test_preflight_does_not_read_normalized_numpy_payloads(tmp_path, monkeypatch, capsys):
    """A launch preflight must stop before any image sidecar is required."""
    entries = []
    for modality in ("t1", "av45", "fdg", "ct"):
        for index in range(16):
            path = tmp_path / f"{modality}-{index}.npy"
            path.touch()
            entries.append({
                "path": str(path), "modality": modality,
                "subject_id": f"{modality}-{index}",
            })
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(entries), encoding="utf-8")

    module = _pretrain_module()
    monkeypatch.setattr("sys.argv", [
        "pretrain.py", "--manifest", str(manifest), "--preflight", "--device", "cpu",
    ])
    module.main()
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "preflight_ok"
    assert report["manifest"]["scans"] == 64
    assert report["atlas_names"] == ["aal3", "ho69", "yeo7"]


def test_lr_schedule_matches_linear_warmup_then_cosine_decay():
    schedule = _pretrain_module().learning_rate_at_epoch
    assert schedule(0, 300, 30, 1.5e-6, 1.5e-4, 1e-6) == 1.5e-6
    assert schedule(30, 300, 30, 1.5e-6, 1.5e-4, 1e-6) == 1.5e-4
    assert schedule(300, 300, 30, 1.5e-6, 1.5e-4, 1e-6) == 1e-6


def test_checkpoint_provenance_records_recipe_and_manifest_hashes(tmp_path):
    module = _pretrain_module()
    manifest = tmp_path / "manifest.json"
    config = tmp_path / "recipe.yaml"
    manifest.write_text("[]\n", encoding="utf-8")
    config.write_text("model: {}\n", encoding="utf-8")

    provenance = module.training_provenance(manifest, config, entries=[
        {"subject_id": "sub-1", "modality": "t1", "cohort": "adni"},
        {"subject_id": "sub-1", "modality": "fdg", "cohort": "adni"},
        {"subject_id": "sub-2", "modality": "ct", "cohort": "oasis"},
    ])

    assert provenance == {
        "manifest_sha256": file_sha256(manifest),
        "manifest_entries": 3,
        "manifest_modalities": {"ct": 1, "fdg": 1, "t1": 1},
        "manifest_unique_participants": 2,
        "manifest_cohort_count": 2,
        "config_sha256": file_sha256(config),
        "config_path": "recipe.yaml",
    }


def test_augmentations_redraw_across_epochs_and_replay_from_seed():
    augment = _pretrain_module().augment_batch
    x = torch.ones(3, 1, 4, 4, 4)
    modality = torch.tensor([0, 1, 3])
    brain = torch.ones_like(x, dtype=torch.bool)

    a0 = augment(x, modality, brain, seed=42, epoch=0, view=0)
    replay = augment(x, modality, brain, seed=42, epoch=0, view=0)
    a1 = augment(x, modality, brain, seed=42, epoch=1, view=0)
    torch.testing.assert_close(a0, replay)
    assert not torch.equal(a0[0], a1[0])
    assert not torch.equal(a0[1], a1[1])
    torch.testing.assert_close(a0[2], x[2])


def test_one_step_updates_paper_core_on_cpu():
    module = _pretrain_module()
    cfg = yaml.safe_load(CONFIG.read_text())
    cfg["model"].update({
        "img_size": [16, 16, 16], "embed_dim": 12, "mlp_depth": 1,
        "mlp_ratio": 2.0, "n_rois": 3, "contrastive_dim": 8,
        "decoder_depth": 1, "decoder_heads": 3,
    })
    cfg["atlases"] = [{"name": "toy", "path": "unused", "n_rois": 3}]
    cfg["heads"][2]["out_dim"] = 8
    model = module.build_model(cfg)
    loss_fn = module.OmniMIRALoss(
        head_names=tuple(item["name"] for item in cfg["heads"]),
        atlas_names=("toy",), patch_size=8,
    )
    optimizer = torch.optim.AdamW(
        [*model.parameters(), *loss_fn.parameters()], lr=1e-4,
        betas=(0.9, 0.95), weight_decay=0.05,
    )
    x = torch.randn(4, 1, 16, 16, 16)
    modalities = torch.tensor([0, 1, 2, 3])
    mapping = {"toy": torch.tensor([0, 0, 1, 1, 2, 2, 2, 2])}
    first = model(x, modalities, mapping)
    second = model.encode_roi_features(x + 0.01, modalities, mapping)
    result = loss_fn(
        module.assemble_atlas_outputs(first), recon_target=x,
        meta={
            "modality_id": modalities,
            "age": torch.tensor([20.0, 30.0, 40.0, 50.0]),
            "age_valid": torch.tensor([True, True, True, True]),
            "foreground_patch_mask": torch.ones(8, dtype=torch.bool),
            "aug_view2_roi": {"toy": second["atlases"]["toy"]["roi_tokens"]},
        },
    )
    optimizer.zero_grad(set_to_none=True)
    result["total"].backward()
    optimizer.step()
    assert torch.isfinite(result["total"])
    assert set(result["per_head"]) == {
        "patch_ae", "toy.roi_mae", "toy.cross_modal",
        "toy.augmentation_consistency", "toy.age",
    }
