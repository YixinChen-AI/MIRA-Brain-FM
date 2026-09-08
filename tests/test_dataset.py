import json
import torch
from pathlib import Path
from omnimira.dataset import ManifestDataset, collate, MODALITY_ID, build_atlas_patch_to_roi

REPO = Path(__file__).resolve().parents[1]
ATLAS = {
    "aal3": REPO / "assets/atlases/AAL3v1_1mm.nii",
    "ho69": REPO / "assets/atlases/harvard_oxford_69_mni_1mm.nii.gz",
    "yeo7": REPO / "assets/atlases/yeo7_mni_1mm.nii.gz",
}


def _toy_manifest(tmp_path, normalized_npy_factory, n=3):
    entries = []
    for i in range(n):
        p = normalized_npy_factory(tmp_path, "t1", i)
        entries.append({"path": str(p), "modality": "t1", "age": 70.0 + i})
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(entries))
    return mpath


def test_dataset_item_and_collate(tmp_path, normalized_npy_factory):
    mpath = _toy_manifest(tmp_path, normalized_npy_factory, n=3)
    ds = ManifestDataset(str(mpath), atlas_files={k: str(v) for k, v in ATLAS.items()})
    item = ds[0]
    assert item["x"].shape == (1, 96, 112, 96)
    assert item["modality_id"] == MODALITY_ID["t1"]
    assert item["age_valid"] is True
    assert item["sex_valid"] is False
    batch = collate([ds[0], ds[1]])
    assert batch["x"].shape == (2, 1, 96, 112, 96)
    assert batch["modality_id"].shape == (2,)
    assert batch["age"].shape == (2,)
    assert batch["roi_volumes"]["aal3"].shape == (2, 166)
    assert not batch["roi_volume_valid"]["aal3"].any()
    assert batch["roi_distances"]["ho69"].shape == (2, 69, 69)
    assert not batch["spatial_valid"]["ho69"].any()


def test_build_atlas_patch_to_roi():
    p2r = build_atlas_patch_to_roi({k: str(v) for k, v in ATLAS.items()},
                                   img_size=(96, 112, 96), patch_size=8)
    assert set(p2r) == {"aal3", "ho69", "yeo7"}
    assert p2r["aal3"].shape == (166, 2016)
    assert p2r["aal3"].dtype == torch.bool
    assert p2r["aal3"].any(dim=1).all()
