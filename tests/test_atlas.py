import numpy as np
import pytest
import shutil
import subprocess
import sys
from pathlib import Path

from omnimira.atlas import (
    ATLAS_RESOURCES,
    atlas_label_path,
    atlas_path,
    build_patch_to_roi_lookup,
    dense_label_map,
    fixed_patch_to_roi,
    raw_label_ids,
)


def test_external_atlas_directory_overrides_packaged_binary(tmp_path, monkeypatch):
    source = atlas_path("aal3")
    external = tmp_path / "atlases"
    external.mkdir()
    target = external / source.name
    shutil.copyfile(source, target)
    label_source = atlas_label_path("aal3")
    label_target = external / label_source.name
    shutil.copyfile(label_source, label_target)

    monkeypatch.setenv("OMNIMIRA_ATLAS_DIR", str(external))

    assert atlas_path("aal3") == target
    assert atlas_label_path("aal3") == label_target
    assert atlas_path("aal3").name == ATLAS_RESOURCES["aal3"][0].split("/")[-1]


def test_external_atlas_directory_rejects_hash_mismatch(tmp_path, monkeypatch):
    source = atlas_path("aal3")
    external = tmp_path / "atlases"
    external.mkdir()
    target = external / source.name
    shutil.copyfile(source, target)
    with target.open("ab") as handle:
        handle.write(b"modified")

    monkeypatch.setenv("OMNIMIRA_ATLAS_DIR", str(external))

    with pytest.raises(ValueError, match="atlas hash mismatch: aal3"):
        atlas_path("aal3")


def test_external_atlas_directory_never_falls_back_to_packaged_binary(tmp_path, monkeypatch):
    monkeypatch.setenv("OMNIMIRA_ATLAS_DIR", str(tmp_path))

    with pytest.raises(FileNotFoundError, match="AAL3v1_1mm.nii"):
        atlas_path("aal3")


def test_external_atlas_directory_requires_matching_label_table(tmp_path, monkeypatch):
    source = atlas_path("aal3")
    shutil.copyfile(source, tmp_path / source.name)
    monkeypatch.setenv("OMNIMIRA_ATLAS_DIR", str(tmp_path))

    with pytest.raises(FileNotFoundError, match="AAL3v1_labels.txt"):
        atlas_label_path("aal3")


def test_verify_atlases_reports_external_verified_assets(tmp_path):
    external = tmp_path / "atlases"
    external.mkdir()
    for name in ATLAS_RESOURCES:
        source = atlas_path(name)
        shutil.copyfile(source, external / source.name)
        label_source = atlas_label_path(name)
        shutil.copyfile(label_source, external / label_source.name)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/verify_atlases.py",
            "--atlas-dir",
            str(external),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert '"aal3"' in result.stdout
    assert '"label_path"' in result.stdout
    assert '"n_rois": 166' in result.stdout


def test_patch_membership_preserves_every_overlapping_roi():
    atlas = np.full((8, 8, 8), 3, dtype=np.int32)
    atlas[4, 4, 4] = 7
    lookup = build_patch_to_roi_lookup(atlas, 8, valid_label_ids=(3, 7))
    assert lookup.tolist() == [[True], [True]]


def test_background_patch_is_minus_one():
    atlas = np.zeros((16, 16, 16), dtype=np.int32)
    atlas[4, 4, 4] = 5
    lookup = build_patch_to_roi_lookup(atlas, 8, valid_label_ids=(5,))
    assert lookup.shape == (1, 8)
    assert lookup[0, 0]
    assert not lookup[0, 1:].any()


def test_aal3_dense_mapping_contains_all_valid_raw_ids():
    ids = raw_label_ids("aal3")
    assert len(ids) == 166
    assert not {35, 36, 81, 82} & set(ids)
    assert {167, 168, 169, 170} <= set(ids)
    mapping = dense_label_map("aal3")
    assert sorted(mapping.values()) == list(range(166))


def test_public_patch_mappings_have_paper_grid_shape_and_dense_range():
    mappings = fixed_patch_to_roi()
    for name, expected in (("aal3", 166), ("ho69", 69), ("yeo7", 7)):
        assert mappings[name].shape == (expected, 2016)
        assert mappings[name].dtype == np.bool_
        assert mappings[name].any(axis=1).all()
