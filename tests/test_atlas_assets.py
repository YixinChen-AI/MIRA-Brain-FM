import numpy as np
from pathlib import Path
import pytest
from omnimira.atlas import AtlasMeta

ASSETS = Path(__file__).resolve().parents[1] / "assets" / "atlases"
CASES = [
    ("AAL3v1_1mm.nii", 166),
    ("harvard_oxford_69_mni_1mm.nii.gz", 69),
    ("yeo7_mni_1mm.nii.gz", 7),
]


@pytest.mark.parametrize("fname,n_rois", CASES)
def test_atlas_resizes_and_has_expected_rois(fname, n_rois):
    path = ASSETS / fname
    if not path.exists():
        pytest.skip(f"{fname} not yet fetched")
    meta = AtlasMeta(str(path), n_rois=n_rois, target_shape=(96, 112, 96))
    assert meta.atlas_volume.shape == (96, 112, 96)
    labels = np.unique(meta.atlas_volume[meta.atlas_volume > 0])
    # AAL3v1 uses label IDs 1-170 with 4 gaps (35,36,81,82) → 166 unique labels;
    # max label is 170 (> n_rois=166), so check count rather than max.
    assert len(labels) == n_rois  # exact count of occupied labels must match n_rois
