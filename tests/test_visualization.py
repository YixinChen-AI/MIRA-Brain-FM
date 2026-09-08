import numpy as np
from pathlib import Path
from omnimira.visualization import render_roi_map

REPO = Path(__file__).resolve().parents[1]
ATLAS = REPO / "assets" / "atlases" / "AAL3v1_1mm.nii"


def test_render_writes_figure(tmp_path):
    scores = np.linspace(0.0, 1.0, 166).astype(np.float32)  # one value per ROI token
    out = tmp_path / "roi_map.png"
    ret = render_roi_map(scores, str(ATLAS), str(out), z_slices=[90])
    assert Path(ret) == out
    assert out.exists()
    assert out.stat().st_size > 1000  # a real PNG, not an empty file


def test_render_accepts_dict_scores(tmp_path):
    scores = {1: 0.9, 41: 0.5, 42: 0.5}  # sparse per-label dict
    out = tmp_path / "roi_map_dict.png"
    render_roi_map(scores, str(ATLAS), str(out), z_slices=[90])
    assert out.exists() and out.stat().st_size > 1000
