"""Render a per-ROI scalar map (one value per ROI token) onto brain slices.

Example:
    python examples/visualize_roi_map.py \
        --scores roi_scores.npy --atlas assets/atlases/AAL3v1_1mm.nii \
        --out roi_map.png
"""
import argparse
from pathlib import Path

import numpy as np

from omnimira.visualization import render_roi_map

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help=".npy of shape (n_rois,)")
    ap.add_argument("--atlas", default=str(REPO / "assets/atlases/AAL3v1_1mm.nii"))
    ap.add_argument("--out", default="roi_map.png")
    ap.add_argument("--cmap", default="hot")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    scores = np.load(args.scores)
    out = render_roi_map(scores, args.atlas, args.out, cmap=args.cmap, title=args.title)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
