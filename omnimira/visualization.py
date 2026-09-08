"""Render a per-ROI scalar map onto brain slices.

Follows the project rendering SOP: each ROI region is gaussian-smoothed
(sigma=0.8) and outlined with a contour at level 0.4 — never a raw binary
overlay. matplotlib is an optional dependency (`pip install omnimira[viz]`)
and is imported lazily so the core package imports without it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np
import nibabel as nib
from scipy import ndimage


def render_roi_map(scores: Union[np.ndarray, Sequence[float], Dict[int, float]],
                   atlas_path: str,
                   out_path: str,
                   *,
                   z_slices: Optional[Sequence[int]] = None,
                   cmap: str = "hot",
                   vmin: Optional[float] = None,
                   vmax: Optional[float] = None,
                   title: Optional[str] = None,
                   dpi: int = 300) -> str:
    """Paint per-ROI scalar scores onto axial brain slices.

    Args:
        scores: array (n_rois,) or dict {label_1indexed: value}. For an array,
            element k is the value for atlas label k+1 (matching the ROI-token
            order returned by extract_roi_features).
        atlas_path: integer-labeled NIfTI atlas (any resolution).
        out_path: output figure path; the extension (.png/.pdf/.svg) sets format.
        z_slices: array z-indices to display; default = 3 evenly spaced slices
            through the labeled volume.
        cmap / vmin / vmax: colormap and score range (default = data range).
        title: optional figure suptitle.

    Returns:
        The written out_path (str).
    """
    try:
        import matplotlib
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "render_roi_map requires matplotlib; install with "
            "`pip install omnimira[viz]`") from e
    # Set a non-interactive file backend only if the caller hasn't chosen one.
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    atlas = np.asarray(nib.load(str(atlas_path)).dataobj).astype(int)

    if isinstance(scores, dict):
        score_map = {int(k): float(v) for k, v in scores.items()}
    else:
        s = np.asarray(scores, dtype=np.float64).ravel()
        score_map = {i + 1: float(s[i]) for i in range(len(s))}

    finite_vals = [v for v in score_map.values() if np.isfinite(v)]
    if not finite_vals:
        raise ValueError("scores contains no finite values to render")
    lo = min(finite_vals) if vmin is None else vmin
    hi = max(finite_vals) if vmax is None else vmax
    if hi <= lo:
        hi = lo + 1e-6
    norm = Normalize(vmin=lo, vmax=hi)
    cmap_obj = matplotlib.colormaps[cmap]

    labeled = atlas > 0
    if z_slices is None:
        znz = np.where(labeled.any(axis=(0, 1)))[0]
        z_slices = list(np.linspace(znz.min(), znz.max(), 5).astype(int)[1:-1])
    z_slices = list(z_slices)

    n = len(z_slices)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, z in zip(axes, z_slices):
        roi = atlas[:, :, int(z)].T
        h, w = roi.shape
        img = np.ones((h, w, 3), dtype=np.float32)
        # Faint grey brain silhouette from the atlas itself (no MNI template dep).
        sil = ndimage.gaussian_filter((roi > 0).astype(np.float32), sigma=0.8)
        sil = np.clip(sil, 0, 1)
        for ch in range(3):
            img[..., ch] = 1.0 - sil * 0.12
        labels_here = sorted(int(l) for l in set(roi[roi > 0].flat))
        # Fill
        for label in labels_here:
            v = score_map.get(label)
            if v is None or not np.isfinite(v):
                continue
            mask = roi == label
            rgba = cmap_obj(norm(v))
            smooth = np.clip(ndimage.gaussian_filter(mask.astype(np.float32), sigma=0.8), 0, 1)
            for ch in range(3):
                img[..., ch] = img[..., ch] * (1 - smooth) + rgba[ch] * smooth
        ax.imshow(img, origin="lower", aspect="equal")
        # Contour outlines
        for label in labels_here:
            v = score_map.get(label)
            if v is None or not np.isfinite(v):
                continue
            mask = roi == label
            smooth = ndimage.gaussian_filter(mask.astype(np.float32), sigma=0.8)
            ax.contour(smooth, levels=[0.4], colors=[(0.2, 0.2, 0.2)],
                       linewidths=0.5, origin="lower")
        ax.set_title(f"z={int(z)}", fontsize=9)
        ax.axis("off")

    sm = ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02)
    if title:
        fig.suptitle(title, fontsize=11)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)
