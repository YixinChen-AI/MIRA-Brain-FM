"""Frozen atlas mappings and ROI metadata for OmniMIRA."""
from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to

from omnimira.schema import (
    InputContractError,
    MODEL_SHAPE,
    file_sha256,
    resource_path,
    verified_resource,
)


ATLAS_ORDER = ("aal3", "ho69", "yeo7")
ATLAS_RESOURCES = {
    "aal3": (
        "assets/atlases/AAL3v1_1mm.nii",
        "aa44bb1767594f560e811cd917fee5833c2f843b81a8febd614db20284b11a34",
        166,
        "assets/atlases/AAL3v1_labels.txt",
    ),
    "ho69": (
        "assets/atlases/harvard_oxford_69_mni_1mm.nii.gz",
        "9ad6acc55d44c988bf30e997737d20aa6b94453624f4669bdad39f3a7a104e2e",
        69,
        "assets/atlases/HarvardOxford69_labels.txt",
    ),
    "yeo7": (
        "assets/atlases/yeo7_mni_1mm.nii.gz",
        "5b13f9eebcfdb103455a4a424f762f936a2df1c4c185ae3726f4eff68e3cf1a7",
        7,
        "assets/atlases/Yeo7_labels.txt",
    ),
}
ATLAS_LABEL_HASHES = {
    "aal3": "5c643d5bef449948af5d92f116cc2eeb670777a500882ce6e523a033837ffd08",
    "ho69": "6ce228f7e86ac10838562b9c39301b0bb0ffa54678edf1bf7c91ad2fefb9c4d7",
    "yeo7": "ca23ed960895a5b847603672ef2110b4e8ff13b0552359920b18fa58abdf48b8",
}


@dataclass(frozen=True)
class RoiMetadata:
    ids: np.ndarray
    names: np.ndarray


def _atlas_resource_path(relative_path: str, expected_hash: str, name: str, kind: str) -> Path:
    external_root = os.environ.get("OMNIMIRA_ATLAS_DIR")
    if external_root:
        candidate = Path(external_root).expanduser() / Path(relative_path).name
        if not candidate.is_file():
            raise FileNotFoundError(
                f"{kind} for atlas {name!r} is unavailable in OMNIMIRA_ATLAS_DIR: "
                f"{candidate.name}"
            )
        path = candidate
    else:
        path = None
    if path is None:
        try:
            path = resource_path(relative_path)
        except InputContractError as exc:
            raise FileNotFoundError(
                f"{kind} for atlas {name!r} is unavailable; set OMNIMIRA_ATLAS_DIR to a directory "
                f"containing {Path(relative_path).name}"
            ) from exc
    if file_sha256(path) != expected_hash:
        raise ValueError(f"{kind} hash mismatch: {name}")
    return path


def atlas_path(name: str) -> Path:
    """Resolve one verified atlas, preferring ``OMNIMIRA_ATLAS_DIR`` when set."""
    if name not in ATLAS_RESOURCES:
        raise ValueError(f"unknown atlas: {name}")
    relative_path, expected_hash, _, _ = ATLAS_RESOURCES[name]
    return _atlas_resource_path(relative_path, expected_hash, name, "atlas")


def atlas_label_path(name: str) -> Path:
    """Resolve the verified ROI label table paired with an atlas."""
    if name not in ATLAS_RESOURCES:
        raise ValueError(f"unknown atlas: {name}")
    _, _, _, relative_path = ATLAS_RESOURCES[name]
    return _atlas_resource_path(relative_path, ATLAS_LABEL_HASHES[name], name, "label table")


def _verified_atlas(name: str):
    return atlas_path(name)


def atlas_hashes() -> dict[str, str]:
    """Return exact SHA256 identities in fixed public atlas order."""
    return {name: ATLAS_RESOURCES[name][1] for name in ATLAS_ORDER}


def _label_table(name: str) -> dict[int, str]:
    names: dict[int, str] = {}
    for line in atlas_label_path(name).read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        index, label = line.split("\t", 1) if "\t" in line else line.split(maxsplit=1)[:2]
        names[int(index)] = label.split()[0] if name == "aal3" else label
    return names


@lru_cache(maxsize=None)
def raw_label_ids(name: str) -> tuple[int, ...]:
    """Return sorted occupied raw atlas IDs used by the public model."""
    image = np.asarray(nib.load(_verified_atlas(name)).dataobj, dtype=np.int32)
    ids = tuple(int(value) for value in np.unique(image) if value > 0)
    expected = ATLAS_RESOURCES[name][2]
    if len(ids) != expected:
        raise ValueError(f"{name} contains {len(ids)} occupied labels; expected {expected}")
    return ids


@lru_cache(maxsize=None)
def dense_label_map(name: str) -> dict[int, int]:
    return {raw_id: dense_id for dense_id, raw_id in enumerate(raw_label_ids(name))}


@lru_cache(maxsize=1)
def roi_metadata() -> dict[str, RoiMetadata]:
    """Return raw atlas IDs and names in the model's dense row order."""
    result = {}
    for name in ATLAS_ORDER:
        ids = raw_label_ids(name)
        labels = _label_table(name)
        result[name] = RoiMetadata(
            ids=np.asarray(ids, dtype=np.int32),
            names=np.asarray([labels[index] for index in ids], dtype=np.str_),
        )
    return result


@lru_cache(maxsize=1)
def fixed_patch_to_roi() -> dict[str, np.ndarray]:
    """Build ROI-by-patch memberships in canonical RAS space."""
    result = {}
    for name in ATLAS_ORDER:
        image = resample_atlas_to_model(name)
        result[name] = build_patch_to_roi_lookup(
            image, patch_size=8, valid_label_ids=raw_label_ids(name)
        )
    return result


@lru_cache(maxsize=None)
def resample_atlas_to_model(name: str) -> np.ndarray:
    """Nearest-neighbour resample an atlas to the public RAS model grid."""
    source = nib.load(_verified_atlas(name))
    template = nib.load(verified_resource("model_grid"))
    target = (MODEL_SHAPE, np.asarray(template.affine, dtype=np.float64))
    aligned = resample_from_to(source, target, order=0, mode="constant", cval=0)
    return np.asarray(aligned.dataobj, dtype=np.int32)

# ---------------------------------------------------------------------------
# Lobe assignment rules (applied in priority order; first match wins)
# ---------------------------------------------------------------------------
LOBE_RULES = [
    # (lobe_id, list of name prefixes)
    (0, ["Cerebellum", "Vermis"]),                                           # cerebellum
    (1, ["Thalamus", "Thal_", "Caudate", "Putamen", "Pallidum", "N_Acc",
         "VTA", "Red_N", "SN_pc", "SN_pr", "LC", "Raphe_"]),                 # subcortical
    (2, ["Hippocampus", "ParaHippocampal", "Amygdala", "Insula",
         "Cingulate_Mid", "Cingulate_Post"]),                                 # limbic
    (3, ["Occipital_", "Calcarine", "Cuneus", "Lingual"]),                   # occipital
    (4, ["Temporal_", "Heschl", "Fusiform"]),                                # temporal
    (5, ["Parietal_", "Postcentral", "Precuneus", "SupraMarginal",
         "Angular"]),                                                          # parietal
    (6, ["Precentral", "Frontal_", "ACC_", "Cingulate_Ant", "OFC",
         "Olfactory", "Rectus", "Paracentral_Lobule",
         "Supp_Motor_Area", "Rolandic_Oper"]),                                # frontal
]
DEFAULT_LOBE = 6  # frontal as catch-all


def _name_to_lobe(name: str) -> int:
    """Map an ROI name to a lobe id via prefix matching (case-insensitive)."""
    name_lower = name.lower()
    for lobe_id, prefixes in LOBE_RULES:
        for prefix in prefixes:
            if name_lower.startswith(prefix.lower()):
                return lobe_id
    return DEFAULT_LOBE


def _parse_labels_file(labels_path: str, raw_ids) -> np.ndarray:
    """Parse a labels file with format '<index> <name> <internal_id>' per line.

    Returns an int64 array of length n_rois where arr[roi_idx] = lobe_id.
    roi_idx = label_index - 1 (1-based labels -> 0-based array).
    Entries not found in the file default to DEFAULT_LOBE.
    """
    raw_to_dense = {raw_id: index for index, raw_id in enumerate(raw_ids)}
    lobe_arr = np.full(len(raw_ids), DEFAULT_LOBE, dtype=np.int64)
    with open(labels_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                idx = int(parts[0])
            except ValueError:
                continue
            name = parts[1]
            if idx in raw_to_dense:
                lobe_arr[raw_to_dense[idx]] = _name_to_lobe(name)
    return lobe_arr


class AtlasMeta:
    """Per-ROI metadata derived from an integer-labeled NIfTI atlas.

    Atlases are resampled in world coordinates to the canonical RAS grid. All
    `roi_centroids` are reported in the *target-shape* voxel space so that
    downstream patch-to-ROI lookups on a target-shaped volume are consistent.

    Only occupied raw labels are represented, in ascending order.

    Args:
        atlas_path:   Path to NIfTI atlas file.
        n_rois:       Number of occupied ROIs expected.
        target_shape: Must be the public model shape when provided.
        labels_path:  Optional path to a labels file with lines of the form
                      ``<index> <name> <internal_id>``. Used to assign each
                      ROI to one of 7 anatomical lobes. If None or missing,
                      ``roi_lobe`` defaults to all zeros and a warning is
                      printed.
    """

    def __init__(self, atlas_path: str, n_rois: int, target_shape=None,
                 labels_path=None):
        assert os.path.exists(atlas_path), f"MISSING atlas: {atlas_path}"
        self.atlas_path = atlas_path
        self.n_rois = n_rois
        source = nib.load(atlas_path)
        source_ids = tuple(
            int(value) for value in np.unique(np.asarray(source.dataobj, dtype=np.int32))
            if value > 0
        )
        if len(source_ids) != n_rois:
            raise ValueError(f"atlas contains {len(source_ids)} occupied labels; expected {n_rois}")
        if target_shape is not None and tuple(target_shape) != MODEL_SHAPE:
            raise ValueError("AtlasMeta target_shape must match the public model grid")
        if target_shape is not None:
            template = nib.load(verified_resource("model_grid"))
            source = resample_from_to(
                source, (MODEL_SHAPE, template.affine), order=0,
                mode="constant", cval=0,
            )
        img = np.asarray(source.dataobj, dtype=np.int32)
        self.atlas_volume = img
        self.raw_label_ids = source_ids

        centroids = np.full((n_rois, 3), np.nan, dtype=np.float32)
        empty = []
        for dense_id, raw_id in enumerate(self.raw_label_ids):
            coords = np.argwhere(img == raw_id)
            if coords.size == 0:
                empty.append(dense_id)
                continue
            centroids[dense_id] = coords.mean(axis=0)
        self.empty_rois = set(empty)
        if empty:
            print(f"[AtlasMeta] {len(empty)} empty ROI labels "
                  f"(e.g. {sorted(empty)[:6]}) - tolerated; no patches "
                  f"will be assigned to them")
        self.roi_centroids = centroids

        cx = img.shape[0] / 2
        hemi = []
        for c in centroids:
            if np.any(np.isnan(c)):
                hemi.append(2)
            elif c[0] < cx:
                hemi.append(0)
            elif c[0] > cx:
                hemi.append(1)
            else:
                hemi.append(2)
        self.roi_hemisphere = np.array(hemi, dtype=np.int64)

        # ------------------------------------------------------------------
        # Lobe assignment
        # ------------------------------------------------------------------
        if labels_path is not None and os.path.exists(labels_path):
            self.roi_lobe = _parse_labels_file(labels_path, self.raw_label_ids)
            # Empty ROIs get sentinel -1 (never positively matched in masks)
            for roi_0 in self.empty_rois:
                self.roi_lobe[roi_0] = -1
            print(f"[AtlasMeta] Loaded lobe assignments from {labels_path}")
        else:
            if labels_path is not None:
                print(f"[AtlasMeta] WARNING: labels_path={labels_path!r} not found; "
                      f"roi_lobe defaults to zeros (lobe subspace is no-op)")
            else:
                print("[AtlasMeta] WARNING: labels_path not provided; "
                      "roi_lobe defaults to zeros (lobe subspace is no-op)")
            self.roi_lobe = np.zeros(n_rois, dtype=np.int64)

    def spatial_neighbors(self, k: int) -> list:
        n = self.n_rois
        c = self.roi_centroids
        diff = c[:, None, :] - c[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        dist = np.where(np.isnan(dist), np.inf, dist)
        np.fill_diagonal(dist, np.inf)
        out = []
        for i in range(n):
            if i in self.empty_rois:
                out.append(set())
                continue
            idx = np.argsort(dist[i])[:k]
            out.append(set(int(j) for j in idx if not np.isinf(dist[i, j])))
        return out


def build_patch_to_roi_lookup(
    atlas: np.ndarray, patch_size: int, valid_label_ids=None
) -> np.ndarray:
    """Return a dense ROI-by-patch membership map.

    A patch belongs to every ROI with which its voxel cube overlaps. This keeps
    small atlas regions represented while leaving patch encoding local.
    """
    if atlas.ndim != 3 or any(size % patch_size for size in atlas.shape):
        raise ValueError("atlas shape must be three-dimensional and divisible by patch_size")
    ids = tuple(
        int(value) for value in (
            valid_label_ids if valid_label_ids is not None
            else sorted(value for value in np.unique(atlas) if value > 0)
        )
    )
    unknown = set(int(value) for value in np.unique(atlas) if value > 0) - set(ids)
    if unknown:
        raise ValueError(f"atlas contains unknown labels: {sorted(unknown)}")
    gd, gh, gw = (size // patch_size for size in atlas.shape)
    patches = atlas.reshape(
        gd, patch_size, gh, patch_size, gw, patch_size
    ).transpose(0, 2, 4, 1, 3, 5).reshape(gd * gh * gw, -1)
    output = np.zeros((len(ids), patches.shape[0]), dtype=np.bool_)
    for dense_id, raw_id in enumerate(ids):
        output[dense_id] = np.any(patches == raw_id, axis=1)
    return output
