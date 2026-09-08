"""OmniMIRA — per-ROI universal brain feature space."""
from omnimira.models.omnimira import OmniMIRA, HeadSpec, HEAD_FAMILIES
from omnimira.inference import (
    extract_roi_features,
    load_omnimira,
    save_roi_features,
    source_metadata,
    validate_source_metadata,
)
from omnimira.atlas import AtlasMeta, build_patch_to_roi_lookup
from omnimira.io import load_volume
from omnimira.losses import OmniMIRALoss, LOSS_REGISTRY

__all__ = [
    "OmniMIRA", "HeadSpec", "HEAD_FAMILIES",
    "load_omnimira", "extract_roi_features", "save_roi_features",
    "source_metadata", "validate_source_metadata",
    "AtlasMeta", "build_patch_to_roi_lookup",
    "load_volume",
    "OmniMIRALoss", "LOSS_REGISTRY",
]
__version__ = "0.1.0"
