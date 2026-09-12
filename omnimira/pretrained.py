"""One-call loading and ROI extraction for the bundled OmniMIRA v9 model."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from omnimira.atlas import ATLAS_ORDER, fixed_patch_to_roi, roi_metadata
from omnimira.io import load_volume
from omnimira.models.v9 import OmniMIRAV9
from omnimira.schema import file_sha256, resource_path

MODALITY_IDS={"t1":0,"av45":1,"fdg":2,"ct":3}
MODEL_VERSION="omnimira-v9-fold0-epoch1000"
CHECKPOINT_SHA256="7ee25387da0cd30f3376325a7d788aff9e3398973ab09a786723dc3686c40fbd"

class OmniMIRAExtractor:
    """Frozen multi-modal encoder returning named ROI feature matrices."""
    def __init__(self,device="cpu",checkpoint=None):
        self.device=torch.device(device)
        path=Path(checkpoint).expanduser() if checkpoint else resource_path("checkpoints/omnimira_v9_fold0_epoch1000.pt")
        if file_sha256(path)!=CHECKPOINT_SHA256 and checkpoint is None: raise ValueError("Bundled checkpoint hash mismatch")
        payload=torch.load(path,map_location="cpu",weights_only=False)
        self.model=OmniMIRAV9().to(self.device).eval(); self.model.load_state_dict(payload["model"],strict=True)
        for parameter in self.model.parameters(): parameter.requires_grad=False
        self.checkpoint_path=path; self.checkpoint_sha256=file_sha256(path)
        self.memberships={name:torch.from_numpy(value.copy()).bool().to(self.device) for name,value in fixed_patch_to_roi().items()}

    @torch.no_grad()
    def extract_array(self,volume,modality):
        if modality not in MODALITY_IDS: raise ValueError(f"modality must be one of {tuple(MODALITY_IDS)}")
        x=torch.as_tensor(np.asarray(volume),dtype=torch.float32,device=self.device)
        if x.shape!=(96,112,96): raise ValueError("normalized volume must have shape (96,112,96)")
        out=self.model.encode(x[None,None],torch.tensor([MODALITY_IDS[modality]],device=self.device),self.memberships)
        return {name:out["atlases"][name]["tokens"][0].cpu().numpy().astype(np.float32) for name in ATLAS_ORDER}

    def extract(self,input_path,modality):
        """Load a model-space NIfTI or normalized NumPy scan and return ROI features."""
        return self.extract_array(load_volume(input_path,modality),modality)

    def save(self,input_path,output_path,modality):
        values=self.extract(input_path,modality); labels=roi_metadata()
        np.savez_compressed(output_path,model_version=np.str_(MODEL_VERSION),checkpoint_sha256=np.str_(self.checkpoint_sha256),modality=np.str_(modality),**{f"{name}_features":values[name] for name in ATLAS_ORDER},**{f"{name}_roi_ids":labels[name].ids for name in ATLAS_ORDER},**{f"{name}_roi_names":labels[name].names for name in ATLAS_ORDER})
        return Path(output_path)

def from_pretrained(device="cpu",checkpoint=None):
    """Load the bundled epoch-1000 model or an explicitly supplied compatible checkpoint."""
    return OmniMIRAExtractor(device=device,checkpoint=checkpoint)
