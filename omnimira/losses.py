"""Seven paper-defined OmniMIRA pretraining objectives and active-only HUW."""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _masked_mean(values, mask):
    return values.masked_select(mask).mean() if mask.any() else None


def _volume_to_patches(volume, patch_size):
    batch, _, depth, height, width = volume.shape
    p = patch_size
    gd, gh, gw = depth // p, height // p, width // p
    return (volume.float().view(batch, 1, gd, p, gh, p, gw, p)
            .permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(batch, gd * gh * gw, -1))


def loss_patch_reconstruction(pred, target, foreground_patch_mask, patch_size):
    per_patch = ((_volume_to_patches(pred, patch_size)
                  - _volume_to_patches(target, patch_size)) ** 2).mean(-1)
    foreground = foreground_patch_mask.bool().unsqueeze(0).expand_as(per_patch)
    return _masked_mean(per_patch, foreground)


def loss_roi_mae(pred, target, roi_mask, roi_valid):
    mask = (roi_mask & roi_valid).unsqueeze(-1).expand_as(pred)
    return _masked_mean((pred.float() - target.detach().float()) ** 2, mask)


def _flatten_valid(tokens, valid):
    batch, rois, dim = tokens.shape
    scan = torch.arange(batch, device=tokens.device)[:, None].expand(batch, rois)
    roi = torch.arange(rois, device=tokens.device)[None, :].expand(batch, rois)
    flat = valid.reshape(-1)
    return tokens.reshape(-1, dim)[flat], scan.reshape(-1)[flat], roi.reshape(-1)[flat]


def _info_nce(tokens, positive, negative, tau):
    if not positive.any():
        return None
    similarity = tokens @ tokens.T / tau
    similarity = similarity - similarity.max(1, keepdim=True).values.detach()
    exponentials = similarity.exp()
    numerator = (exponentials * positive).sum(1)
    denominator = (exponentials * (positive | negative)).sum(1)
    active = positive.any(1)
    return -(numerator[active] / denominator[active].clamp_min(1e-12)).log().mean()


def loss_cross_modal(tokens, roi_valid, modality_id, tau):
    flat, scan, roi = _flatten_valid(tokens, roi_valid)
    if flat.shape[0] < 2:
        return None
    modality = modality_id[scan]
    self_mask = torch.eye(flat.shape[0], dtype=torch.bool, device=flat.device)
    same_roi = roi[:, None] == roi[None, :]
    positive = same_roi & (modality[:, None] != modality[None, :]) & ~self_mask
    negative = ~same_roi & ~self_mask
    return _info_nce(flat, positive, negative, tau)


def loss_augmentation_consistency(view1, view2, roi_valid, tau):
    first, scan, roi = _flatten_valid(F.normalize(view1, dim=-1), roi_valid)
    second, _, _ = _flatten_valid(F.normalize(view2, dim=-1), roi_valid)
    if first.shape[0] == 0:
        return None
    count = first.shape[0]
    tokens = torch.cat([first, second])
    scans = torch.cat([scan, scan])
    rois = torch.cat([roi, roi])
    views = torch.cat([torch.zeros(count, device=tokens.device, dtype=torch.long),
                       torch.ones(count, device=tokens.device, dtype=torch.long)])
    self_mask = torch.eye(2 * count, dtype=torch.bool, device=tokens.device)
    same_roi = rois[:, None] == rois[None, :]
    same_scan = scans[:, None] == scans[None, :]
    positive = same_roi & same_scan & (views[:, None] != views[None, :])
    negative = same_roi & ~same_scan & ~self_mask
    return _info_nce(tokens, positive, negative, tau)


def loss_age(pred, age, sample_valid, roi_valid):
    mask = (sample_valid[:, None] & roi_valid).unsqueeze(-1).expand_as(pred)
    target = age[:, None, None].expand_as(pred)
    return _masked_mean((pred - target).abs(), mask)


def _expand_sample_valid(sample_valid, roi_valid):
    return (sample_valid[:, None].expand_as(roi_valid)
            if sample_valid.ndim == 1 else sample_valid)


def loss_roi_volume(pred, target, sample_valid, roi_valid):
    valid = _expand_sample_valid(sample_valid, roi_valid) & roi_valid & target.isfinite()
    return _masked_mean((pred.squeeze(-1) - target).abs(), valid)


def loss_spatial_metric(pred_coords, target_distances, sample_valid, roi_valid):
    predicted = torch.cdist(pred_coords.float(), pred_coords.float())
    valid_roi = _expand_sample_valid(sample_valid, roi_valid) & roi_valid
    pair_valid = valid_roi[:, :, None] & valid_roi[:, None, :]
    diagonal = torch.eye(pred_coords.shape[1], dtype=torch.bool,
                         device=pred_coords.device).unsqueeze(0)
    return _masked_mean((predicted - target_distances).abs(), pair_valid & ~diagonal)


def _patch(output, valid, meta):
    target = meta.get("recon_target")
    foreground = meta.get("foreground_patch_mask")
    if output.get("patch_recon") is None or target is None or foreground is None:
        return None
    return loss_patch_reconstruction(output["patch_recon"], target, foreground,
                                     meta["patch_size"])


def _roi_mae(output, valid, meta):
    if output.get("roi_mae_pred") is None or output.get("roi_mae_target") is None:
        return None
    return loss_roi_mae(output["roi_mae_pred"], output["roi_mae_target"],
                        output["roi_mask"], valid)


def _cross(output, valid, meta):
    value = output.get("head_proj", {}).get("cross_modal")
    required = "modality_id" in meta
    return None if value is None or not required else loss_cross_modal(
        value, valid, meta["modality_id"], meta["tau"]
    )


def _augmentation(output, valid, meta):
    second = meta.get("aug_view2_roi", {}).get(meta["atlas_name"])
    return None if second is None else loss_augmentation_consistency(
        output["roi_tokens"], second, valid, meta["tau"]
    )


def _age(output, valid, meta):
    pred = output.get("head_proj", {}).get("age")
    if pred is None or "age" not in meta or not meta.get("age_valid", torch.tensor(False)).any():
        return None
    return loss_age(pred, meta["age"], meta["age_valid"], valid)


def _volume(output, valid, meta):
    pred = output.get("head_proj", {}).get("roi_volume")
    targets = meta.get("roi_volumes", {}).get(meta["atlas_name"])
    sample_valid = meta.get("roi_volume_valid", {}).get(meta["atlas_name"])
    if pred is None or targets is None or sample_valid is None or not sample_valid.any():
        return None
    return loss_roi_volume(pred, targets, sample_valid, valid)


def _spatial(output, valid, meta):
    pred = output.get("head_proj", {}).get("spatial_metric")
    targets = meta.get("roi_distances", {}).get(meta["atlas_name"])
    sample_valid = meta.get("spatial_valid", {}).get(meta["atlas_name"])
    if pred is None or targets is None or sample_valid is None or not sample_valid.any():
        return None
    return loss_spatial_metric(pred, targets, sample_valid, valid)


LOSS_REGISTRY = {
    "patch_ae": _patch,
    "roi_mae": _roi_mae,
    "cross_modal": _cross,
    "augmentation_consistency": _augmentation,
    "age": _age,
    "roi_volume": _volume,
    "spatial_metric": _spatial,
}


class OmniMIRALoss(nn.Module):
    """Apply HUW only to objective terms that are active in this batch."""

    def __init__(self, tau=0.2, age_normalize=1.0,
                 patch_size=8, head_names=None, atlas_names=("default",),
                 use_uncertainty_weighting=True, supported_objectives=None,
                 log_sigma_clamp=(-5.0, 5.0)):
        super().__init__()
        del age_normalize
        self.tau = tau
        self.patch_size = patch_size
        self.head_names = tuple(head_names or LOSS_REGISTRY)
        self.atlas_names = tuple(atlas_names)
        unknown = set(self.head_names) - set(LOSS_REGISTRY)
        if unknown:
            raise ValueError(f"unknown objectives: {sorted(unknown)}")
        roi_objectives = tuple(name for name in self.head_names if name != "patch_ae")
        self.supported_objectives = {
            atlas: tuple((supported_objectives or {}).get(atlas, roi_objectives))
            for atlas in self.atlas_names
        }
        for atlas, names in self.supported_objectives.items():
            unsupported = set(names) - set(roi_objectives)
            if unsupported:
                raise ValueError(
                    f"unsupported objectives for {atlas}: {sorted(unsupported)}"
                )
        self.task_keys = []
        if "patch_ae" in self.head_names:
            self.task_keys.append("patch_ae")
        self.task_keys.extend(f"{atlas}.{name}" for atlas in self.atlas_names
                              for name in self.supported_objectives[atlas])
        self.log_sigma2 = nn.Parameter(torch.zeros(len(self.task_keys)))
        self.use_uncertainty_weighting = use_uncertainty_weighting
        self.log_sigma_clamp = tuple(float(value) for value in log_sigma_clamp)

    @torch.no_grad()
    def clamp_log_sigma2_(self):
        self.log_sigma2.clamp_(*self.log_sigma_clamp)

    def forward(self, atlas_outputs, meta=None, recon_target=None,
                model_heads=None, **unused):
        del model_heads, unused
        common = dict(meta or {})
        common.update(tau=self.tau, patch_size=self.patch_size,
                      recon_target=recon_target)
        losses: Dict[str, torch.Tensor] = {}
        effective: Dict[str, torch.Tensor] = {}
        total = self.log_sigma2.new_zeros(())
        sigma = self.log_sigma2.clamp(*self.log_sigma_clamp)
        sigma_by_key = dict(zip(self.task_keys, sigma))
        for atlas_name in self.atlas_names:
            output = atlas_outputs.get(atlas_name)
            if output is None or output.get("roi_valid") is None:
                continue
            local = dict(common, atlas_name=atlas_name)
            for name in self.head_names:
                if name == "patch_ae" and atlas_name != self.atlas_names[0]:
                    continue
                if (name != "patch_ae"
                        and name not in self.supported_objectives[atlas_name]):
                    continue
                value = LOSS_REGISTRY[name](output, output["roi_valid"], local)
                if value is None:
                    continue
                key = name if name == "patch_ae" else f"{atlas_name}.{name}"
                losses[key] = value
                if self.use_uncertainty_weighting:
                    s = sigma_by_key[key]
                    effective[key] = 0.5 * torch.exp(-s) * value
                    total = total + effective[key] + 0.5 * s
                else:
                    effective[key] = value
                    total = total + value
        return {"per_head": losses, "per_head_effective": effective,
                "weighted": total, "total": total,
                "log_sigma2": sigma.detach()}
