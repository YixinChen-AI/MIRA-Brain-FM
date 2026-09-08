"""Paper-defined OmniMIRA MCLP encoder, AART tokenizer, and pretraining heads."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


PUBLIC_COMPONENT_NAMES = ("MCLP", "AART")
RELEASE_HEAD_SPECS = (
    ("patch_ae", "patch_recon", 0),
    ("roi_mae", "roi_mae", 0),
    ("cross_modal", "contrast", 64),
    ("augmentation_consistency", "augmentation", 0),
    ("age", "regression", 1),
    ("roi_volume", "roi_volume", 1),
    ("spatial_metric", "spatial", 3),
)
ACTIVE_HEAD_NAMES = tuple(item[0] for item in RELEASE_HEAD_SPECS)
_INTEGER_DTYPES = {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}


def _build_1d_sincos(embed_dim: int, positions: np.ndarray) -> np.ndarray:
    if embed_dim % 2:
        raise ValueError("per-axis positional dimension must be even")
    omega = 1.0 / (10000.0 ** (np.arange(embed_dim // 2) / (embed_dim / 2.0)))
    phase = np.einsum("m,d->md", positions.astype(np.float64), omega)
    return np.concatenate([np.sin(phase), np.cos(phase)], axis=1)


def build_3d_sincos_pos_embed(
    embed_dim: int, grid_size: tuple[int, int, int]
) -> np.ndarray:
    per_axis = (embed_dim // 3) // 2 * 2
    grids = []
    for axis, size in enumerate(grid_size):
        shape = [1, 1, 1, per_axis]
        shape[axis] = size
        expanded = [*grid_size, per_axis]
        grids.append(np.broadcast_to(
            _build_1d_sincos(per_axis, np.arange(size)).reshape(shape), expanded
        ))
    result = np.concatenate(grids, axis=-1)
    if result.shape[-1] < embed_dim:
        result = np.pad(result, [(0, 0), (0, 0), (0, 0),
                                 (0, embed_dim - result.shape[-1])])
    return result.reshape(-1, embed_dim)


@dataclass(frozen=True)
class HeadSpec:
    name: str
    family: str
    out_dim: int


HEAD_FAMILIES = {
    "contrast": lambda dim, spec: nn.Linear(dim, spec.out_dim, bias=False),
    "regression": lambda dim, spec: nn.Linear(dim, 1),
    "roi_volume": lambda dim, spec: nn.Linear(dim, 1),
    "spatial": lambda dim, spec: nn.Linear(dim, 3),
}


class CondLN(nn.Module):
    def __init__(self, dim: int, n_modalities: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.scale = nn.Embedding(n_modalities, dim)
        self.shift = nn.Embedding(n_modalities, dim)
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.shift.weight)

    def forward(self, x: torch.Tensor, modality: torch.Tensor) -> torch.Tensor:
        return (
            self.norm(x) * (1 + self.scale(modality).unsqueeze(1))
            + self.shift(modality).unsqueeze(1)
        )


class ModalityPatchEmbed3D(nn.Module):
    """Four modality-specific, non-overlapping 3D patch embeddings."""

    def __init__(self, img_size, patch_size, embed_dim, n_modalities):
        super().__init__()
        self.n_patches = math.prod(size // patch_size for size in img_size)
        self.convs = nn.ModuleList([
            nn.Conv3d(1, embed_dim, patch_size, stride=patch_size)
            for _ in range(n_modalities)
        ])

    def forward(self, x: torch.Tensor, modality: torch.Tensor) -> torch.Tensor:
        output = x.new_empty(x.shape[0], self.n_patches, self.convs[0].out_channels)
        for modality_id, conv in enumerate(self.convs):
            rows = torch.where(modality == modality_id)[0]
            if rows.numel():
                features = conv(x[rows]).flatten(2).transpose(1, 2)
                output[rows] = features
        return output


class MLPBlockCond(nn.Module):
    """Shared residual MLP applied independently at every patch position."""

    def __init__(self, dim: int, mlp_ratio: float, n_modalities: int):
        super().__init__()
        self.norm = CondLN(dim, n_modalities)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x: torch.Tensor, modality: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(self.norm(x, modality))


class AART(nn.Module):
    """Pool patches by dense ROI index using one learned scalar score per patch."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.score = nn.Linear(embed_dim, 1, bias=False)

    @staticmethod
    def dense_roi_indices(patch_to_roi, n_rois):
        """Map occupied raw atlas labels to contiguous tokenizer indices."""
        labels = patch_to_roi.clone()
        foreground = labels >= 0
        if n_rois == 166 and foreground.any() and labels[foreground].max() >= 166:
            placeholders = (34, 35, 80, 81)  # raw AAL3 IDs 35, 36, 81, 82
            occupied = foreground.clone()
            for placeholder in placeholders:
                occupied &= labels != placeholder
            for placeholder in reversed(placeholders):
                labels = labels - ((labels > placeholder) & occupied).to(labels.dtype)
            labels[~occupied] = -1
            return labels
        if foreground.any() and labels[foreground].max() >= n_rois:
            unique = torch.unique(labels[foreground], sorted=True)
            if unique.numel() > n_rois:
                raise ValueError("atlas has more occupied labels than configured ROIs")
            dense = torch.full_like(labels, -1)
            for index, raw_label in enumerate(unique):
                dense[labels == raw_label] = index
            return dense
        return labels

    def forward(self, patch_tokens, patch_to_roi, n_rois):
        batch, n_patches, dim = patch_tokens.shape
        if patch_to_roi.ndim == 2:
            if patch_to_roi.shape != (n_rois, n_patches):
                raise ValueError(
                    f"patch_to_roi must have shape ({n_rois},{n_patches})"
                )
            roi_index, patch_index = torch.where(patch_to_roi.bool())
            scores = self.score(patch_tokens).squeeze(-1)[:, patch_index]
            index = roi_index.unsqueeze(0).expand(batch, -1)
            maxima = patch_tokens.new_full((batch, n_rois), float("-inf"))
            maxima.scatter_reduce_(1, index, scores, reduce="amax", include_self=True)
            exponentials = (scores - maxima.gather(1, index)).exp()
            denominators = patch_tokens.new_zeros(batch, n_rois)
            denominators.scatter_add_(1, index, exponentials)
            weights = exponentials / denominators.gather(1, index).clamp_min(1e-12)
            pooled = patch_tokens.new_zeros(batch, n_rois, dim)
            pooled.scatter_add_(
                1, index.unsqueeze(-1).expand(-1, -1, dim),
                patch_tokens[:, patch_index] * weights.unsqueeze(-1),
            )
            return pooled, denominators > 0
        if patch_to_roi.shape != (n_patches,):
            raise ValueError(f"patch_to_roi must have shape ({n_patches},)")
        patch_to_roi = self.dense_roi_indices(patch_to_roi, n_rois)
        valid_patch = (patch_to_roi >= 0) & (patch_to_roi < n_rois)
        roi_index = patch_to_roi.clamp(0, n_rois - 1)
        scores = self.score(patch_tokens).squeeze(-1)
        scores = scores.masked_fill(~valid_patch.unsqueeze(0), float("-inf"))
        index = roi_index.unsqueeze(0).expand(batch, -1)
        maxima = patch_tokens.new_full((batch, n_rois), float("-inf"))
        maxima.scatter_reduce_(1, index, scores, reduce="amax", include_self=True)
        shifted = scores - maxima.gather(1, index)
        exponentials = shifted.exp().masked_fill(~valid_patch.unsqueeze(0), 0)
        denominators = patch_tokens.new_zeros(batch, n_rois)
        denominators.scatter_add_(1, index, exponentials)
        weights = exponentials / denominators.gather(1, index).clamp_min(1e-12)
        pooled = patch_tokens.new_zeros(batch, n_rois, dim)
        pooled.scatter_add_(
            1, index.unsqueeze(-1).expand(-1, -1, dim),
            patch_tokens * weights.unsqueeze(-1),
        )
        return pooled, denominators > 0


ROIAttentionPool = AART


class PatchReconHead(nn.Module):
    def __init__(self, embed_dim, patch_size, img_size):
        super().__init__()
        self.patch_size = patch_size
        self.grid = tuple(size // patch_size for size in img_size)
        self.projection = nn.Linear(embed_dim, patch_size ** 3)

    def forward(self, patches):
        batch = patches.shape[0]
        gd, gh, gw = self.grid
        p = self.patch_size
        voxels = self.projection(patches).view(batch, gd, gh, gw, p, p, p)
        return voxels.permute(0, 1, 4, 2, 5, 3, 6).reshape(
            batch, 1, gd * p, gh * p, gw * p
        )


class DecoderBlock(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio, n_modalities):
        super().__init__()
        self.norm1 = CondLN(dim, n_modalities)
        self.attention = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = CondLN(dim, n_modalities)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x, modality, padding_mask=None):
        normalized = self.norm1(x, modality)
        attended = self.attention(
            normalized, normalized, normalized,
            key_padding_mask=padding_mask, need_weights=False,
        )[0]
        x = x + attended
        return x + self.mlp(self.norm2(x, modality))


class ROIMAEDecoder(nn.Module):
    """Cross-ROI decoder with atlas-specific ROI identity embeddings."""

    def __init__(self, embed_dim, atlas_n_rois, n_modalities, depth, heads, mlp_ratio):
        super().__init__()
        self.roi_identity = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(1, count, embed_dim))
            for name, count in atlas_n_rois.items()
        })
        for parameter in self.roi_identity.values():
            nn.init.trunc_normal_(parameter, std=0.02)
        self.blocks = nn.ModuleList([
            DecoderBlock(embed_dim, heads, mlp_ratio, n_modalities)
            for _ in range(depth)
        ])
        self.norm = CondLN(embed_dim, n_modalities)

    def forward(self, atlas, tokens, modality, roi_valid):
        x = tokens + self.roi_identity[atlas]
        for block in self.blocks:
            x = block(x, modality, ~roi_valid)
        return self.norm(x, modality)


class OmniMIRAHeads(nn.Module):
    def __init__(self, embed_dim, head_specs):
        super().__init__()
        self.names = []
        for spec in head_specs:
            if spec.family not in HEAD_FAMILIES:
                continue
            self.names.append(spec.name)
            self.add_module(spec.name, HEAD_FAMILIES[spec.family](embed_dim, spec))
        for name in self.names:
            if name == "cross_modal":
                nn.init.orthogonal_(getattr(self, name).weight)

    def forward(self, tokens):
        result = {}
        for name in self.names:
            value = getattr(self, name)(tokens)
            result[name] = F.normalize(value, dim=-1) if name == "cross_modal" else value
        return result


class OmniMIRA(nn.Module):
    public_component_names = PUBLIC_COMPONENT_NAMES
    active_head_names = ACTIVE_HEAD_NAMES

    def __init__(
        self, img_size: Tuple[int, int, int] = (96, 112, 96), patch_size: int = 8,
        embed_dim: int = 128, mlp_depth: int = 4, mlp_ratio: float = 4.0,
        n_modalities: int = 4, n_rois: int = 166, contrastive_dim: int = 64,
        mask_ratio: float = 0.9, decoder_depth: int = 2, decoder_heads: int = 4,
        head_specs: Optional[List[HeadSpec]] = None,
        atlas_n_rois: Optional[Dict[str, int]] = None,
        pool_kind: str = "attention", pool_k: int = 1,
    ):
        super().__init__()
        self.img_size = tuple(img_size)
        self.patch_size = patch_size
        self.patch_grid = tuple(size // patch_size for size in img_size)
        self.embed_dim = embed_dim
        self.n_modalities = n_modalities
        self.n_rois = n_rois
        self.mask_ratio = mask_ratio
        self.atlas_n_rois = atlas_n_rois or {"aal3": 166, "ho69": 69, "yeo7": 7}
        specs = head_specs or [HeadSpec(*item) for item in RELEASE_HEAD_SPECS]
        names = tuple(spec.name for spec in specs)
        if names != ACTIVE_HEAD_NAMES:
            raise ValueError(f"release head names must be {ACTIVE_HEAD_NAMES}, got {names}")
        if pool_kind != "attention" or pool_k != 1:
            raise ValueError("paper-defined AART uses one scalar attention score")

        self.patch_embed = ModalityPatchEmbed3D(
            img_size, patch_size, embed_dim, n_modalities
        )
        self.n_patches = self.patch_embed.n_patches
        position = build_3d_sincos_pos_embed(embed_dim, self.patch_grid)
        self.register_buffer("pos_embed", torch.from_numpy(position).float().unsqueeze(0))
        self.blocks = nn.ModuleList([
            MLPBlockCond(embed_dim, mlp_ratio, n_modalities) for _ in range(mlp_depth)
        ])
        self.norm = CondLN(embed_dim, n_modalities)
        self.attn_pool = AART(embed_dim)
        self.heads = OmniMIRAHeads(embed_dim, specs)
        self.patch_recon_head = PatchReconHead(embed_dim, patch_size, img_size)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.roi_mae_decoder = ROIMAEDecoder(
            embed_dim, self.atlas_n_rois, n_modalities, decoder_depth,
            decoder_heads, mlp_ratio,
        )
    def _validate_modality(self, x, modality):
        if not isinstance(modality, torch.Tensor):
            raise TypeError("modality must be a torch.Tensor")
        if modality.shape != (x.shape[0],):
            raise ValueError(f"modality must have shape ({x.shape[0]},)")
        if modality.dtype not in _INTEGER_DTYPES:
            raise TypeError("modality must use an integer dtype")
        invalid = (modality < 0) | (modality >= self.n_modalities)
        if invalid.any():
            raise ValueError(f"invalid modality rows {torch.where(invalid)[0].tolist()}")
        return modality.long()

    def _encode(self, x, modality):
        patches = self.patch_embed(x, modality) + self.pos_embed
        for block in self.blocks:
            patches = block(patches, modality)
        return self.norm(patches, modality)

    @staticmethod
    def _atlas_maps(patch_to_roi):
        return patch_to_roi if isinstance(patch_to_roi, dict) else {"default": patch_to_roi}

    def encode_roi_features(self, x, modality, patch_to_roi):
        modality = self._validate_modality(x, modality)
        patches = self._encode(x, modality)
        atlases = {}
        for name, mapping in self._atlas_maps(patch_to_roi).items():
            count = self.atlas_n_rois.get(name, self.n_rois)
            tokens, valid = self.attn_pool(patches, mapping, count)
            atlases[name] = {
                "roi_tokens": tokens, "roi_valid": valid, "patch_to_roi": mapping
            }
        primary = next(iter(atlases))
        return {
            "patches": patches, "atlases": atlases, "primary": primary,
            "roi_tokens": atlases[primary]["roi_tokens"],
            "roi_valid": atlases[primary]["roi_valid"],
        }

    def _sample_roi_mask(self, valid):
        masks = torch.zeros_like(valid)
        random = torch.rand(valid.shape, device=valid.device)
        for row in range(valid.shape[0]):
            indices = torch.where(valid[row])[0]
            count = int(round(indices.numel() * self.mask_ratio))
            if count:
                selected = indices[torch.argsort(random[row, indices])[:count]]
                masks[row, selected] = True
        return masks

    def forward(self, x, modality, patch_to_roi):
        modality = self._validate_modality(x, modality)
        encoded = self.encode_roi_features(x, modality, patch_to_roi)
        for name, output in encoded["atlases"].items():
            clean = output["roi_tokens"]
            mask = self._sample_roi_mask(output["roi_valid"])
            masked = torch.where(mask.unsqueeze(-1), self.mask_token, clean)
            output.update({
                "head_proj": self.heads(clean),
                "roi_mask": mask,
                "roi_mae_pred": self.roi_mae_decoder(
                    name, masked, modality, output["roi_valid"]
                ),
                "roi_mae_target": clean.detach(),
            })
        primary = encoded["primary"]
        patch_recon = self.patch_recon_head(encoded["patches"])
        encoded["atlases"][primary]["patch_recon"] = patch_recon
        encoded["patch_recon"] = patch_recon
        encoded["head_proj"] = encoded["atlases"][primary]["head_proj"]
        encoded["roi_mask"] = encoded["atlases"][primary]["roi_mask"]
        return encoded
