"""Single-node training entry for the paper-defined public OmniMIRA core."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import nibabel as nib
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Sampler

from omnimira.dataset import (
    MODALITY_ID,
    ManifestDataset,
    build_atlas_patch_to_roi,
    collate,
)
from omnimira.atlas import atlas_path
from omnimira.losses import OmniMIRALoss
from omnimira.models.omnimira import HeadSpec, OmniMIRA
from omnimira.schema import file_sha256, verified_resource


SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
from validate_pretraining_manifest import validate_entries


def learning_rate_at_epoch(epoch, total_epochs, warmup_epochs,
                           warmup_start_lr, base_lr, min_lr):
    if epoch <= warmup_epochs:
        fraction = epoch / max(warmup_epochs, 1)
        return warmup_start_lr + fraction * (base_lr - warmup_start_lr)
    progress = min(1.0, (epoch - warmup_epochs) /
                   max(total_epochs - warmup_epochs, 1))
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def training_provenance(manifest: Path, config: Path, entries: list[dict]) -> dict:
    """Record immutable recipe identities without exposing data locations."""
    modalities = Counter(str(entry["modality"]) for entry in entries)
    participants = {str(entry["subject_id"]) for entry in entries}
    cohorts = {
        str(entry["cohort"]) for entry in entries
        if entry.get("cohort") is not None and str(entry["cohort"]).strip()
    }
    return {
        "manifest_sha256": file_sha256(manifest),
        "manifest_entries": len(entries),
        "manifest_modalities": dict(sorted(modalities.items())),
        "manifest_unique_participants": len(participants),
        "manifest_cohort_count": len(cohorts),
        "config_sha256": file_sha256(config),
        "config_path": config.name,
    }


def augment_batch(x, modality, brain_mask, *, seed, epoch, view,
                  t1_sigma=0.02, pet_jitter=0.05):
    """Draw an epoch-specific view reproducibly from seed and sampler position."""
    generator = torch.Generator(device=x.device)
    generator.manual_seed(int(seed) + 1_000_003 * int(epoch) + 10_007 * int(view))
    result = x.clone()
    t1 = modality == MODALITY_ID["t1"]
    if t1.any():
        noise = torch.randn(result[t1].shape, generator=generator,
                            device=x.device, dtype=x.dtype) * t1_sigma
        result[t1] = result[t1] + noise
    pet = (modality == MODALITY_ID["av45"]) | (modality == MODALITY_ID["fdg"])
    if pet.any():
        factors = 1 + (torch.rand((int(pet.sum()), 1, 1, 1, 1),
                                  generator=generator, device=x.device,
                                  dtype=x.dtype) * 2 - 1) * pet_jitter
        selected = result[pet]
        selected_mask = brain_mask[pet]
        result[pet] = torch.where(selected_mask, selected * factors, selected)
    return result


class V4BatchSampler(Sampler):
    """Balanced 16-per-modality batches with at least 32 participants."""

    def __init__(self, entries, n_per_modality, min_unique_subjects, seed,
                 num_batches=None):
        self.entries = entries
        self.n_per_modality = n_per_modality
        self.min_unique_subjects = min_unique_subjects
        self.seed = seed
        self.num_batches = num_batches
        self.epoch = 0
        self.groups = {
            name: [i for i, entry in enumerate(entries) if entry["modality"] == name]
            for name in MODALITY_ID
        }
        if any(not indices for indices in self.groups.values()):
            raise ValueError("V4 sampler requires all four modalities")

    def __len__(self):
        natural_batches = max(1, len(self.entries) // (4 * self.n_per_modality))
        return self.num_batches or natural_batches

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        for _ in range(len(self)):
            for _attempt in range(100):
                batch = []
                for indices in self.groups.values():
                    picks = torch.randint(len(indices), (self.n_per_modality,),
                                          generator=generator)
                    batch.extend(indices[index] for index in picks.tolist())
                participants = {
                    self.entries[index].get("subject_id", index) for index in batch
                }
                if len(participants) >= self.min_unique_subjects:
                    order = torch.randperm(len(batch), generator=generator).tolist()
                    yield [batch[index] for index in order]
                    break
            else:
                raise ValueError("unable to construct a batch with 32 participants")


def build_model(cfg):
    specs = [HeadSpec(**item) for item in cfg["heads"]]
    atlases = {item["name"]: int(item["n_rois"]) for item in cfg["atlases"]}
    pool = cfg["model"]["pool"]
    return OmniMIRA(
        img_size=tuple(cfg["model"]["img_size"]),
        patch_size=cfg["model"]["patch_size"],
        embed_dim=cfg["model"]["embed_dim"],
        mlp_depth=cfg["model"]["mlp_depth"],
        mlp_ratio=cfg["model"]["mlp_ratio"],
        n_modalities=cfg["model"]["n_modalities"],
        n_rois=cfg["model"]["n_rois"],
        contrastive_dim=cfg["model"]["contrastive_dim"],
        mask_ratio=cfg["model"]["mask_ratio"],
        decoder_depth=cfg["model"]["decoder_depth"],
        decoder_heads=cfg["model"]["decoder_heads"],
        head_specs=specs, atlas_n_rois=atlases,
        pool_kind=pool["kind"], pool_k=pool["k"],
    )


def assemble_atlas_outputs(output):
    result = {}
    for name, atlas in output["atlases"].items():
        result[name] = dict(atlas)
    return result


def _loader(dataset, cfg, requested_batch_size, generator, steps_per_epoch=0):
    recipe = cfg["sampler"]
    if requested_batch_size == recipe["batch_size"]:
        sampler = V4BatchSampler(
            dataset.entries, recipe["n_per_modality"],
            recipe["min_unique_subjects"], cfg["system"]["seed"],
            num_batches=steps_per_epoch or None,
        )
        return DataLoader(dataset, batch_sampler=sampler, collate_fn=collate,
                          num_workers=0), sampler
    return DataLoader(dataset, batch_size=requested_batch_size, shuffle=True,
                      generator=generator, collate_fn=collate,
                      num_workers=0), None


def loss_audit_record(result, task_keys, *, epoch, step):
    sigma = result["log_sigma2"].detach().float().cpu()
    sigma_by_key = dict(zip(task_keys, sigma.tolist()))
    raw = {
        key: float(value.detach().float().cpu())
        for key, value in result["per_head"].items()
    }
    effective = {
        key: float(value.detach().float().cpu())
        for key, value in result["per_head_effective"].items()
    }
    contribution = {
        key: effective[key] + 0.5 * sigma_by_key[key]
        for key in effective
    }
    total = float(result["total"].detach().float().cpu())
    denominator = sum(abs(value) for value in contribution.values()) or 1.0
    return {
        "event": "loss_audit",
        "epoch": epoch,
        "step": step,
        "total": total,
        "raw": raw,
        "effective": effective,
        "huw_contribution": contribution,
        "absolute_contribution_fraction": {
            key: abs(value) / denominator for key, value in contribution.items()
        },
        "log_sigma2": sigma_by_key,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(REPO / "configs/omnimira_3atlas.yaml"))
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", default="runs/omnimira_pretrain")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--steps-per-epoch", type=int, default=0)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--resume")
    parser.add_argument(
        "--audit-only", action="store_true",
        help="run one checkpoint batch, print all loss components, and exit without updating",
    )
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument(
        "--preflight", action="store_true",
        help="validate the recipe, manifest, atlases and model without reading training images",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    cfg = yaml.safe_load(config_path.read_text())
    seed = int(cfg["system"]["seed"])
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    device = torch.device(args.device)
    image_size = tuple(cfg["model"]["img_size"])
    patch_size = cfg["model"]["patch_size"]
    try:
        entries = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        manifest_summary = validate_entries(entries, cfg)
    except (OSError, ValueError, json.JSONDecodeError, KeyError) as exc:
        raise SystemExit(f"manifest validation failed: {exc}") from exc
    provenance = training_provenance(Path(args.manifest).expanduser(), config_path, entries)
    atlas_files = {item["name"]: str(atlas_path(item["name"])) for item in cfg["atlases"]}
    mappings = {name: value.to(device) for name, value in
                build_atlas_patch_to_roi(atlas_files, image_size, patch_size).items()}
    model = build_model(cfg).to(device)
    loss_fn = OmniMIRALoss(
        tau=cfg["loss"]["tau"], patch_size=patch_size,
        head_names=tuple(item["name"] for item in cfg["heads"]),
        atlas_names=tuple(atlas_files),
        use_uncertainty_weighting=cfg["loss"]["use_uncertainty_weighting"],
        supported_objectives={
            item["name"]: tuple(item["objectives"]) for item in cfg["atlases"]
        },
        log_sigma_clamp=tuple(cfg["loss"].get("log_sigma_clamp", (-5.0, 5.0))),
    ).to(device)
    if args.preflight:
        print(json.dumps({
            "status": "preflight_ok",
            "manifest": manifest_summary,
            "atlas_names": list(atlas_files),
            "model_version": "omnimira-public-v1",
            "training_provenance": provenance,
        }, sort_keys=True))
        return

    dataset = ManifestDataset(entries, atlas_files, image_size)
    batch_size = args.batch_size or cfg["sampler"]["batch_size"]
    loader, batch_sampler = _loader(
        dataset, cfg, batch_size, generator, args.steps_per_epoch
    )
    parameters = [*model.parameters(), *loss_fn.parameters()]
    optimizer = torch.optim.AdamW(
        parameters, lr=args.lr or cfg["optim"]["lr"],
        betas=tuple(cfg["optim"]["betas"]),
        weight_decay=cfg["optim"]["weight_decay"],
    )
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        loss_fn.load_state_dict(checkpoint["loss"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        generator.set_state(checkpoint["sampler_rng_state"].cpu())
        if device.type == "cuda" and checkpoint.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state(checkpoint["cuda_rng_state"].cpu(), device)
        start_epoch = checkpoint["epoch"] + 1

    epochs = args.epochs if args.epochs is not None else cfg["optim"]["total_epochs"]
    mask_array = torch.from_numpy(
        np.asarray(nib.load(verified_resource("model_brain_mask")).dataobj, dtype=np.uint8)
    ).bool().to(device)
    primary_mask = mask_array.unsqueeze(0).unsqueeze(0)
    foreground_patch_mask = mask_array.reshape(
        model.patch_grid[0], patch_size,
        model.patch_grid[1], patch_size,
        model.patch_grid[2], patch_size,
    ).permute(0, 2, 4, 1, 3, 5).reshape(model.n_patches, -1).any(dim=1)
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(start_epoch, epochs):
        if batch_sampler is not None:
            batch_sampler.set_epoch(epoch)
        lr = args.lr or learning_rate_at_epoch(
            epoch, cfg["optim"]["total_epochs"], cfg["optim"]["warmup_epochs"],
            cfg["optim"]["warmup_start_lr"], cfg["optim"]["lr"],
            cfg["optim"]["min_lr"],
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        for step, batch in enumerate(loader):
            if args.steps_per_epoch and step >= args.steps_per_epoch:
                break
            x = batch["x"].to(device)
            modality = batch["modality_id"].to(device)
            mask = primary_mask.expand(x.shape[0], -1, -1, -1, -1)
            view1 = augment_batch(x, modality, mask, seed=seed, epoch=epoch, view=2 * step)
            view2 = augment_batch(x, modality, mask, seed=seed, epoch=epoch, view=2 * step + 1)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=use_bf16):
                output1 = model(view1, modality, mappings)
                output2 = model.encode_roi_features(view2, modality, mappings)
                meta = {
                    "modality_id": modality,
                    "age": batch["age"].to(device),
                    "age_valid": batch["age_valid"].to(device),
                    "foreground_patch_mask": foreground_patch_mask,
                    "roi_volumes": {
                        name: value.to(device) for name, value in batch["roi_volumes"].items()
                    },
                    "roi_volume_valid": {
                        name: value.to(device)
                        for name, value in batch["roi_volume_valid"].items()
                    },
                    "roi_distances": {
                        name: value.to(device) for name, value in batch["roi_distances"].items()
                    },
                    "spatial_valid": {
                        name: value.to(device) for name, value in batch["spatial_valid"].items()
                    },
                    "aug_view2_roi": {
                        name: atlas["roi_tokens"]
                        for name, atlas in output2["atlases"].items()
                    },
                }
                result = loss_fn(assemble_atlas_outputs(output1), meta=meta,
                                 recon_target=view1)
            if not torch.isfinite(result["total"]):
                raise FloatingPointError(
                    f"non-finite loss at epoch {epoch}, step {step}: "
                    f"{result['total'].item()}"
                )
            if args.audit_only or step % args.log_every == 0:
                print(json.dumps(
                    loss_audit_record(
                        result, loss_fn.task_keys, epoch=epoch, step=step
                    ),
                    sort_keys=True,
                ), flush=True)
            if args.audit_only:
                return
            optimizer.zero_grad(set_to_none=True)
            result["total"].backward()
            torch.nn.utils.clip_grad_norm_(
                parameters, max_norm=float("inf"), error_if_nonfinite=True
            )
            optimizer.step()
            loss_fn.clamp_log_sigma2_()
            print(f"epoch {epoch} step {step} loss {result['total'].item():.4f}", flush=True)

        torch.save({
            "format_version": 1,
            "model_version": "omnimira-public-v1",
            "model": model.state_dict(), "loss": loss_fn.state_dict(),
            "optimizer": optimizer.state_dict(), "epoch": epoch,
            "config": args.config, "training_provenance": provenance,
            "torch_rng_state": torch.get_rng_state(),
            "sampler_rng_state": generator.get_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state(device) if device.type == "cuda" else None
            ),
        }, Path(args.out_dir) / "omnimira_pretrain.pt")
    print(f"saved checkpoint -> {Path(args.out_dir) / 'omnimira_pretrain.pt'}", flush=True)


if __name__ == "__main__":
    main()
