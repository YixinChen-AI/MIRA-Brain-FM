"""External effectiveness audit for a frozen OmniMIRA checkpoint."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from omnimira.atlas import ATLAS_ORDER, fixed_patch_to_roi
from omnimira.inference import (
    _build_omnimira_for_tests,
    load_omnimira,
)
from omnimira.io import _brain_mask, _normalize_pet, _pet_reference_mask


def build_entries(labels_csv: Path, npy_dir: Path) -> tuple[list[dict], list[str]]:
    entries = []
    excluded = []
    reference = _pet_reference_mask()
    with labels_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            scan_id = row["clean_id"]
            path = npy_dir / f"{scan_id}.npy"
            if not path.is_file():
                excluded.append(scan_id)
                continue
            raw = np.load(path, mmap_mode="r", allow_pickle=False)
            reference_values = np.asarray(raw[reference])
            if (
                reference_values.size == 0
                or not np.isfinite(reference_values).all()
                or float(reference_values.mean()) <= 1e-8
            ):
                excluded.append(scan_id)
                continue
            entries.append({
                "path": str(path),
                "subject_id": scan_id,
                "age": float(row["age"]),
            })
    if len(entries) < 100:
        raise ValueError(f"expected at least 100 external scans, found {len(entries)}")
    return entries, excluded


class ExternalFDGDataset(Dataset):
    """Normalize legacy model-grid arrays without claiming public sidecar compliance."""

    def __init__(self, entries):
        self.entries = entries
        self.mask = _brain_mask("fdg")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]
        raw = np.load(entry["path"], allow_pickle=False).astype(np.float32, copy=False)
        if raw.shape != (96, 112, 96):
            raise ValueError(f"unexpected external array shape: {raw.shape}")
        normalized = _normalize_pet(raw, self.mask)
        return {
            "x": torch.from_numpy(normalized).unsqueeze(0),
            "modality_id": torch.tensor(2, dtype=torch.long),
            "age": torch.tensor(entry["age"], dtype=torch.float32),
        }


@torch.no_grad()
def extract(model, loader, mappings, device):
    feature_rows = []
    aal3_rows = []
    direct_age = []
    ages = []
    model.eval()
    for batch in loader:
        x = batch["x"].to(device)
        modality = batch["modality_id"].to(device)
        output = model.encode_roi_features(x, modality, mappings)
        pooled = []
        predictions = []
        for name in ATLAS_ORDER:
            atlas = output["atlases"][name]
            valid = atlas["roi_valid"].float()
            tokens = atlas["roi_tokens"]
            denominator = valid.sum(1, keepdim=True).clamp_min(1.0)
            pooled.append((tokens * valid.unsqueeze(-1)).sum(1) / denominator)
            age_roi = model.heads(tokens)["age"].squeeze(-1)
            predictions.append((age_roi * valid).sum(1) / denominator.squeeze(1))
            if name == "aal3":
                aal3_rows.append(tokens.cpu().float().numpy())
        feature_rows.append(torch.cat(pooled, dim=1).cpu().float().numpy())
        direct_age.append(torch.stack(predictions, dim=1).mean(1).cpu().float().numpy())
        ages.append(batch["age"].numpy())
    return (
        np.concatenate(feature_rows),
        np.concatenate(aal3_rows),
        np.concatenate(direct_age),
        np.concatenate(ages),
    )


def effective_rank(features: np.ndarray) -> float:
    centered = features - features.mean(0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    variance = singular ** 2
    probability = variance / variance.sum()
    return float(np.exp(-(probability * np.log(probability + 1e-12)).sum()))


def ridge_cv(features, ages, splits):
    predictions = np.full((len(splits), len(ages)), np.nan, dtype=np.float64)
    for fold, (train, test) in enumerate(splits):
        model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        model.fit(features[train], ages[train])
        predictions[fold, test] = model.predict(features[test])
    valid = np.isfinite(predictions)
    return {
        "mae": float(np.abs(predictions[valid] - np.broadcast_to(ages, predictions.shape)[valid]).mean()),
        "pearson_r": float(np.corrcoef(
            predictions[valid],
            np.broadcast_to(ages, predictions.shape)[valid],
        )[0, 1]),
    }


def per_roi_ridge_cv(features, ages, splits):
    predictions = np.full((len(splits), len(ages)), np.nan, dtype=np.float64)
    for fold, (train, test) in enumerate(splits):
        roi_predictions = []
        for roi in range(features.shape[1]):
            model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
            model.fit(features[train, roi], ages[train])
            roi_predictions.append(model.predict(features[test, roi]))
        predictions[fold, test] = np.mean(roi_predictions, axis=0)
    valid = np.isfinite(predictions)
    targets = np.broadcast_to(ages, predictions.shape)
    return {
        "mae": float(np.abs(predictions[valid] - targets[valid]).mean()),
        "pearson_r": float(np.corrcoef(predictions[valid], targets[valid])[0, 1]),
    }


def summarize(features, roi_features, direct_age, ages, splits):
    roi_residual = roi_features - roi_features.mean(0, keepdims=True)
    return {
        "feature_finite": bool(np.isfinite(features).all()),
        "mean_feature_sd": float(features.std(0).mean()),
        "global_mean_effective_rank": effective_rank(features),
        "mean_roi_feature_sd": float(roi_features.std(0).mean()),
        "roi_subject_effective_rank": effective_rank(
            roi_residual.reshape(roi_residual.shape[0], -1)
        ),
        "direct_age_mae": float(mean_absolute_error(ages, direct_age)),
        "direct_age_pearson_r": float(np.corrcoef(ages, direct_age)[0, 1]),
        "frozen_feature_ridge": ridge_cv(features, ages, splits),
        "per_roi_ridge": per_roi_ridge_cv(roi_features, ages, splits),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--npy-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    entries, excluded = build_entries(
        Path(args.labels).expanduser(), Path(args.npy_dir).expanduser()
    )
    dataset = ExternalFDGDataset(entries)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4,
    )
    mappings = {
        name: torch.from_numpy(value.copy()).bool().to(device)
        for name, value in fixed_patch_to_roi().items()
    }

    trained = load_omnimira(args.checkpoint, args.config, device=str(device))
    trained_features, trained_roi, trained_age, ages = extract(
        trained, loader, mappings, device
    )
    del trained
    torch.cuda.empty_cache()

    torch.manual_seed(args.seed)
    random_model = _build_omnimira_for_tests(args.config, device=str(device))
    random_features, random_roi, random_age, random_ages = extract(
        random_model, loader, mappings, device
    )
    if not np.array_equal(ages, random_ages):
        raise RuntimeError("trained and random evaluations used different subject order")

    splitter = RepeatedKFold(n_splits=5, n_repeats=5, random_state=args.seed)
    splits = list(splitter.split(np.arange(len(ages))))
    checkpoint_epoch = int(torch.load(
        Path(args.checkpoint).expanduser(), map_location="cpu", weights_only=False
    ).get("epoch", -1))
    report = {
        "checkpoint": str(Path(args.checkpoint).expanduser()),
        "checkpoint_epoch": checkpoint_epoch,
        "external_cohort": "BJFDG",
        "n_scans": len(ages),
        "excluded_input_contract_failures": len(excluded),
        "age_range": [float(ages.min()), float(ages.max())],
        "trained": summarize(
            trained_features, trained_roi, trained_age, ages, splits
        ),
        "random_init": summarize(
            random_features, random_roi, random_age, ages, splits
        ),
    }
    report["trained_minus_random"] = {
        "ridge_mae": (
            report["trained"]["frozen_feature_ridge"]["mae"]
            - report["random_init"]["frozen_feature_ridge"]["mae"]
        ),
        "ridge_pearson_r": (
            report["trained"]["frozen_feature_ridge"]["pearson_r"]
            - report["random_init"]["frozen_feature_ridge"]["pearson_r"]
        ),
        "per_roi_ridge_mae": (
            report["trained"]["per_roi_ridge"]["mae"]
            - report["random_init"]["per_roi_ridge"]["mae"]
        ),
        "per_roi_ridge_pearson_r": (
            report["trained"]["per_roi_ridge"]["pearson_r"]
            - report["random_init"]["per_roi_ridge"]["pearson_r"]
        ),
    }
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
