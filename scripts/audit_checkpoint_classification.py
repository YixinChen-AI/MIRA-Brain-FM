"""Audit frozen OmniMIRA features on a subject-disjoint disease task."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from omnimira.atlas import ATLAS_ORDER, fixed_patch_to_roi
from omnimira.inference import _build_omnimira_for_tests, load_omnimira
from omnimira.io import _brain_mask, _normalize_pet, _pet_reference_mask


def _read_subjects(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def build_entries(
    labels_csv: Path,
    npy_dir: Path,
    train_subjects: set[str],
    test_subjects: set[str],
) -> tuple[list[dict], list[dict], list[str]]:
    train, test, excluded = [], [], []
    reference = _pet_reference_mask()
    with labels_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (
                row["dataset"] != "adni"
                or row["modality"] != "fdg"
                or row["dx"] not in {"CN", "AD"}
            ):
                continue
            path = npy_dir / row["npy_basename"]
            if not path.is_file():
                excluded.append(row["npy_basename"])
                continue
            raw = np.load(path, mmap_mode="r", allow_pickle=False)
            values = np.asarray(raw[reference])
            if (
                raw.shape != (96, 112, 96)
                or values.size == 0
                or not np.isfinite(values).all()
                or float(values.mean()) <= 1e-8
            ):
                excluded.append(row["npy_basename"])
                continue
            entry = {
                "path": str(path),
                "subject_id": row["subject_id"],
                "label": int(row["dx"] == "AD"),
            }
            if row["subject_id"] in train_subjects:
                train.append(entry)
            elif row["subject_id"] in test_subjects:
                test.append(entry)

    for name, entries in (("train", train), ("test", test)):
        subjects = [entry["subject_id"] for entry in entries]
        if len(subjects) != len(set(subjects)):
            raise ValueError(f"{name} set contains repeated FDG scans for a subject")
        labels = np.asarray([entry["label"] for entry in entries])
        if len(np.unique(labels)) != 2:
            raise ValueError(f"{name} set does not contain both classes")
    if {entry["subject_id"] for entry in train} & {entry["subject_id"] for entry in test}:
        raise ValueError("train and test subjects overlap")
    return train, test, excluded


class FDGDataset(Dataset):
    def __init__(self, entries):
        self.entries = entries
        self.mask = _brain_mask("fdg")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]
        raw = np.load(entry["path"], allow_pickle=False).astype(np.float32, copy=False)
        normalized = _normalize_pet(raw, self.mask)
        patch_mean = normalized.reshape(12, 8, 14, 8, 12, 8).mean((1, 3, 5))
        return {
            "x": torch.from_numpy(normalized).unsqueeze(0),
            "patch_mean": torch.from_numpy(patch_mean.reshape(-1)),
            "label": torch.tensor(entry["label"], dtype=torch.long),
        }


@torch.no_grad()
def extract(model, loader, mappings, device):
    roi_rows, pooled_rows, intensity_rows, labels = [], [], [], []
    model.eval()
    for batch in loader:
        x = batch["x"].to(device)
        modality = torch.full((x.shape[0],), 2, dtype=torch.long, device=device)
        output = model.encode_roi_features(x, modality, mappings)
        roi_parts, pooled_parts = [], []
        for name in ATLAS_ORDER:
            atlas = output["atlases"][name]
            tokens = atlas["roi_tokens"]
            valid = atlas["roi_valid"].float()
            roi_parts.append(tokens.reshape(tokens.shape[0], -1))
            denominator = valid.sum(1, keepdim=True).clamp_min(1.0)
            pooled_parts.append(
                (tokens * valid.unsqueeze(-1)).sum(1) / denominator
            )
        roi_rows.append(torch.cat(roi_parts, dim=1).cpu().float().numpy())
        pooled_rows.append(torch.cat(pooled_parts, dim=1).cpu().float().numpy())
        intensity_rows.append(batch["patch_mean"].numpy())
        labels.append(batch["label"].numpy())
    return {
        "roi_flat": np.concatenate(roi_rows),
        "atlas_pooled": np.concatenate(pooled_rows),
        "patch_intensity": np.concatenate(intensity_rows),
        "labels": np.concatenate(labels),
    }


def effective_rank(features: np.ndarray) -> float:
    centered = features - features.mean(0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    variance = singular ** 2
    probability = variance / variance.sum()
    return float(np.exp(-(probability * np.log(probability + 1e-12)).sum()))


def _classifier(c_value: float):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c_value,
            max_iter=3000,
            solver="liblinear",
            class_weight="balanced",
            random_state=42,
        ),
    )


def select_c(features, labels, candidates):
    n_splits = min(5, int(np.bincount(labels).min()))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = {}
    for c_value in candidates:
        fold_scores = []
        for train, valid in splitter.split(features, labels):
            model = _classifier(c_value)
            model.fit(features[train], labels[train])
            probability = model.predict_proba(features[valid])[:, 1]
            fold_scores.append(roc_auc_score(labels[valid], probability))
        scores[str(c_value)] = [float(value) for value in fold_scores]
    selected = max(
        candidates,
        key=lambda value: (np.mean(scores[str(value)]), -value),
    )
    return float(selected), scores


def bootstrap_auc(labels, probability, seed=42, repetitions=2000):
    rng = np.random.default_rng(seed)
    by_class = [np.flatnonzero(labels == value) for value in (0, 1)]
    values = []
    for _ in range(repetitions):
        sample = np.concatenate([
            rng.choice(indices, size=len(indices), replace=True)
            for indices in by_class
        ])
        values.append(roc_auc_score(labels[sample], probability[sample]))
    return [float(value) for value in np.percentile(values, [2.5, 97.5])]


def evaluate(train_x, train_y, test_x, test_y, permutation_repetitions=200):
    candidates = [0.0001, 0.001, 0.01, 0.1, 1.0]
    selected_c, cv_scores = select_c(train_x, train_y, candidates)
    model = _classifier(selected_c)
    model.fit(train_x, train_y)
    probability = model.predict_proba(test_x)[:, 1]
    prediction = (probability >= 0.5).astype(int)

    rng = np.random.default_rng(2026)
    null_auc = []
    for _ in range(permutation_repetitions):
        shuffled = rng.permutation(train_y)
        null_model = _classifier(selected_c)
        null_model.fit(train_x, shuffled)
        null_auc.append(roc_auc_score(test_y, null_model.predict_proba(test_x)[:, 1]))

    return {
        "selected_c": selected_c,
        "inner_cv_auc": {
            key: {
                "folds": values,
                "mean": float(np.mean(values)),
            }
            for key, values in cv_scores.items()
        },
        "test_auc": float(roc_auc_score(test_y, probability)),
        "test_auc_95_ci": bootstrap_auc(test_y, probability),
        "test_accuracy": float(accuracy_score(test_y, prediction)),
        "test_balanced_accuracy": float(balanced_accuracy_score(test_y, prediction)),
        "permutation_auc_95th": float(np.percentile(null_auc, 95)),
        "permutation_p": float((1 + np.sum(np.asarray(null_auc) >= roc_auc_score(
            test_y, probability
        ))) / (1 + len(null_auc))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--npy-dir", required=True)
    parser.add_argument("--train-subjects", required=True)
    parser.add_argument("--test-subjects", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device(args.device)
    train_entries, test_entries, excluded = build_entries(
        Path(args.labels).expanduser(),
        Path(args.npy_dir).expanduser(),
        _read_subjects(Path(args.train_subjects).expanduser()),
        _read_subjects(Path(args.test_subjects).expanduser()),
    )
    train_loader = DataLoader(
        FDGDataset(train_entries), batch_size=args.batch_size,
        shuffle=False, num_workers=4,
    )
    test_loader = DataLoader(
        FDGDataset(test_entries), batch_size=args.batch_size,
        shuffle=False, num_workers=4,
    )
    mappings = {
        name: torch.from_numpy(value.copy()).long().to(device)
        for name, value in fixed_patch_to_roi().items()
    }

    trained_model = load_omnimira(args.checkpoint, args.config, device=str(device))
    trained_train = extract(trained_model, train_loader, mappings, device)
    trained_test = extract(trained_model, test_loader, mappings, device)
    del trained_model
    torch.cuda.empty_cache()

    torch.manual_seed(42)
    random_model = _build_omnimira_for_tests(args.config, device=str(device))
    random_train = extract(random_model, train_loader, mappings, device)
    random_test = extract(random_model, test_loader, mappings, device)

    train_y = trained_train.pop("labels")
    test_y = trained_test.pop("labels")
    random_train.pop("labels")
    random_test.pop("labels")
    report = {
        "checkpoint": str(Path(args.checkpoint).expanduser()),
        "task": "ADNI FDG CN vs AD",
        "protocol": "fixed subject-disjoint train/test; C selected by train-only 5-fold CV",
        "n_train": int(len(train_y)),
        "n_test": int(len(test_y)),
        "train_class_counts": np.bincount(train_y, minlength=2).tolist(),
        "test_class_counts": np.bincount(test_y, minlength=2).tolist(),
        "excluded_input_contract_failures": len(excluded),
        "trained_roi_flat": evaluate(
            trained_train["roi_flat"], train_y,
            trained_test["roi_flat"], test_y,
        ),
        "trained_atlas_pooled": evaluate(
            trained_train["atlas_pooled"], train_y,
            trained_test["atlas_pooled"], test_y,
        ),
        "random_roi_flat": evaluate(
            random_train["roi_flat"], train_y,
            random_test["roi_flat"], test_y,
        ),
        "patch_intensity": evaluate(
            trained_train["patch_intensity"], train_y,
            trained_test["patch_intensity"], test_y,
        ),
        "feature_diagnostics": {
            "trained_roi_effective_rank": effective_rank(trained_train["roi_flat"]),
            "trained_pooled_effective_rank": effective_rank(
                trained_train["atlas_pooled"]
            ),
            "random_roi_effective_rank": effective_rank(random_train["roi_flat"]),
        },
    }
    trained_auc = report["trained_roi_flat"]["test_auc"]
    random_auc = report["random_roi_flat"]["test_auc"]
    permutation_limit = report["trained_roi_flat"]["permutation_auc_95th"]
    report["pass_criteria"] = {
        "trained_auc_at_least_0.65": bool(trained_auc >= 0.65),
        "trained_exceeds_random_by_0.05": bool(trained_auc >= random_auc + 0.05),
        "trained_exceeds_permutation_95th": bool(trained_auc > permutation_limit),
    }
    report["passed"] = bool(all(report["pass_criteria"].values()))

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
