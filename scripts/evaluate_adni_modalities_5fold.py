"""Evaluate the current GitHub checkpoint on ADNI modalities."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from evaluate_roi_ensemble_nested import evaluate_full_cohort_fivefold
from omnimira.atlas import ATLAS_ORDER, fixed_patch_to_roi
from omnimira.inference import load_omnimira
from omnimira.io import (
    _brain_mask,
    _normalize_pet,
    _normalize_t1,
    _pet_reference_mask,
)


MODALITY_ID = {"t1": 0, "av45": 1, "fdg": 2, "ct": 3}


class TaskDataset(Dataset):
    def __init__(self, entries, modality):
        self.entries = entries
        self.modality = modality
        self.mask = _brain_mask(modality)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]
        raw = np.load(entry["path"], allow_pickle=False).astype(np.float32, copy=False)
        if self.modality == "t1":
            normalized = _normalize_t1(raw, self.mask)
        else:
            normalized = _normalize_pet(raw, self.mask)
        return {
            "x": torch.from_numpy(normalized).unsqueeze(0),
            "label": torch.tensor(entry["label"], dtype=torch.long),
        }


def build_entries(labels_path, npy_dir, modality):
    rows = []
    excluded = []
    reference = _pet_reference_mask() if modality in {"av45", "fdg"} else None
    with labels_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (
                row["dataset"] == "adni"
                and row["modality"] == modality
                and row["dx"] in {"CN", "AD"}
            ):
                path = npy_dir / row["npy_basename"]
                if path.is_file():
                    if reference is not None:
                        raw = np.load(path, mmap_mode="r", allow_pickle=False)
                        values = np.asarray(raw[reference])
                        if (
                            values.size == 0
                            or not np.isfinite(values).all()
                            or float(values.mean()) <= 1e-8
                        ):
                            excluded.append(row["npy_basename"])
                            continue
                    rows.append(
                        {
                            "path": str(path),
                            "subject_id": row["subject_id"],
                            "basename": row["npy_basename"],
                            "label": int(row["dx"] == "AD"),
                        }
                    )
    first_by_subject = {}
    for row in sorted(rows, key=lambda value: value["basename"]):
        first_by_subject.setdefault(row["subject_id"], row)
    return (
        [first_by_subject[subject] for subject in sorted(first_by_subject)],
        excluded,
    )


@torch.no_grad()
def extract(model, loader, mappings, modality, device):
    features, labels = [], []
    model.eval()
    for batch in loader:
        x = batch["x"].to(device)
        modality_tensor = torch.full(
            (x.shape[0],),
            MODALITY_ID[modality],
            dtype=torch.long,
            device=device,
        )
        output = model.encode_roi_features(x, modality_tensor, mappings)
        features.append(
            torch.cat(
                [
                    output["atlases"][name]["roi_tokens"]
                    for name in ATLAS_ORDER
                ],
                dim=1,
            )
            .cpu()
            .float()
            .numpy()
        )
        labels.append(batch["label"].numpy())
    return np.concatenate(features), np.concatenate(labels)


def parse_modality(value):
    name, directory = value.split("=", 1)
    if name not in {"t1", "av45", "fdg"}:
        raise argparse.ArgumentTypeError("unsupported modality")
    return name, Path(directory).expanduser()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument(
        "--modality", action="append", type=parse_modality, required=True
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_omnimira(args.checkpoint, args.config, device=str(device))
    mappings = {
        name: torch.from_numpy(value.copy()).long().to(device)
        for name, value in fixed_patch_to_roi().items()
    }

    report = {
        "checkpoint": args.checkpoint,
        "mapping": "overlap",
        "protocol": "full-cohort nested subject-level five-fold",
        "tasks": {},
    }
    labels_path = Path(args.labels).expanduser()
    for modality, npy_dir in args.modality:
        entries, excluded = build_entries(labels_path, npy_dir, modality)
        loader = DataLoader(
            TaskDataset(entries, modality),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
        )
        features, labels = extract(model, loader, mappings, modality, device)
        report["tasks"][modality] = evaluate_full_cohort_fivefold(
            features, labels, args.n_jobs
        )
        report["tasks"][modality]["excluded_input_contract_failures"] = len(
            excluded
        )

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
