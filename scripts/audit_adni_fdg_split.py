"""Audit the ADNI FDG disease split without fitting downstream models."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


def read_subjects(path: Path) -> set[str]:
    return {value.strip() for value in path.read_text().splitlines() if value.strip()}


def summarize(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"n": 0}
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "sd": float(array.std()),
        "median": float(np.median(array)),
        "q05": float(np.percentile(array, 5)),
        "q95": float(np.percentile(array, 95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--npy-dir", required=True)
    parser.add_argument("--train-subjects", required=True)
    parser.add_argument("--test-subjects", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    train_subjects = read_subjects(Path(args.train_subjects).expanduser())
    test_subjects = read_subjects(Path(args.test_subjects).expanduser())
    if train_subjects & test_subjects:
        raise ValueError("train and test subject files overlap")

    npy_dir = Path(args.npy_dir).expanduser()
    rows = []
    with Path(args.labels).expanduser().open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (
                row["dataset"] == "adni"
                and row["modality"] == "fdg"
                and row["dx"] in {"CN", "AD"}
                and (npy_dir / row["npy_basename"]).is_file()
            ):
                if row["subject_id"] in train_subjects:
                    split = "train"
                elif row["subject_id"] in test_subjects:
                    split = "test"
                else:
                    continue
                row["split"] = split
                rows.append(row)

    keys = [(row["split"], row["subject_id"]) for row in rows]
    duplicates = [key for key, count in Counter(keys).items() if count > 1]
    if duplicates:
        raise ValueError(f"repeated subject scans: {duplicates[:10]}")

    manifest = json.loads(Path(args.manifest).expanduser().read_text())
    manifest_subjects = {row["subject_id"] for row in manifest}
    manifest_by_modality = {}
    for modality in ("t1", "av45", "fdg", "ct"):
        manifest_by_modality[modality] = {
            row["subject_id"] for row in manifest if row["modality"] == modality
        }

    report = {
        "n_rows": len(rows),
        "class_counts": {},
        "offset_days": {},
        "raw_volume_foreground": {},
        "pretraining_subject_overlap": {},
    }
    for split in ("train", "test"):
        split_rows = [row for row in rows if row["split"] == split]
        report["class_counts"][split] = dict(Counter(row["dx"] for row in split_rows))
        report["offset_days"][split] = {}
        report["raw_volume_foreground"][split] = {}
        subjects = {row["subject_id"] for row in split_rows}
        report["pretraining_subject_overlap"][split] = {
            "any_modality": len(subjects & manifest_subjects),
            **{
                modality: len(subjects & modality_subjects)
                for modality, modality_subjects in manifest_by_modality.items()
            },
        }
        for diagnosis in ("CN", "AD"):
            diagnosis_rows = [row for row in split_rows if row["dx"] == diagnosis]
            offsets = [
                abs(float(row["offset_days"]))
                for row in diagnosis_rows
                if row["offset_days"] not in {"", "nan", "None"}
            ]
            report["offset_days"][split][diagnosis] = summarize(offsets)

            means, standard_deviations, foreground_fractions = [], [], []
            for row in diagnosis_rows:
                array = np.load(
                    npy_dir / row["npy_basename"],
                    mmap_mode="r",
                    allow_pickle=False,
                )
                finite = np.asarray(array[np.isfinite(array)], dtype=np.float64)
                foreground = finite[finite > 0]
                if not len(foreground):
                    continue
                means.append(float(foreground.mean()))
                standard_deviations.append(float(foreground.std()))
                foreground_fractions.append(float(len(foreground) / array.size))
            report["raw_volume_foreground"][split][diagnosis] = {
                "mean": summarize(means),
                "within_scan_sd": summarize(standard_deviations),
                "fraction": summarize(foreground_fractions),
            }

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
