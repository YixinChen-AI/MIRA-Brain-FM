"""Strict nested evaluation of one frozen-feature classifier per ROI."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from audit_checkpoint_classification import FDGDataset, build_entries
from audit_checkpoint_roi_ensemble import (
    _bootstrap_auc,
    _parse_checkpoint,
    _permutation_test,
    _roi_predictions,
    extract_roi_tokens,
    majority_patch_to_roi,
)
from omnimira.atlas import ATLAS_ORDER, roi_metadata
from omnimira.atlas import fixed_patch_to_roi
from omnimira.inference import load_omnimira


BASE_CONFIGS = (
    ("standardized_C0.01", True, 0.01),
    ("standardized_C0.1", True, 0.1),
    ("standardized_C1.0", True, 1.0),
    ("raw_C1.0", False, 1.0),
)
TOP_K = (1, 3, 5, 10, 20, 40, 80)
WEIGHT_POWERS = (1, 2, 4)
ATLAS_RANGES = {
    "aal3": np.arange(0, 166),
    "ho69": np.arange(166, 235),
    "yeo7": np.arange(235, 242),
}


def roi_records() -> list[dict]:
    records = []
    metadata = roi_metadata()
    global_index = 0
    for atlas in ATLAS_ORDER:
        for dense_index, (raw_id, name) in enumerate(
            zip(metadata[atlas].ids, metadata[atlas].names)
        ):
            records.append(
                {
                    "global_index": global_index,
                    "atlas": atlas,
                    "dense_index": dense_index,
                    "raw_id": int(raw_id),
                    "name": str(name),
                }
            )
            global_index += 1
    return records


def per_roi_auc(labels: np.ndarray, probability: np.ndarray) -> np.ndarray:
    return np.asarray(
        [roc_auc_score(labels, probability[:, roi]) for roi in range(probability.shape[1])]
    )


def ensemble_candidates(
    probability: np.ndarray, auc: np.ndarray
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    n_roi = probability.shape[1]
    order = np.argsort(auc)[::-1]
    candidates = {}

    equal = np.full(n_roi, 1.0 / n_roi, dtype=np.float64)
    candidates["all_equal"] = (probability @ equal, equal)

    for count in TOP_K:
        selected = order[: min(count, n_roi)]
        weights = np.zeros(n_roi, dtype=np.float64)
        weights[selected] = 1.0 / len(selected)
        candidates[f"top_{count}_equal"] = (probability @ weights, weights)

    for power in WEIGHT_POWERS:
        raw = np.maximum(auc - 0.5, 0.0) ** power
        if raw.sum() <= 1e-12:
            raw = np.ones(n_roi, dtype=np.float64)
        weights = raw / raw.sum()
        candidates[f"all_auc_power_{power}"] = (probability @ weights, weights)
        for count in TOP_K:
            selected = order[: min(count, n_roi)]
            weights = np.zeros(n_roi, dtype=np.float64)
            selected_weights = raw[selected]
            if selected_weights.sum() <= 1e-12:
                selected_weights = np.ones(len(selected), dtype=np.float64)
            weights[selected] = selected_weights / selected_weights.sum()
            candidates[f"top_{count}_auc_power_{power}"] = (
                probability @ weights,
                weights,
            )

    for atlas, indices in ATLAS_RANGES.items():
        atlas_order = indices[np.argsort(auc[indices])[::-1]]
        for count in TOP_K:
            selected = atlas_order[: min(count, len(atlas_order))]
            weights = np.zeros(n_roi, dtype=np.float64)
            weights[selected] = 1.0 / len(selected)
            candidates[f"{atlas}_top_{count}_equal"] = (
                probability @ weights,
                weights,
            )
        for power in WEIGHT_POWERS:
            raw = np.maximum(auc[indices] - 0.5, 0.0) ** power
            if raw.sum() <= 1e-12:
                raw = np.ones(len(indices), dtype=np.float64)
            weights = np.zeros(n_roi, dtype=np.float64)
            weights[indices] = raw / raw.sum()
            candidates[f"{atlas}_all_auc_power_{power}"] = (
                probability @ weights,
                weights,
            )

    for count in TOP_K:
        weights = np.zeros(n_roi, dtype=np.float64)
        for indices in ATLAS_RANGES.values():
            atlas_order = indices[np.argsort(auc[indices])[::-1]]
            selected = atlas_order[: min(count, len(atlas_order))]
            weights[selected] += 1.0 / (len(ATLAS_RANGES) * len(selected))
        candidates[f"atlas_balanced_top_{count}_equal"] = (
            probability @ weights,
            weights,
        )
        for power in WEIGHT_POWERS:
            weights = np.zeros(n_roi, dtype=np.float64)
            for indices in ATLAS_RANGES.values():
                atlas_order = indices[np.argsort(auc[indices])[::-1]]
                selected = atlas_order[: min(count, len(atlas_order))]
                raw = np.maximum(auc[selected] - 0.5, 0.0) ** power
                if raw.sum() <= 1e-12:
                    raw = np.ones(len(selected), dtype=np.float64)
                weights[selected] += raw / (len(ATLAS_RANGES) * raw.sum())
            candidates[f"atlas_balanced_top_{count}_auc_power_{power}"] = (
                probability @ weights,
                weights,
            )
    return candidates


def select_configuration(
    features: np.ndarray,
    labels: np.ndarray,
    n_jobs: int,
    seed: int,
) -> dict:
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    reports = {}
    states = {}
    for name, standardize, c_value in BASE_CONFIGS:
        oof = np.zeros((len(labels), features.shape[1]), dtype=np.float64)
        for train, valid in splitter.split(features, labels):
            oof[valid] = _roi_predictions(
                features[train],
                labels[train],
                features[valid],
                c_value,
                standardize,
                n_jobs,
            )
        auc = per_roi_auc(labels, oof)
        strategies = ensemble_candidates(oof, auc)
        strategy_auc = {
            strategy: float(roc_auc_score(labels, probability))
            for strategy, (probability, _) in strategies.items()
        }
        selected_strategy = max(
            strategy_auc,
            key=lambda strategy: (strategy_auc[strategy], strategy),
        )
        reports[name] = {
            "selected_strategy": selected_strategy,
            "selected_strategy_auc": strategy_auc[selected_strategy],
            "strategy_auc": strategy_auc,
            "median_roi_auc": float(np.median(auc)),
        }
        states[name] = {
            "standardize": standardize,
            "c_value": c_value,
            "roi_auc": auc,
            "strategy": selected_strategy,
            "weights": strategies[selected_strategy][1],
        }

    selected_name = max(
        reports,
        key=lambda name: (reports[name]["selected_strategy_auc"], name),
    )
    return {
        "selected_name": selected_name,
        "selected": states[selected_name],
        "reports": reports,
    }


def fit_selected(
    train_x: np.ndarray,
    train_y: np.ndarray,
    target_x: np.ndarray,
    selection: dict,
    n_jobs: int,
) -> tuple[np.ndarray, np.ndarray]:
    state = selection["selected"]
    roi_probability = _roi_predictions(
        train_x,
        train_y,
        target_x,
        state["c_value"],
        state["standardize"],
        n_jobs,
    )
    return roi_probability @ state["weights"], roi_probability


def evaluate_nested(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    n_jobs: int,
) -> dict:
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    nested_oof = np.zeros(len(train_y), dtype=np.float64)
    outer_reports = []
    selected_rois = Counter()

    for fold, (train, valid) in enumerate(outer.split(train_x, train_y), start=1):
        selection = select_configuration(
            train_x[train], train_y[train], n_jobs, seed=1000 + fold
        )
        probability, _ = fit_selected(
            train_x[train], train_y[train], train_x[valid], selection, n_jobs
        )
        nested_oof[valid] = probability
        weights = selection["selected"]["weights"]
        selected = np.flatnonzero(weights > 0)
        selected_rois.update(selected.tolist())
        outer_reports.append(
            {
                "fold": fold,
                "selected_base_model": selection["selected_name"],
                "selected_strategy": selection["selected"]["strategy"],
                "inner_auc": selection["reports"][selection["selected_name"]][
                    "selected_strategy_auc"
                ],
                "outer_auc": float(roc_auc_score(train_y[valid], probability)),
                "selected_roi_indices": selected.astype(int).tolist(),
            }
        )

    final_selection = select_configuration(train_x, train_y, n_jobs, seed=42)
    test_probability, test_roi_probability = fit_selected(
        train_x, train_y, test_x, final_selection, n_jobs
    )
    final_weights = final_selection["selected"]["weights"]
    train_roi_auc = final_selection["selected"]["roi_auc"]
    test_roi_auc = per_roi_auc(test_y, test_roi_probability)
    prediction = (test_probability >= 0.5).astype(np.int64)
    metadata = roi_records()

    roi_table = []
    for record, train_auc, test_auc, weight in zip(
        metadata, train_roi_auc, test_roi_auc, final_weights
    ):
        roi_table.append(
            {
                **record,
                "train_oof_auc": float(train_auc),
                "test_auc_diagnostic_only": float(test_auc),
                "ensemble_weight": float(weight),
                "outer_fold_selection_count": int(
                    selected_rois[record["global_index"]]
                ),
            }
        )

    permutation = _permutation_test(test_y, test_probability)
    return {
        "protocol": {
            "feature": "one 128-dimensional frozen embedding per ROI",
            "base_model": "one class-balanced logistic regression per ROI",
            "selection": (
                "five-fold outer CV; inner five-fold CV selects scaling, C, "
                "ROI count, and probability weighting"
            ),
            "test_policy": (
                "the fixed subject-disjoint test set is evaluated once after "
                "all choices are fixed from training data"
            ),
        },
        "n_train": int(len(train_y)),
        "n_test": int(len(test_y)),
        "n_roi": int(train_x.shape[1]),
        "nested_train_auc": float(roc_auc_score(train_y, nested_oof)),
        "outer_folds": outer_reports,
        "final_selection": {
            "base_model": final_selection["selected_name"],
            "strategy": final_selection["selected"]["strategy"],
            "inner_auc": final_selection["reports"][
                final_selection["selected_name"]
            ]["selected_strategy_auc"],
            "all_candidate_reports": final_selection["reports"],
        },
        "test_auc": float(roc_auc_score(test_y, test_probability)),
        "test_auc_95_ci": _bootstrap_auc(test_y, test_probability),
        "test_accuracy": float(accuracy_score(test_y, prediction)),
        "test_balanced_accuracy": float(
            balanced_accuracy_score(test_y, prediction)
        ),
        "permutation_p": permutation["p"],
        "permutation_auc_95th": permutation["auc_95th"],
        "selected_rois": sorted(
            [row for row in roi_table if row["ensemble_weight"] > 0],
            key=lambda row: row["ensemble_weight"],
            reverse=True,
        ),
        "all_rois": roi_table,
        "test_oracle_warning": (
            "Per-ROI test AUC is included only for error analysis. It was not "
            "used to select ROIs, hyperparameters, or ensemble weights."
        ),
    }


def evaluate_full_cohort_fivefold(
    features: np.ndarray,
    labels: np.ndarray,
    n_jobs: int,
) -> dict:
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    oof_probability = np.zeros(len(labels), dtype=np.float64)
    oof_roi_probability = np.zeros(
        (len(labels), features.shape[1]), dtype=np.float64
    )
    reports = []
    selected_rois = Counter()

    for fold, (train, test) in enumerate(outer.split(features, labels), start=1):
        selection = select_configuration(
            features[train], labels[train], n_jobs, seed=1000 + fold
        )
        probability, roi_probability = fit_selected(
            features[train], labels[train], features[test], selection, n_jobs
        )
        oof_probability[test] = probability
        oof_roi_probability[test] = roi_probability
        selected = np.flatnonzero(selection["selected"]["weights"] > 0)
        selected_rois.update(selected.tolist())
        reports.append(
            {
                "fold": fold,
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "selected_base_model": selection["selected_name"],
                "selected_strategy": selection["selected"]["strategy"],
                "inner_auc": selection["reports"][selection["selected_name"]][
                    "selected_strategy_auc"
                ],
                "test_auc": float(roc_auc_score(labels[test], probability)),
                "selected_roi_indices": selected.astype(int).tolist(),
            }
        )

    prediction = (oof_probability >= 0.5).astype(np.int64)
    roi_auc = per_roi_auc(labels, oof_roi_probability)
    metadata = roi_records()
    roi_table = sorted(
        [
            {
                **record,
                "cross_fitted_auc": float(auc),
                "outer_fold_selection_count": int(
                    selected_rois[record["global_index"]]
                ),
            }
            for record, auc in zip(metadata, roi_auc)
        ],
        key=lambda row: row["cross_fitted_auc"],
        reverse=True,
    )
    permutation = _permutation_test(labels, oof_probability)
    return {
        "protocol": {
            "split": "subject-level stratified five-fold cross-validation",
            "feature": "one 128-dimensional frozen embedding per ROI",
            "base_model": "one class-balanced logistic regression per ROI",
            "prior": (
                "ROI reliability, scaling, C, ROI count, and weighting are "
                "selected by five-fold OOF predictions within each outer "
                "training fold"
            ),
            "test_policy": (
                "each subject is predicted once by a model and ROI prior that "
                "were fitted without that subject"
            ),
        },
        "n_subjects": int(len(labels)),
        "class_counts": np.bincount(labels).astype(int).tolist(),
        "cross_fitted_auc": float(roc_auc_score(labels, oof_probability)),
        "cross_fitted_auc_95_ci": _bootstrap_auc(labels, oof_probability),
        "accuracy": float(accuracy_score(labels, prediction)),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, prediction)
        ),
        "permutation_p": permutation["p"],
        "folds": reports,
        "roi_cross_fitted_auc": roi_table,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", action="append", type=_parse_checkpoint, required=True
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--npy-dir", required=True)
    parser.add_argument("--train-subjects", required=True)
    parser.add_argument("--test-subjects", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--full-cohort-fivefold", action="store_true")
    parser.add_argument(
        "--mapping-mode",
        choices=("overlap", "majority"),
        default="overlap",
    )
    args = parser.parse_args()

    train_subjects = {
        value.strip()
        for value in Path(args.train_subjects).expanduser().read_text().splitlines()
        if value.strip()
    }
    test_subjects = {
        value.strip()
        for value in Path(args.test_subjects).expanduser().read_text().splitlines()
        if value.strip()
    }
    train_entries, test_entries, excluded = build_entries(
        Path(args.labels).expanduser(),
        Path(args.npy_dir).expanduser(),
        train_subjects,
        test_subjects,
    )
    train_loader = DataLoader(
        FDGDataset(train_entries),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )
    test_loader = DataLoader(
        FDGDataset(test_entries),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )
    device = torch.device(args.device)
    mapping_arrays = (
        fixed_patch_to_roi()
        if args.mapping_mode == "overlap"
        else majority_patch_to_roi()
    )
    mappings = {
        name: torch.from_numpy(value.copy()).long().to(device)
        for name, value in mapping_arrays.items()
    }

    report = {
        "task": "ADNI FDG CN vs AD",
        "mapping": args.mapping_mode,
        "n_train": len(train_entries),
        "n_test": len(test_entries),
        "excluded_input_contract_failures": len(excluded),
        "checkpoints": {},
    }
    for name, checkpoint in args.checkpoint:
        model = load_omnimira(checkpoint, args.config, device=str(device))
        train_x, train_y = extract_roi_tokens(
            model, train_loader, mappings, device
        )
        test_x, test_y = extract_roi_tokens(
            model, test_loader, mappings, device
        )
        del model
        torch.cuda.empty_cache()
        if args.full_cohort_fivefold:
            report["checkpoints"][name] = evaluate_full_cohort_fivefold(
                np.concatenate([train_x, test_x], axis=0),
                np.concatenate([train_y, test_y], axis=0),
                args.n_jobs,
            )
        else:
            report["checkpoints"][name] = evaluate_nested(
                train_x, train_y, test_x, test_y, args.n_jobs
            )

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
