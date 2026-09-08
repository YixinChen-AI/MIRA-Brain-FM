"""Evaluate frozen OmniMIRA features with one classifier per ROI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from audit_checkpoint_classification import FDGDataset, build_entries
from omnimira.atlas import (
    ATLAS_ORDER,
    fixed_patch_to_roi,
    raw_label_ids,
    resample_atlas_to_model,
)
from omnimira.inference import _build_omnimira_for_tests, load_omnimira


@torch.no_grad()
def extract_roi_tokens(model, loader, mappings, device):
    features, labels = [], []
    model.eval()
    for batch in loader:
        x = batch["x"].to(device)
        modality = torch.full((x.shape[0],), 2, dtype=torch.long, device=device)
        output = model.encode_roi_features(x, modality, mappings)
        features.append(torch.cat([
            output["atlases"][name]["roi_tokens"] for name in ATLAS_ORDER
        ], dim=1).cpu().float().numpy())
        labels.append(batch["label"].numpy())
    return np.concatenate(features), np.concatenate(labels)


def majority_patch_to_roi():
    """Assign each patch to the ROI occupying most foreground voxels."""
    mappings = {}
    for name in ATLAS_ORDER:
        atlas = resample_atlas_to_model(name)
        raw_ids = raw_label_ids(name)
        raw_to_dense = {raw_id: index for index, raw_id in enumerate(raw_ids)}
        patches = (
            atlas.reshape(12, 8, 14, 8, 12, 8)
            .transpose(0, 2, 4, 1, 3, 5)
            .reshape(12 * 14 * 12, -1)
        )
        mapping = np.full(len(patches), -1, dtype=np.int64)
        for index, patch in enumerate(patches):
            foreground = patch[patch > 0]
            if foreground.size:
                labels, counts = np.unique(foreground, return_counts=True)
                mapping[index] = raw_to_dense[int(labels[np.argmax(counts)])]
        mappings[name] = mapping
    return mappings


def _classifier(c_value, standardize):
    classifier = LogisticRegression(
        C=c_value,
        max_iter=2000,
        solver="liblinear",
        class_weight="balanced",
        random_state=42,
    )
    if standardize:
        return make_pipeline(StandardScaler(), classifier)
    return classifier


def _fit_predict(train_x, train_y, target_x, c_value, standardize):
    if float(np.std(train_x)) < 1e-10:
        return np.full(len(target_x), float(np.mean(train_y)), dtype=np.float64)
    model = _classifier(c_value, standardize)
    model.fit(train_x, train_y)
    return model.predict_proba(target_x)[:, 1]


def _roi_predictions(
    train_x, train_y, target_x, c_value, standardize, n_jobs
):
    columns = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_fit_predict)(
            train_x[:, roi], train_y, target_x[:, roi], c_value, standardize
        )
        for roi in range(train_x.shape[1])
    )
    return np.column_stack(columns)


def _weights(per_roi_auc):
    weights = np.maximum(per_roi_auc - 0.5, 0.0) ** 2
    if float(weights.sum()) <= 1e-12:
        weights = np.ones_like(weights)
    return weights / weights.sum()


def _oof_predictions(features, labels, c_value, standardize, n_jobs):
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    predictions = np.zeros((len(labels), features.shape[1]), dtype=np.float64)
    for train, valid in splitter.split(features, labels):
        predictions[valid] = _roi_predictions(
            features[train], labels[train], features[valid],
            c_value, standardize, n_jobs
        )
    return predictions


def _ensemble_strategies(roi_probability, roi_auc):
    strategies = {"equal": roi_probability.mean(axis=1)}
    for power in (1, 2, 4, 8, 16):
        weights = np.maximum(roi_auc - 0.5, 0.0) ** power
        if float(weights.sum()) <= 1e-12:
            weights = np.ones_like(weights)
        weights /= weights.sum()
        strategies[f"auc_power_{power}"] = roi_probability @ weights
    for temperature in (5, 10, 20, 50, 100, 200):
        scores = roi_auc * temperature
        scores -= scores.max()
        weights = np.exp(scores)
        weights /= weights.sum()
        strategies[f"softmax_{temperature}"] = roi_probability @ weights
    order = np.argsort(roi_auc)[::-1]
    for count in (1, 3, 5, 10, 20, 40, 80):
        strategies[f"top_{count}"] = roi_probability[:, order[:count]].mean(axis=1)
    for threshold in (0.55, 0.58, 0.60, 0.65, 0.70):
        selected = roi_auc >= threshold
        if selected.any():
            strategies[f"threshold_{threshold:.2f}"] = roi_probability[:, selected].mean(axis=1)
    return strategies


def _bootstrap_auc(labels, probability, repetitions=2000):
    rng = np.random.default_rng(42)
    by_class = [np.flatnonzero(labels == value) for value in (0, 1)]
    values = []
    for _ in range(repetitions):
        sample = np.concatenate([
            rng.choice(indices, size=len(indices), replace=True)
            for indices in by_class
        ])
        values.append(roc_auc_score(labels[sample], probability[sample]))
    return [float(value) for value in np.percentile(values, [2.5, 97.5])]


def _permutation_test(labels, probability, repetitions=10000):
    observed = roc_auc_score(labels, probability)
    rng = np.random.default_rng(2026)
    null = np.asarray([
        roc_auc_score(rng.permutation(labels), probability)
        for _ in range(repetitions)
    ])
    return {
        "p": float((1 + np.sum(null >= observed)) / (1 + repetitions)),
        "auc_95th": float(np.percentile(null, 95)),
    }


def _stack_oof_predictions(oof, labels, test_probability):
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=73)
    reports = {}
    for c_value in (0.001, 0.01, 0.1, 1.0, 10.0):
        meta_oof = np.zeros(len(labels), dtype=np.float64)
        for train, valid in splitter.split(oof, labels):
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=c_value,
                    max_iter=2000,
                    solver="liblinear",
                    class_weight="balanced",
                    random_state=42,
                ),
            )
            model.fit(oof[train], labels[train])
            meta_oof[valid] = model.predict_proba(oof[valid])[:, 1]
        reports[str(c_value)] = float(roc_auc_score(labels, meta_oof))
    selected_c = float(max(reports, key=reports.get))
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=selected_c,
            max_iter=2000,
            solver="liblinear",
            class_weight="balanced",
            random_state=42,
        ),
    )
    model.fit(oof, labels)
    return model.predict_proba(test_probability)[:, 1], selected_c, reports


def evaluate(train_x, train_y, test_x, test_y, n_jobs):
    candidates = [
        (True, 0.001), (True, 0.01), (True, 0.1), (True, 1.0),
        (False, 0.1), (False, 1.0), (False, 10.0),
    ]
    candidate_reports = {}
    candidate_state = {}
    for standardize, c_value in candidates:
        key = f"{'standardized' if standardize else 'raw'}_C{c_value}"
        oof = _oof_predictions(
            train_x, train_y, c_value, standardize, n_jobs
        )
        roi_auc = np.asarray([
            roc_auc_score(train_y, oof[:, roi])
            for roi in range(oof.shape[1])
        ])
        strategy_auc = {
            name: float(roc_auc_score(train_y, probability))
            for name, probability in _ensemble_strategies(oof, roi_auc).items()
        }
        best_strategy = max(strategy_auc, key=strategy_auc.get)
        candidate_reports[key] = {
            "best_oof_ensemble_auc": strategy_auc[best_strategy],
            "best_strategy": best_strategy,
            "median_roi_auc": float(np.median(roi_auc)),
            "strategies": strategy_auc,
        }
        candidate_state[key] = (
            roi_auc, standardize, c_value, best_strategy, oof
        )

    selected_key = max(
        candidate_reports,
        key=lambda key: candidate_reports[key]["best_oof_ensemble_auc"],
    )
    roi_auc, standardize, selected_c, selected_strategy, selected_oof = (
        candidate_state[selected_key]
    )
    test_roi_probability = _roi_predictions(
        train_x, train_y, test_x, selected_c, standardize, n_jobs
    )
    test_strategies = _ensemble_strategies(test_roi_probability, roi_auc)
    probability = test_strategies[selected_strategy]
    prediction = (probability >= 0.5).astype(int)
    permutation = _permutation_test(test_y, probability)

    order = np.argsort(roi_auc)[::-1]
    test_roi_auc = np.asarray([
        roc_auc_score(test_y, test_roi_probability[:, roi])
        for roi in range(test_roi_probability.shape[1])
    ])
    oracle_order = np.argsort(test_roi_auc)[::-1]
    oracle_auc_power = {}
    for power in (1, 2, 4, 8, 16):
        oracle_weights = np.maximum(test_roi_auc - 0.5, 0.0) ** power
        if float(oracle_weights.sum()) <= 1e-12:
            oracle_weights = np.ones_like(oracle_weights)
        oracle_weights /= oracle_weights.sum()
        oracle_auc_power[str(power)] = float(
            roc_auc_score(test_y, test_roi_probability @ oracle_weights)
        )
    oracle_topk = {
        str(k): float(roc_auc_score(
            test_y, test_roi_probability[:, oracle_order[:k]].mean(axis=1)
        ))
        for k in (1, 3, 5, 10, 20, 40)
    }
    equal_probability = test_roi_probability.mean(axis=1)
    top20_probability = test_roi_probability[:, order[:20]].mean(axis=1)

    paper_roi_auc, _, _, _, paper_oof = candidate_state["raw_C1.0"]
    if selected_key == "raw_C1.0":
        paper_test_roi_probability = test_roi_probability
    else:
        paper_test_roi_probability = _roi_predictions(
            train_x, train_y, test_x, 1.0, False, n_jobs
        )
    paper_probability = _ensemble_strategies(
        paper_test_roi_probability, paper_roi_auc
    )["auc_power_1"]
    stacked_probability, stacked_c, stacked_cv = _stack_oof_predictions(
        paper_oof, train_y, paper_test_roi_probability
    )

    return {
        "selected_c": float(selected_c),
        "selected_standardization": bool(standardize),
        "selected_strategy": selected_strategy,
        "selected_configuration": selected_key,
        "selection": candidate_reports,
        "test_auc": float(roc_auc_score(test_y, probability)),
        "test_auc_95_ci": _bootstrap_auc(test_y, probability),
        "test_accuracy": float(accuracy_score(test_y, prediction)),
        "test_balanced_accuracy": float(
            balanced_accuracy_score(test_y, prediction)
        ),
        "permutation_p": permutation["p"],
        "permutation_auc_95th": permutation["auc_95th"],
        "equal_weight_test_auc": float(
            roc_auc_score(test_y, equal_probability)
        ),
        "top20_equal_test_auc": float(
            roc_auc_score(test_y, top20_probability)
        ),
        "paper_protocol": {
            "classifier": "raw logistic regression C=1",
            "ensemble": "all ROI probabilities weighted by max(OOF AUC - 0.5, 0)",
            "test_auc": float(roc_auc_score(test_y, paper_probability)),
            "test_auc_95_ci": _bootstrap_auc(test_y, paper_probability),
        },
        "oof_stacking_diagnostic": {
            "selected_c": stacked_c,
            "train_cv_auc_by_c": stacked_cv,
            "test_auc": float(roc_auc_score(test_y, stacked_probability)),
            "test_auc_95_ci": _bootstrap_auc(test_y, stacked_probability),
        },
        "median_train_oof_roi_auc": float(np.median(roi_auc)),
        "n_roi_models": int(train_x.shape[1]),
        "top_train_oof_roi_indices": order[:20].astype(int).tolist(),
        "top_train_oof_roi_auc": roi_auc[order[:20]].astype(float).tolist(),
        "test_oracle_diagnostic": {
            "warning": (
                "uses test labels for ROI selection and weighting; "
                "diagnostic only, not a valid reported estimate"
            ),
            "best_single_roi_auc": float(test_roi_auc[oracle_order[0]]),
            "top_test_roi_indices": oracle_order[:20].astype(int).tolist(),
            "top_test_roi_auc": test_roi_auc[oracle_order[:20]].astype(float).tolist(),
            "auc_weighted_power": oracle_auc_power,
            "topk_equal_auc": oracle_topk,
        },
    }


def _parse_checkpoint(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("checkpoint must use NAME=PATH")
    name, path = value.split("=", 1)
    return name, path


def main():
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
        FDGDataset(train_entries), batch_size=args.batch_size,
        shuffle=False, num_workers=4,
    )
    test_loader = DataLoader(
        FDGDataset(test_entries), batch_size=args.batch_size,
        shuffle=False, num_workers=4,
    )
    device = torch.device(args.device)
    mapping_arrays = (
        majority_patch_to_roi()
        if args.mapping_mode == "majority"
        else fixed_patch_to_roi()
    )
    mappings = {
        name: torch.from_numpy(value.copy()).long().to(device)
        for name, value in mapping_arrays.items()
    }

    report = {
        "task": "ADNI FDG CN vs AD",
        "protocol": (
            "one logistic model per ROI; ensemble weights from train-only "
            "out-of-fold AUC; fixed subject-disjoint test"
        ),
        "n_train": len(train_entries),
        "n_test": len(test_entries),
        "mapping_mode": args.mapping_mode,
        "excluded_input_contract_failures": len(excluded),
        "checkpoints": {},
    }
    reference_labels = None
    for name, checkpoint in args.checkpoint:
        model = load_omnimira(checkpoint, args.config, device=str(device))
        train_x, train_y = extract_roi_tokens(
            model, train_loader, mappings, device
        )
        test_x, test_y = extract_roi_tokens(model, test_loader, mappings, device)
        del model
        torch.cuda.empty_cache()
        if reference_labels is None:
            reference_labels = (train_y.copy(), test_y.copy())
        elif not (
            np.array_equal(reference_labels[0], train_y)
            and np.array_equal(reference_labels[1], test_y)
        ):
            raise RuntimeError("checkpoint evaluations used different labels")
        report["checkpoints"][name] = evaluate(
            train_x, train_y, test_x, test_y, args.n_jobs
        )

    torch.manual_seed(42)
    random_model = _build_omnimira_for_tests(args.config, device=str(device))
    random_train, random_train_y = extract_roi_tokens(
        random_model, train_loader, mappings, device
    )
    random_test, random_test_y = extract_roi_tokens(
        random_model, test_loader, mappings, device
    )
    if not (
        np.array_equal(reference_labels[0], random_train_y)
        and np.array_equal(reference_labels[1], random_test_y)
    ):
        raise RuntimeError("random control used different labels")
    report["random_init"] = evaluate(
        random_train, random_train_y, random_test, random_test_y, args.n_jobs
    )

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
