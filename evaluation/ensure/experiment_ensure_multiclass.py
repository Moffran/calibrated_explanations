"""Run ensured evaluation over all multiclass datasets.

This uses `WrapCalibratedExplainer` in classification mode (multiclass is
handled automatically in core).

Evaluation-only code (ADR-010).
"""

from __future__ import annotations

import argparse
import pickle
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import sys


# NOTE: VS Code's Python execution wrapper may not include the repo root on
# sys.path. Add it so `import evaluation...` works when running this file.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
from calibrated_explanations import WrapCalibratedExplainer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from evaluation.ensure.common_ensure import (
    EnsureRunConfig,
    ablation_counts,
    ranking_validation_for_instance,
    subsample_calibration,
    summarize_ranking_validation,
    timed_call,
    _safe_calibration_pool_size,
    _safe_test_size,
)
from evaluation.ensure.common_ensure import can_run_dataset
from evaluation.ensure.datasets_ensure import MULTICLASS_DATASETS, load_multiclass_dataset


def _mean_dict(rows: list[dict[str, int]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def run_dataset(dataset_name: str, *, config: EnsureRunConfig) -> dict:
    ds = load_multiclass_dataset(dataset_name)
    ok, reason = can_run_dataset(
        n_samples=int(ds.X.shape[0]), task="multiclass", config=config, n_classes=int(len(np.unique(ds.y)))
    )
    if not ok:
        return {"meta": {"name": ds.name, "skipped": True, "reason": reason}, "by_calibration_size": {}}

    n_samples = int(ds.X.shape[0])
    test_size = _safe_test_size(n_samples, config.test_size)

    x_train, x_test, y_train, y_test = train_test_split(
        ds.X,
        ds.y,
        test_size=test_size,
        random_state=config.random_state,
        stratify=ds.y,
    )

    n_classes = int(len(np.unique(y_train)))
    cal_pool_size = _safe_calibration_pool_size(
        len(x_train),
        max(config.calibration_sizes),
        min_remaining=max(100, n_classes),
    )
    X_prop_train, x_cal_pool, y_prop_train, y_cal_pool = train_test_split(
        x_train,
        y_train,
        test_size=cal_pool_size,
        random_state=config.random_state,
        stratify=y_train,
    )

    model = WrapCalibratedExplainer(RandomForestClassifier(random_state=config.random_state))
    model.fit(X_prop_train, y_prop_train)

    n_classes = int(len(np.unique(y_train)))
    effective_cal_sizes = [
        s for s in config.calibration_sizes if s <= len(x_cal_pool) and int(s) >= n_classes
    ]
    if not effective_cal_sizes:
        if len(x_cal_pool) >= n_classes:
            effective_cal_sizes = [len(x_cal_pool)]
        else:
            return {"meta": {"name": ds.name, "skipped": True, "reason": f"calibration pool too small for {n_classes} classes"}, "by_calibration_size": {}}

    dataset_results: dict = {
        "meta": {
            "name": ds.name,
            "n_samples": n_samples,
            "n_features": int(ds.X.shape[1]),
            "n_classes": int(len(np.unique(ds.y))),
            "test_size": int(test_size),
            "calibration_pool_size": int(len(x_cal_pool)),
            "effective_calibration_sizes": effective_cal_sizes,
        },
        "by_calibration_size": {},
    }

    for cal_size in effective_cal_sizes:
        x_cal, y_cal = subsample_calibration(
            x_cal_pool,
            y_cal_pool,
            cal_size,
            random_state=config.random_state + int(cal_size),
        )

        try:
            model.calibrate(
                x_cal,
                y_cal,
                mode="classification",
                feature_names=ds.feature_names,
                categorical_features=ds.categorical_features,
            )
        except Exception as exc:
            return {
                "meta": {
                    "name": ds.name,
                    "skipped": True,
                    "reason": f"calibration failed: {type(exc).__name__}: {exc}",
                },
                "by_calibration_size": {},
            }

        explanations, explore_total_time = timed_call(model.explore_alternatives, x_test)
        explore_counts: list[dict[str, int]] = []
        explore_validations: list[dict] = []

        for expl in explanations:
            explore_counts.append(ablation_counts(expl))
            explore_validations.append(
                ranking_validation_for_instance(
                    expl,
                    weights=config.ranking_weights,
                    normalize=False,
                )
            )

        conj_counts: list[dict[str, int]] = []
        conj_validations: list[dict] = []
        conj_add_times: list[float] = []

        for expl in explanations:
            conj_expl = deepcopy(expl)
            _, conj_add_time = timed_call(
                conj_expl.add_conjunctions,
                n_top_features=config.n_top_features,
                max_rule_size=config.max_rule_size,
            )
            conj_add_times.append(conj_add_time)
            conj_counts.append(ablation_counts(conj_expl))
            conj_validations.append(
                ranking_validation_for_instance(
                    conj_expl,
                    weights=config.ranking_weights,
                    normalize=False,
                )
            )

        dataset_results["by_calibration_size"][int(cal_size)] = {
            "explore": {
                "counts_mean": _mean_dict(explore_counts),
                "time_mean_seconds": float(explore_total_time / max(1, len(explanations))),
                "ranking_validation": summarize_ranking_validation(
                    explore_validations, weights=config.ranking_weights
                ),
            },
            "conjugate": {
                "counts_mean": _mean_dict(conj_counts),
                "time_mean_seconds": float(np.mean(conj_add_times)) if conj_add_times else 0.0,
                "ranking_validation": summarize_ranking_validation(
                    conj_validations, weights=config.ranking_weights
                ),
            },
        }

    return dataset_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=str,
        default=str(
            Path(__file__).resolve().parent / "results" / "results_ensure_multiclass.pkl"
        ),
        help="Output pickle path",
    )
    parser.add_argument(
        "--limit-datasets",
        type=int,
        default=0,
        help="If >0, run only the first N datasets (for sanity checks)",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=EnsureRunConfig.test_size,
        help="Number of test instances (default matches EnsureRunConfig)",
    )
    parser.add_argument(
        "--calibration-sizes",
        type=int,
        nargs="+",
        default=list(EnsureRunConfig.calibration_sizes),
        help="Calibration sizes (default matches EnsureRunConfig)",
    )
    parser.add_argument(
        "--n-top-features",
        type=int,
        default=EnsureRunConfig.n_top_features,
        help="Top features for conjunctions (default matches EnsureRunConfig)",
    )
    parser.add_argument(
        "--max-rule-size",
        type=int,
        default=EnsureRunConfig.max_rule_size,
        help="Max conjunction size (default matches EnsureRunConfig)",
    )
    args = parser.parse_args()

    config = EnsureRunConfig(
        test_size=int(args.test_size),
        calibration_sizes=tuple(int(s) for s in args.calibration_sizes),
        n_top_features=int(args.n_top_features),
        max_rule_size=int(args.max_rule_size),
    )

    datasets = list(MULTICLASS_DATASETS)
    if args.limit_datasets and args.limit_datasets > 0:
        datasets = datasets[: args.limit_datasets]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        print(f"Loading existing results from {out_path}")
        with out_path.open("rb") as f:
            results = pickle.load(f)
    else:
        results = {
            "task": "multiclass",
            "config": asdict(config),
            "datasets": datasets,
            "results": {},
        }

    for name in datasets:
        if name in results["results"]:
            print(f"[ensure-multiclass] Skipping {name} (already done)")
            continue

        print(f"[ensure-multiclass] {name}")
        results["results"][name] = run_dataset(name, config=config)
        with out_path.open("wb") as f:
            pickle.dump(results, f)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
