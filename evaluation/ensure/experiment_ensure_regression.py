"""Run ensured evaluation over all regression datasets.

This script evaluates two regression settings:
1) Plain regression (no threshold): numeric prediction + conformal interval.
2) Probabilistic regression (thresholded regression, ADR-021): calibrated
   probability for the event y <= t with thresholds at the 25th/50th/75th
   percentiles computed from non-test targets.

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
from crepes.extras import DifficultyEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from evaluation.ensure.common_ensure import (
    EnsureRunConfig,
    ablation_counts,
    compute_thresholds_from_non_test_targets,
    ranking_validation_for_instance,
    subsample_calibration,
    summarize_ranking_validation,
    timed_call,
    _safe_calibration_pool_size,
    _safe_test_size,
)
from evaluation.ensure.common_ensure import can_run_dataset
from evaluation.ensure.datasets_ensure import list_regression_txt_datasets, load_regression_dataset_from_txt


def _mean_dict(rows: list[dict[str, int]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def _run_one_setting(
    *,
    ds_name: str,
    X_prop_train: np.ndarray,
    y_prop_train: np.ndarray,
    x_cal_pool: np.ndarray,
    y_cal_pool: np.ndarray,
    x_test: np.ndarray,
    config: EnsureRunConfig,
    feature_names: list[str],
    categorical_features: list[int],
    threshold: float | None,
    normalize_score: bool,
) -> dict:
    model = WrapCalibratedExplainer(RandomForestRegressor(random_state=config.random_state))
    model.fit(X_prop_train, y_prop_train)

    difficulty_estimator = None
    if threshold is None:
        # Plain regression needs a difficulty estimator for meaningful interval-width comparisons.
        # Fit once per dataset/setting and reuse across calibration sizes.
        difficulty_estimator = DifficultyEstimator().fit(
            X=X_prop_train,
            learner=model.learner,
            scaler=True,
        )

    effective_cal_sizes = [s for s in config.calibration_sizes if s <= len(x_cal_pool)]
    if not effective_cal_sizes:
        effective_cal_sizes = [len(x_cal_pool)]

    per_cal: dict[int, dict] = {}

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
                mode="regression",
                feature_names=feature_names,
                categorical_features=categorical_features,
            )
        except Exception as exc:
            return {
                "meta": {"name": ds_name, "skipped": True, "reason": f"calibration failed: {type(exc).__name__}: {exc}"},
                "by_calibration_size": {},
            }

        if difficulty_estimator is not None:
            model.set_difficulty_estimator(difficulty_estimator)

        explanations, explore_total_time = timed_call(
            model.explore_alternatives,
            x_test,
            threshold=threshold,
            low_high_percentiles=config.low_high_percentiles,
        )

        explore_counts: list[dict[str, int]] = []
        explore_validations: list[dict] = []
        for expl in explanations:
            explore_counts.append(ablation_counts(expl))
            explore_validations.append(
                ranking_validation_for_instance(
                    expl,
                    weights=config.ranking_weights,
                    normalize=normalize_score,
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
                    normalize=normalize_score,
                )
            )

        per_cal[int(cal_size)] = {
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

    return {
        "threshold": threshold,
        "normalize_score": bool(normalize_score),
        "by_calibration_size": per_cal,
    }


def run_dataset(dataset_name: str, *, config: EnsureRunConfig) -> dict:
    ds = load_regression_dataset_from_txt(dataset_name)
    
    ok, reason = can_run_dataset(n_samples=int(ds.X.shape[0]), task="regression", config=config)
    if not ok:
        return {"meta": {"name": ds.name, "skipped": True, "reason": reason}, "plain": {}, "probabilistic": {}}

    n_samples = int(ds.X.shape[0])
    test_size = _safe_test_size(n_samples, config.test_size)

    x_train, x_test, y_train, y_test = train_test_split(
        ds.X,
        ds.y,
        test_size=test_size,
        random_state=config.random_state,
    )

    cal_pool_size = _safe_calibration_pool_size(len(x_train), max(config.calibration_sizes))
    X_prop_train, x_cal_pool, y_prop_train, y_cal_pool = train_test_split(
        x_train,
        y_train,
        test_size=cal_pool_size,
        random_state=config.random_state,
    )

    # Thresholds are computed from non-test targets (train+cal pool), not from test.
    thresholds = compute_thresholds_from_non_test_targets(y_train, percentiles=(25, 50, 75))

    dataset_results: dict = {
        "meta": {
            "name": ds.name,
            "n_samples": n_samples,
            "n_features": int(ds.X.shape[1]),
            "test_size": int(test_size),
            "calibration_pool_size": int(len(x_cal_pool)),
            "thresholds": thresholds,
        },
        "plain": _run_one_setting(
            ds_name=ds.name,
            X_prop_train=X_prop_train,
            y_prop_train=y_prop_train,
            x_cal_pool=x_cal_pool,
            y_cal_pool=y_cal_pool,
            x_test=x_test,
            config=config,
            feature_names=ds.feature_names,
            categorical_features=ds.categorical_features,
            threshold=None,
            normalize_score=config.normalize_plain_regression,
        ),
        "probabilistic": {
            key: _run_one_setting(
                ds_name=ds.name,
                X_prop_train=X_prop_train,
                y_prop_train=y_prop_train,
                x_cal_pool=x_cal_pool,
                y_cal_pool=y_cal_pool,
                x_test=x_test,
                config=config,
                feature_names=ds.feature_names,
                categorical_features=ds.categorical_features,
                threshold=float(value),
                normalize_score=False,
            )
            for key, value in thresholds.items()
        },
    }

    return dataset_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=str,
        default=str(
            Path(__file__).resolve().parent / "results" / "results_ensure_regression.pkl"
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

    datasets = list_regression_txt_datasets()
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
            "task": "regression",
            "config": asdict(config),
            "datasets": datasets,
            "results": {},
        }

    for name in datasets:
        if name in results["results"]:
            print(f"[ensure-regression] Skipping {name} (already done)")
            continue

        print(f"[ensure-regression] {name}")
        results["results"][name] = run_dataset(name, config=config)
        with out_path.open("wb") as f:
            pickle.dump(results, f)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
