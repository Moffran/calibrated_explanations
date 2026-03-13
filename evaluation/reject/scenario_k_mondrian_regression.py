"""Scenario K: paper-style and heuristic regression reject comparison."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from crepes.extras import DifficultyEstimator
from sklearn.ensemble import RandomForestRegressor

from .common_reject import (
    RunConfig,
    accepted_interval_metrics,
    build_regression_bundle,
    predicted_value_categorizer,
    quantile_grid,
    regression_mse,
    task_specs,
    write_csv_json_md,
)


def _paper_style_rows(spec, config: RunConfig) -> list[dict[str, float | str]]:
    learner = RandomForestRegressor(
        n_estimators=60 if config.quick else 120,
        random_state=config.seed,
        max_depth=8 if config.quick else None,
        n_jobs=1,
    )
    # K1 (paper-style): sigma-normalized conformal regression reject.
    # Step 1: fit difficulty estimator on training data.
    base_bundle = build_regression_bundle(spec, config, learner=learner)
    de = DifficultyEstimator().fit(X=base_bundle.x_fit, learner=base_bundle.wrapper.learner, y=base_bundle.y_fit)
    # Step 2: rebuild bundle with sigma-normalization only (no conflicting mc).
    bundle = build_regression_bundle(spec, config, learner=learner)
    bundle.wrapper.set_difficulty_estimator(de)
    # Step 3: compute difficulty scores on calibration and test sets once.
    diff_cal = np.asarray(de.apply(bundle.x_cal), dtype=float)
    diff_test = np.asarray(de.apply(bundle.x_test), dtype=float)
    pred, (low, high) = bundle.wrapper.predict(bundle.x_test, uq_interval=True)

    rows: list[dict[str, float | str]] = []
    for requested_reject_rate in quantile_grid(config.quick):
        threshold = float(np.quantile(diff_cal, 1.0 - requested_reject_rate))
        rejected = diff_test > threshold
        accepted = ~rejected
        metrics = accepted_interval_metrics(bundle.y_test, pred, low, high, accepted)
        rows.append(
            {
                "dataset": spec.name,
                "confidence": 0.95,
                "method": "paper_difficulty_mondrian",
                "difficulty_estimator": "default",
                "requested_reject_rate": float(requested_reject_rate),
                "empirical_reject_rate": float(np.mean(rejected)),
                "accepted_coverage": metrics["accepted_coverage"],
                "accepted_interval_width": metrics["accepted_interval_width"],
                "accepted_mae": metrics["accepted_mae"],
                "accepted_mse": metrics["accepted_mse"],
                "interval_coverage_all": 1.0 - float(np.mean((bundle.y_test < low) | (bundle.y_test > high))),
                "mse_all": regression_mse(bundle.y_test, pred),
                "guarantee_status": "target_formal_result",
            }
        )
    return rows


def _threshold_rows(spec, config: RunConfig) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for requested_reject_rate in quantile_grid(config.quick):
        bundle = build_regression_bundle(spec, config, seed_offset=int(requested_reject_rate * 1000))
        threshold = float(np.quantile(bundle.y_cal, requested_reject_rate))
        bundle.wrapper.explainer.reject_orchestrator.initialize_reject_learner(
            threshold=threshold
        )
        breakdown = bundle.wrapper.explainer.reject_orchestrator.predict_reject_breakdown(
            bundle.x_test,
            confidence=0.95,
            threshold=threshold,
        )
        rejected = np.asarray(breakdown["rejected"], dtype=bool)
        accepted = ~rejected
        metrics = accepted_interval_metrics(
            bundle.y_test,
            bundle.baseline_pred,
            bundle.baseline_low,
            bundle.baseline_high,
            accepted,
        )
        rows.append(
            {
                "dataset": spec.name,
                "confidence": 0.95,
                "method": "threshold_baseline",
                "difficulty_estimator": "none",
                "requested_reject_rate": float(requested_reject_rate),
                "empirical_reject_rate": float(np.mean(rejected)),
                "accepted_coverage": metrics["accepted_coverage"],
                "accepted_interval_width": metrics["accepted_interval_width"],
                "accepted_mae": metrics["accepted_mae"],
                "accepted_mse": metrics["accepted_mse"],
                "interval_coverage_all": 1.0
                - float(np.mean((bundle.y_test < bundle.baseline_low) | (bundle.y_test > bundle.baseline_high))),
                "mse_all": regression_mse(bundle.y_test, bundle.baseline_pred),
                "guarantee_status": "heuristic",
            }
        )
    return rows


def _value_bin_rows(spec, config: RunConfig) -> list[dict[str, float | str]]:
    learner = RandomForestRegressor(
        n_estimators=60 if config.quick else 120,
        random_state=config.seed,
        max_depth=8 if config.quick else None,
        n_jobs=1,
    )
    rows: list[dict[str, float | str]] = []
    for n_bins in ((4,) if config.quick else (4, 8)):
        base_bundle = build_regression_bundle(spec, config, learner=learner)
        cal_pred = np.asarray(base_bundle.wrapper.learner.predict(base_bundle.x_cal), dtype=float)
        edges = np.quantile(cal_pred, np.linspace(0.0, 1.0, n_bins + 1))
        edges[0] = -np.inf
        edges[-1] = np.inf
        categorizer = predicted_value_categorizer(base_bundle.wrapper.learner.predict, edges)
        bundle = build_regression_bundle(
            spec,
            config,
            learner=learner,
            explainer_kwargs={"mc": categorizer},
        )
        cal_pred2, (cal_low, cal_high) = bundle.wrapper.predict(bundle.x_cal, uq_interval=True)
        _ = cal_pred2
        cal_bins = categorizer(bundle.x_cal)
        widths = np.asarray(cal_high) - np.asarray(cal_low)
        overall_median = float(np.median(widths))
        reject_bins = {
            int(bin_id)
            for bin_id in np.unique(cal_bins)
            if float(np.median(widths[cal_bins == bin_id])) > overall_median * 1.5
        }
        pred, (low, high) = bundle.wrapper.predict(bundle.x_test, uq_interval=True)
        test_bins = categorizer(bundle.x_test)
        rejected = np.isin(test_bins, list(reject_bins))
        accepted = ~rejected
        metrics = accepted_interval_metrics(bundle.y_test, pred, low, high, accepted)
        rows.append(
            {
                "dataset": spec.name,
                "confidence": 0.95,
                "method": "value_bin_width_baseline",
                "difficulty_estimator": "predicted_value_bins",
                "requested_reject_rate": float("nan"),
                "empirical_reject_rate": float(np.mean(rejected)),
                "accepted_coverage": metrics["accepted_coverage"],
                "accepted_interval_width": metrics["accepted_interval_width"],
                "accepted_mae": metrics["accepted_mae"],
                "accepted_mse": metrics["accepted_mse"],
                "interval_coverage_all": 1.0 - float(np.mean((bundle.y_test < low) | (bundle.y_test > high))),
                "mse_all": regression_mse(bundle.y_test, pred),
                "guarantee_status": "heuristic",
            }
        )
    return rows


def run(config: RunConfig) -> None:
    """Compare paper-style and heuristic regression reject methods."""
    rows: list[dict[str, float | str]] = []
    for spec in task_specs("regression", quick=config.quick):
        rows.extend(_paper_style_rows(spec, config))
        rows.extend(_threshold_rows(spec, config))
        rows.extend(_value_bin_rows(spec, config))

    df = pd.DataFrame(rows)
    meta = {
        "scenario": "scenario_k_mondrian_regression",
        "display_name": "Scenario K — Regression reject comparison",
        "quick": config.quick,
        "highlights": [
            "The primary method follows the 2024 conformal regression with reject option paper via difficulty-based Mondrian categories.",
            "Threshold and value-bin methods are retained only as heuristic baselines.",
            "Accepted-subset metrics are explicitly empirical for the heuristic baselines.",
        ],
        "outcome": {
            "datasets": int(df["dataset"].nunique()) if not df.empty else 0,
            "methods": ", ".join(sorted(df["method"].unique())) if not df.empty else "",
            "best_accepted_mae": float(df["accepted_mae"].min()) if not df.empty else float("nan"),
        },
    }
    write_csv_json_md("scenario_k_mondrian_regression", df, meta)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    arguments = parser.parse_args()
    run(RunConfig(seed=42, quick=arguments.quick))
