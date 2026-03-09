"""Scenario G: multi-dataset threshold-based regression reject evaluation."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .common_reject import (
    RunConfig,
    accepted_interval_metrics,
    build_regression_bundle,
    quantile_grid,
    regression_mse,
    task_specs,
    write_csv_json_md,
)


def run(config: RunConfig) -> None:
    """Measure empirical accepted-subset behavior under threshold-based reject."""
    rows: list[dict[str, float | str | int]] = []
    confidences = (0.90, 0.95)
    quantiles = tuple(float(q) for q in quantile_grid(config.quick))

    for spec in task_specs("regression", quick=config.quick):
        for threshold_quantile in quantiles:
            bundle = build_regression_bundle(spec, config, seed_offset=int(threshold_quantile * 1000))
            threshold = float(np.quantile(bundle.y_cal, threshold_quantile))
            bundle.wrapper.initialize_reject_learner(threshold=threshold)
            for confidence in confidences:
                breakdown = bundle.wrapper.explainer.reject_orchestrator.predict_reject_breakdown(
                    bundle.x_test,
                    confidence=float(confidence),
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
                        "confidence": float(confidence),
                        "threshold_quantile": threshold_quantile,
                        "n_cal": int(len(bundle.x_cal)),
                        "n_test": int(len(bundle.x_test)),
                        "interval_coverage_all": 1.0
                        - float(
                            np.mean(
                                (bundle.y_test < bundle.baseline_low)
                                | (bundle.y_test > bundle.baseline_high)
                            )
                        ),
                        "accepted_coverage_empirical": metrics["accepted_coverage"],
                        "interval_width_all": float(np.mean(bundle.baseline_high - bundle.baseline_low)),
                        "accepted_interval_width_empirical": metrics["accepted_interval_width"],
                        "mse_all": regression_mse(bundle.y_test, bundle.baseline_pred),
                        "accepted_mse_empirical": metrics["accepted_mse"],
                        "reject_rate": float(breakdown["reject_rate"]),
                    }
                )

    df = pd.DataFrame(rows)
    meta = {
        "scenario": "scenario_g_regression_coverage",
        "display_name": "Scenario G — Threshold regression empirical coverage",
        "quick": config.quick,
        "highlights": [
            "Accepted-subset coverage is reported as an empirical quantity only.",
            "Threshold-based regression reject remains explicitly heuristic in this suite.",
            "Both interval width and MSE are tracked on the accepted subset to capture the trade-off.",
        ],
        "outcome": {
            "datasets": int(df["dataset"].nunique()) if not df.empty else 0,
            "mean_reject_rate": float(df["reject_rate"].mean()) if not df.empty else float("nan"),
            "mean_accepted_mse_empirical": float(df["accepted_mse_empirical"].mean()) if not df.empty else float("nan"),
        },
    }
    write_csv_json_md("scenario_g_regression_coverage", df, meta)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    arguments = parser.parse_args()
    run(RunConfig(seed=42, quick=arguments.quick))
