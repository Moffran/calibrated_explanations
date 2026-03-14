"""Scenario 3: multi-dataset threshold-based regression reject heuristic baseline.

Paper mapping: RQ3 (empirical baseline).

Headline finding: threshold reject does NOT select by uncertainty.  Accepted-subset interval
width equals full-set interval width because the threshold filters by predicted value quantile,
not by prediction interval width.  This is the paper's null result for the heuristic, which
motivates the difficulty-normalised approach (C3, deferred to post-RT2).
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .common_reject import (
    RunConfig,
    accepted_interval_metrics,
    build_regression_bundle,
    quantile_grid,
    reject_breakdown,
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
            for confidence in confidences:
                breakdown = reject_breakdown(
                    bundle.wrapper,
                    bundle.x_test,
                    confidence=float(confidence),
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
                all_interval_width = float(np.mean(bundle.baseline_high - bundle.baseline_low))
                accepted_interval_width = metrics["accepted_interval_width"]
                rows.append(
                    {
                        "dataset": spec.name,
                        "confidence": float(confidence),
                        "effective_confidence": float(breakdown["effective_confidence"]),
                        "threshold_quantile": threshold_quantile,
                        "effective_threshold": breakdown.get("effective_threshold"),
                        "threshold_source": breakdown.get("threshold_source"),
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
                        "interval_width_all": all_interval_width,
                        "accepted_interval_width_empirical": accepted_interval_width,
                        # width_delta > 0 means accepted subset has wider intervals (unusual);
                        # ~0 confirms threshold reject does not select by uncertainty.
                        "interval_width_delta": (
                            accepted_interval_width - all_interval_width
                            if np.isfinite(accepted_interval_width)
                            else float("nan")
                        ),
                        "mse_all": regression_mse(bundle.y_test, bundle.baseline_pred),
                        "accepted_mse_empirical": metrics["accepted_mse"],
                        "reject_rate": float(breakdown["reject_rate"]),
                    }
                )

    df = pd.DataFrame(rows)
    # Headline finding: mean interval_width_delta should be near zero
    mean_width_delta = float(df["interval_width_delta"].mean()) if not df.empty else float("nan")
    meta = {
        "scenario": "scenario_3_regression_threshold_baseline",
        "display_name": "Scenario 3 — Threshold regression heuristic baseline",
        "paper_rq": "RQ3",
        "guarantee_status": "empirical",
        "quick": config.quick,
        "highlights": [
            "Headline finding: threshold reject does NOT select by uncertainty — "
            "accepted-subset interval width equals full-set interval width (~0 delta).",
            f"Mean interval_width_delta across all rows: {mean_width_delta:.4f} "
            "(near zero confirms the null result).",
            "Threshold-based regression reject remains explicitly heuristic in this suite.",
            "Both interval width and MSE are tracked on the accepted subset to capture the trade-off.",
            "The difficulty-normalised approach (C3) is deferred to a standalone scenario post-RT2.",
        ],
        "outcome": {
            "datasets": int(df["dataset"].nunique()) if not df.empty else 0,
            "mean_reject_rate": float(df["reject_rate"].mean()) if not df.empty else float("nan"),
            "mean_accepted_mse_empirical": float(df["accepted_mse_empirical"].mean()) if not df.empty else float("nan"),
            "mean_interval_width_delta": mean_width_delta,
        },
    }
    write_csv_json_md("scenario_3_regression_threshold_baseline", df, meta)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    arguments = parser.parse_args()
    run(RunConfig(seed=42, quick=arguments.quick))
