"""Scenario E: multi-dataset binary coverage sweep."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .common_reject import (
    RunConfig,
    accepted_accuracy,
    build_classification_bundle,
    clopper_pearson_interval,
    empirical_coverage,
    reject_breakdown,
    task_specs,
    write_csv_json_md,
)


def run(config: RunConfig) -> None:
    """Measure binary label-set coverage and accepted accuracy across datasets."""
    epsilons = (0.01, 0.05, 0.10)
    rows: list[dict[str, float | str | int]] = []

    for spec in task_specs("binary", quick=config.quick):
        bundle = build_classification_bundle(spec, config)
        for epsilon in epsilons:
            confidence = 1.0 - float(epsilon)
            breakdown = reject_breakdown(bundle.wrapper, bundle.x_test, confidence=confidence)
            prediction_set = np.asarray(breakdown["prediction_set"], dtype=bool)
            rejected = np.asarray(breakdown["rejected"], dtype=bool)
            accepted = ~rejected
            coverage = empirical_coverage(prediction_set, bundle.y_test)
            successes = int(np.sum(prediction_set[np.arange(len(bundle.y_test)), bundle.y_test]))
            lower_ci, upper_ci = clopper_pearson_interval(successes, len(bundle.y_test))
            rows.append(
                {
                    "dataset": spec.name,
                    "epsilon": float(epsilon),
                    "n_cal": int(len(bundle.x_cal)),
                    "n_test": int(len(bundle.x_test)),
                    "coverage": coverage,
                    "lower_ci": lower_ci,
                    "upper_ci": upper_ci,
                    "violation": bool(coverage < 1.0 - epsilon),
                    "reject_rate": float(breakdown["reject_rate"]),
                    "accepted_accuracy_empirical": accepted_accuracy(
                        bundle.y_test,
                        bundle.baseline_pred,
                        accepted,
                    ),
                }
            )

    df = pd.DataFrame(rows)
    violation_rate = float(df["violation"].mean()) if not df.empty else float("nan")
    meta = {
        "scenario": "scenario_e_binary_coverage_sweep",
        "display_name": "Scenario E — Binary coverage sweep",
        "quick": config.quick,
        "highlights": [
            "Coverage is reported as standard label-set coverage from the conformal prediction sets.",
            "Accepted accuracy is included as a separate empirical metric and not treated as the conformal guarantee.",
            f"Observed coverage violations: {int(df['violation'].sum())}/{len(df)} ({violation_rate:.4f}).",
        ],
        "outcome": {
            "datasets": int(df["dataset"].nunique()) if not df.empty else 0,
            "rows": int(len(df)),
            "violation_rate": violation_rate,
            "mean_coverage": float(df["coverage"].mean()) if not df.empty else float("nan"),
        },
    }
    write_csv_json_md("scenario_e_binary_coverage", df, meta)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    arguments = parser.parse_args()
    run(RunConfig(seed=42, quick=arguments.quick))
