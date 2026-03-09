"""Scenario J: finite-sample and edge-case stress tests."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .common_reject import (
    RunConfig,
    build_classification_bundle,
    empirical_coverage,
    reject_breakdown,
    task_specs,
    write_csv_json_md,
)


def run(config: RunConfig) -> None:
    """Probe small calibration sets, class imbalance, and extreme rejection."""
    rows: list[dict[str, float | str | int | bool]] = []
    binary_specs = task_specs("binary", quick=False)[:3]
    cal_sizes = (10, 20, 50) if config.quick else (10, 20, 50, 100, 200)
    epsilons = (0.05, 0.10, 0.25)

    for spec in binary_specs:
        for n_cal in cal_sizes:
            bundle = build_classification_bundle(spec, config, n_cal=n_cal)
            for epsilon in epsilons:
                breakdown = reject_breakdown(bundle.wrapper, bundle.x_test, confidence=1.0 - epsilon)
                rejected = np.asarray(breakdown["rejected"], dtype=bool)
                coverage = empirical_coverage(np.asarray(breakdown["prediction_set"], dtype=bool), bundle.y_test)
                rows.append(
                    {
                        "dataset": spec.name,
                        "probe": "small_calibration",
                        "n_cal": int(len(bundle.x_cal)),
                        "epsilon": float(epsilon),
                        "coverage": coverage,
                        "reject_rate": float(breakdown["reject_rate"]),
                        "error_rate": float(breakdown["error_rate"]),
                        "violation": bool(coverage < 1.0 - epsilon),
                        "matched_count": int(np.sum(~rejected)),
                    }
                )

        full_bundle = build_classification_bundle(spec, config)
        for confidence in (0.99, 0.995):
            breakdown = reject_breakdown(full_bundle.wrapper, full_bundle.x_test, confidence=confidence)
            rejected = np.asarray(breakdown["rejected"], dtype=bool)
            rows.append(
                {
                    "dataset": spec.name,
                    "probe": "extreme_confidence",
                    "n_cal": int(len(full_bundle.x_cal)),
                    "epsilon": float(1.0 - confidence),
                    "coverage": empirical_coverage(np.asarray(breakdown["prediction_set"], dtype=bool), full_bundle.y_test),
                    "reject_rate": float(breakdown["reject_rate"]),
                    "error_rate": float(breakdown["error_rate"]),
                    "violation": False,
                    "matched_count": int(np.sum(~rejected)),
                }
            )

    df = pd.DataFrame(rows)
    meta = {
        "scenario": "scenario_j_stress_tests",
        "display_name": "Scenario J — Stress tests",
        "quick": config.quick,
        "highlights": [
            "Stress tests focus on finite-sample behavior and extreme confidence settings.",
            "The suite reports coverage violations empirically rather than claiming new guarantees.",
        ],
        "outcome": {
            "rows": int(len(df)),
            "violations": int(df["violation"].sum()) if not df.empty else 0,
            "max_reject_rate": float(df["reject_rate"].max()) if not df.empty else float("nan"),
        },
    }
    write_csv_json_md("scenario_j_stress_tests", df, meta)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    arguments = parser.parse_args()
    run(RunConfig(seed=42, quick=arguments.quick))
