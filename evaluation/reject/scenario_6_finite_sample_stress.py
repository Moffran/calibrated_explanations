"""Scenario 6: finite-sample and edge-case stress tests.

Paper mapping: RQ6 (empirical).

Bugs fixed vs old Scenario J:
  1. extreme_confidence rows previously hard-coded `violation=False` unconditionally.
     Now coverage is actually computed and compared to 1-epsilon.
  2. `task_specs("binary", quick=False)[:3]` ignored the config.quick flag.
     Now dataset slice follows config.quick: 2 datasets in quick mode, 3 in full mode.
"""

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
    """Probe small calibration sets and extreme rejection confidence."""
    rows: list[dict[str, float | str | int | bool]] = []
    # Respect quick flag for dataset selection
    n_datasets = 2 if config.quick else 3
    binary_specs = task_specs("binary", quick=config.quick)[:n_datasets]
    cal_sizes = (10, 20, 50) if config.quick else (10, 20, 50, 100, 200)
    epsilons = (0.05, 0.10, 0.25)

    for spec in binary_specs:
        # --- Small calibration set probe ---
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
                        # Actual coverage check — not a hard-coded constant
                        "violation": bool(coverage < 1.0 - epsilon),
                        "matched_count": int(np.sum(~rejected)),
                    }
                )

        # --- Extreme confidence probe ---
        full_bundle = build_classification_bundle(spec, config)
        for confidence in (0.99, 0.995):
            epsilon = 1.0 - confidence
            breakdown = reject_breakdown(full_bundle.wrapper, full_bundle.x_test, confidence=confidence)
            rejected = np.asarray(breakdown["rejected"], dtype=bool)
            coverage = empirical_coverage(
                np.asarray(breakdown["prediction_set"], dtype=bool), full_bundle.y_test
            )
            rows.append(
                {
                    "dataset": spec.name,
                    "probe": "extreme_confidence",
                    "n_cal": int(len(full_bundle.x_cal)),
                    "epsilon": float(epsilon),
                    "coverage": coverage,
                    "reject_rate": float(breakdown["reject_rate"]),
                    "error_rate": float(breakdown["error_rate"]),
                    # Bug fix: actual coverage check, not unconditional False
                    "violation": bool(coverage < 1.0 - epsilon),
                    "matched_count": int(np.sum(~rejected)),
                }
            )

    df = pd.DataFrame(rows)
    meta = {
        "scenario": "scenario_6_finite_sample_stress",
        "display_name": "Scenario 6 — Finite-sample stress tests",
        "paper_rq": "RQ6",
        "guarantee_status": "empirical",
        "quick": config.quick,
        "highlights": [
            "Stress tests focus on finite-sample behavior and extreme confidence settings.",
            "The suite reports coverage violations empirically rather than claiming new guarantees.",
            "violation is computed from actual coverage, not hard-coded.",
            "extreme_confidence probe uses the same violation logic as small_calibration.",
        ],
        "outcome": {
            "rows": int(len(df)),
            "violations": int(df["violation"].sum()) if not df.empty else 0,
            "max_reject_rate": float(df["reject_rate"].max()) if not df.empty else float("nan"),
            "small_cal_violations": int(
                df[df["probe"] == "small_calibration"]["violation"].sum()
            ) if not df.empty else 0,
            "extreme_conf_violations": int(
                df[df["probe"] == "extreme_confidence"]["violation"].sum()
            ) if not df.empty else 0,
        },
    }
    write_csv_json_md("scenario_6_finite_sample_stress", df, meta)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    arguments = parser.parse_args()
    run(RunConfig(seed=42, quick=arguments.quick))
