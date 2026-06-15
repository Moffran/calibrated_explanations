"""Scenario 1: multi-dataset binary marginal coverage sweep.

Paper mapping: C1 / RQ1 (formal target).

Contribution C1 / Proposition 1: blended NCF preserves ICP marginal coverage at level 1-epsilon
across the binary benchmark.  This scenario verifies the formal guarantee empirically and
distinguishes finite-sample CI-lower-bound failures from structural (unconditional) violations.

Uses the same 5-seed grid as Scenarios 8-12 so violation counts are stable across runs and
comparable across scenarios.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .common_reject import (
    RunConfig,
    _markdown_table_from_df,
    accepted_accuracy,
    build_classification_bundle,
    clopper_pearson_interval,
    empirical_coverage,
    reject_breakdown,
    seed_grid,
    singleton_precision_recall,
    task_specs,
    write_csv_json_md,
)


def run(config: RunConfig) -> None:
    """Measure binary label-set coverage and accepted accuracy across datasets and seeds."""
    epsilons = (0.01, 0.05, 0.10)
    rows: list[dict[str, float | str | int | bool]] = []

    for spec in task_specs("binary", quick=config.quick):
        for seed_offset in seed_grid(config):
            bundle = build_classification_bundle(spec, config, seed_offset=seed_offset)
            for epsilon in epsilons:
                confidence = 1.0 - float(epsilon)
                breakdown = reject_breakdown(bundle.wrapper, bundle.x_test, confidence=confidence)
                prediction_set = np.asarray(breakdown["prediction_set"], dtype=bool)
                rejected = np.asarray(breakdown["rejected"], dtype=bool)
                accepted = ~rejected
                coverage = empirical_coverage(prediction_set, bundle.y_test)
                singleton_metrics = singleton_precision_recall(prediction_set, bundle.y_test)
                successes = int(
                    np.sum(prediction_set[np.arange(len(bundle.y_test)), bundle.y_test])
                )
                lower_ci, upper_ci = clopper_pearson_interval(successes, len(bundle.y_test))
                # A structural violation is when the CI upper bound is below 1-epsilon:
                # finite-sample noise cannot explain the shortfall.
                structural_violation = bool(upper_ci < 1.0 - epsilon)
                rows.append(
                    {
                        "dataset": spec.name,
                        "seed": int(config.seed + seed_offset),
                        "epsilon": float(epsilon),
                        "n_cal": int(len(bundle.x_cal)),
                        "n_test": int(len(bundle.x_test)),
                        "coverage": coverage,
                        "lower_ci": lower_ci,
                        "upper_ci": upper_ci,
                        "violation": bool(coverage < 1.0 - epsilon),
                        "structural_violation": structural_violation,
                        "reject_rate": float(breakdown["reject_rate"]),
                        **singleton_metrics,
                        "accepted_accuracy_empirical": accepted_accuracy(
                            bundle.y_test,
                            bundle.baseline_pred,
                            accepted,
                        ),
                    }
                )

    df = pd.DataFrame(rows)
    violation_count = int(df["violation"].sum()) if not df.empty else 0
    structural_count = int(df["structural_violation"].sum()) if not df.empty else 0
    violation_rate = float(df["violation"].mean()) if not df.empty else float("nan")
    n_seeds = int(df["seed"].nunique()) if not df.empty else 0
    meta = {
        "scenario": "scenario_1_binary_coverage",
        "display_name": "Scenario 1 — Binary marginal coverage sweep",
        "paper_contribution": "C1",
        "paper_rq": "RQ1",
        "guarantee_status": "formal_target",
        "quick": config.quick,
        "highlights": [
            "Coverage is reported as standard label-set coverage from the conformal prediction sets.",
            "Accepted accuracy is included as a separate empirical metric and not treated as the conformal guarantee.",
            f"Observed coverage violations: {violation_count}/{len(df)} ({violation_rate:.4f}).",
            f"Structural violations (CI upper bound < 1-epsilon, cannot be attributed to finite-sample noise): "
            f"{structural_count}/{len(df)}.",
            f"Uses {n_seeds}-seed grid matching Scenarios 8-12 for stable violation counts.",
        ],
        "outcome": {
            "datasets": int(df["dataset"].nunique()) if not df.empty else 0,
            "seeds": n_seeds,
            "rows": int(len(df)),
            "violation_rate": violation_rate,
            "structural_violations": structural_count,
            "mean_coverage": float(df["coverage"].mean()) if not df.empty else float("nan"),
        },
    }
    # --- Extra sections ---
    extra_sections: list[str] = []

    # Section: Coverage by epsilon
    if not df.empty:
        by_eps = (
            df.groupby("epsilon")
            .agg(
                n_rows=("coverage", "size"),
                violations=("violation", "sum"),
                structural_violations=("structural_violation", "sum"),
                violation_rate=("violation", "mean"),
                mean_coverage=("coverage", "mean"),
                mean_reject_rate=("reject_rate", "mean"),
            )
            .reset_index()
        )
        extra_sections += [
            "## Coverage by epsilon",
            "",
            _markdown_table_from_df(by_eps),
            "",
        ]

        # Section: All datasets by structural violations (sorted desc; shows 0-violation datasets too)
        by_dataset = (
            df.groupby("dataset")
            .agg(
                structural_violations=("structural_violation", "sum"),
                violations=("violation", "sum"),
                mean_coverage=("coverage", "mean"),
                mean_reject_rate=("reject_rate", "mean"),
            )
            .reset_index()
            .sort_values("structural_violations", ascending=False)
        )
        extra_sections += [
            "## All datasets — structural violations",
            "",
            _markdown_table_from_df(by_dataset),
            "",
        ]

    write_csv_json_md("scenario_1_binary_coverage", df, meta, extra_sections=extra_sections)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    arguments = parser.parse_args()
    run(RunConfig(seed=42, quick=arguments.quick))
