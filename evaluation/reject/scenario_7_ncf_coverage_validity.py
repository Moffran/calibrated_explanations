"""Scenario 7: NCF coverage validity sweep.

Paper mapping: C1 / Proposition 1 companion (empirical, not formal).

Proposition 1 applies when the conformal score path satisfies the paper's
assumptions. This scenario is a supplementary empirical diagnostic for the
implemented score and prediction-set path; it is not itself a proof of validity.

Grid:
  NCFs    : default, ensured
  w       : 0.3, 0.5, 0.7, 1.0
  epsilon : 0.05, 0.10
  seeds   : seed_grid(config)
  datasets: all binary benchmark datasets
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import pandas as pd

from calibrated_explanations import RejectPolicySpec

from .common_reject import (
    RunConfig,
    _markdown_table_from_df,
    build_classification_bundle,
    clopper_pearson_interval,
    empirical_coverage,
    seed_grid,
    singleton_precision_recall,
    task_specs,
    write_csv_json_md,
)

_NCFS = ("default", "ensured")
_W_VALUES = (0.3, 0.5, 0.7, 1.0)
_EPSILONS = (0.05, 0.10)


def run(config: RunConfig) -> None:
    """Sweep (NCF, w, epsilon) grid and flag empirical coverage violations."""
    rows: list[dict[str, Any]] = []

    for spec in task_specs("binary", quick=config.quick):
        for seed_offset in seed_grid(config):
            seed = int(config.seed + seed_offset)
            bundle = build_classification_bundle(spec, config, seed_offset=seed_offset)
            for ncf in _NCFS:
                for w in _W_VALUES:
                    for epsilon in _EPSILONS:
                        confidence = 1.0 - float(epsilon)
                        result = bundle.wrapper.predict(
                            bundle.x_test,
                            reject_policy=RejectPolicySpec.flag(ncf=ncf, w=w),
                            confidence=confidence,
                        )
                        rejected = np.asarray(result.rejected, dtype=bool)
                        accepted = ~rejected

                        # FLAG stores prediction sets on the legacy metadata envelope.
                        prediction_set_attr = (
                            result.metadata.get("prediction_set") if result.metadata else None
                        )
                        coverage_defined = prediction_set_attr is not None
                        if coverage_defined:
                            prediction_set = np.asarray(prediction_set_attr, dtype=bool)
                            set_sizes = np.sum(prediction_set, axis=1)
                            coverage = empirical_coverage(prediction_set, bundle.y_test)
                            singleton_metrics = singleton_precision_recall(
                                prediction_set,
                                bundle.y_test,
                            )
                            successes = int(
                                np.sum(
                                    prediction_set[
                                        np.arange(len(bundle.y_test)),
                                        bundle.y_test,
                                    ]
                                )
                            )
                            lower_ci, upper_ci = clopper_pearson_interval(
                                successes,
                                len(bundle.y_test),
                            )
                        else:
                            coverage = float("nan")
                            lower_ci, upper_ci = float("nan"), float("nan")
                            set_sizes = np.full(len(bundle.y_test), -1, dtype=int)
                            singleton_metrics = singleton_precision_recall(None, bundle.y_test)

                        baseline_accuracy = float(
                            np.mean(np.asarray(bundle.baseline_pred) == bundle.y_test)
                        )
                        singleton_rate = (
                            float(np.mean(set_sizes == 1)) if coverage_defined else float("nan")
                        )
                        ambiguity_rate = (
                            float(np.mean(set_sizes >= 2)) if coverage_defined else float("nan")
                        )
                        novelty_rate = (
                            float(np.mean(set_sizes == 0)) if coverage_defined else float("nan")
                        )
                        mean_prediction_set_size = (
                            float(np.mean(set_sizes)) if coverage_defined else float("nan")
                        )

                        rows.append(
                            {
                                "dataset": spec.name,
                                "seed": seed,
                                "ncf": ncf,
                                "w": float(w),
                                "epsilon": float(epsilon),
                                "n_cal": int(len(bundle.x_cal)),
                                "n_test": int(len(bundle.x_test)),
                                "coverage_defined": bool(coverage_defined),
                                "coverage": coverage,
                                "baseline_accuracy": baseline_accuracy,
                                "coverage_lift_over_baseline_accuracy": (
                                    float(coverage - baseline_accuracy)
                                    if np.isfinite(coverage)
                                    else float("nan")
                                ),
                                "lower_ci": lower_ci,
                                "upper_ci": upper_ci,
                                "violation": bool(coverage < 1.0 - epsilon)
                                if np.isfinite(coverage)
                                else False,
                                "structural_violation": bool(
                                    np.isfinite(upper_ci) and upper_ci < 1.0 - epsilon
                                ),
                                "accept_rate": float(np.mean(accepted)),
                                "mean_prediction_set_size": mean_prediction_set_size,
                                "singleton_rate": singleton_rate,
                                "ambiguity_rate": ambiguity_rate,
                                "novelty_rate": novelty_rate,
                                **singleton_metrics,
                            }
                        )

    df = pd.DataFrame(rows)
    violations_by_ncf_w = (
        df.groupby(["ncf", "w"])["violation"].sum().to_dict()
        if not df.empty
        else {}
    )
    structural_by_ncf_w = (
        df.groupby(["ncf", "w"])["structural_violation"].sum().to_dict()
        if not df.empty
        else {}
    )
    independent_groups = (
        df.groupby(["dataset", "seed", "ncf", "epsilon"])["structural_violation"].any()
        if not df.empty
        else pd.Series(dtype=bool)
    )
    independent_violation_groups = (
        df.groupby(["dataset", "seed", "ncf", "epsilon"])["violation"].any()
        if not df.empty
        else pd.Series(dtype=bool)
    )
    violation_count = int(df["violation"].sum()) if not df.empty else 0
    structural_count = int(df["structural_violation"].sum()) if not df.empty else 0
    independent_group_count = int(len(independent_groups))
    independent_violation_count = int(independent_violation_groups.sum())
    independent_structural_count = int(independent_groups.sum())
    hard_dataset_summary = (
        df.groupby("dataset")
        .agg(
            structural=("structural_violation", "sum"),
            mean_coverage=("coverage", "mean"),
            mean_accept_rate=("accept_rate", "mean"),
            mean_singleton_rate=("singleton_rate", "mean"),
        )
        .sort_values(["structural", "mean_coverage"], ascending=[False, True])
        .head(8)
        .reset_index()
        if not df.empty
        else pd.DataFrame()
    )
    mean_by_ncf_epsilon = (
        df.groupby(["ncf", "epsilon"])
        .agg(
            mean_coverage=("coverage", "mean"),
            mean_baseline_accuracy=("baseline_accuracy", "mean"),
            mean_accept_rate=("accept_rate", "mean"),
            mean_singleton_rate=("singleton_rate", "mean"),
            structural_violations=("structural_violation", "sum"),
        )
        .reset_index()
        if not df.empty
        else pd.DataFrame()
    )
    high_accept_structural = int(
        ((df["structural_violation"]) & (df["accept_rate"] >= 0.95)).sum()
    ) if not df.empty else 0

    meta = {
        "scenario": "scenario_7_ncf_coverage_validity",
        "display_name": "Scenario 7 - NCF coverage validity sweep (supplementary)",
        "paper_contribution": "C1",
        "guarantee_status": "empirical_companion_to_proposition_1",
        "supplementary": True,
        "quick": config.quick,
        "highlights": [
            "SUPPLEMENTARY empirical diagnostic; not a standalone proof of conformal validity.",
            "Coverage is measured from prediction sets stored in result.metadata['prediction_set'].",
            f"Observed row-level coverage violations: {violation_count}/{len(df)}.",
            f"Observed row-level structural violations: {structural_count}/{len(df)}.",
            (
                "The dominant tendency is singleton collapse on harder datasets: when "
                "accept_rate/singleton_rate is high, prediction-set coverage tracks ordinary "
                "baseline accuracy rather than gaining much from ambiguity sets."
            ),
            (
                f"High-accept structural rows (accept_rate >= 0.95): "
                f"{high_accept_structural}/{structural_count}."
            ),
            (
                "Collapsed by (dataset, seed, ncf, epsilon), structural violations are "
                f"{independent_structural_count}/{independent_group_count}; this avoids "
                "over-reading repeated w rows for default NCF."
            ),
            (
                "structural_violation means the Clopper-Pearson upper bound is below "
                "1-epsilon in this finite test batch; it is strong diagnostic evidence, "
                "not a separate theorem."
            ),
        ],
        "outcome": {
            "datasets": int(df["dataset"].nunique()) if not df.empty else 0,
            "seeds": int(df["seed"].nunique()) if not df.empty else 0,
            "rows": int(len(df)),
            "coverage_defined_count": int(df["coverage_defined"].sum()) if not df.empty else 0,
            "coverage_undefined_count": int((~df["coverage_defined"]).sum())
            if not df.empty
            else 0,
            "total_violations": violation_count,
            "structural_violations": structural_count,
            "independent_condition_groups": independent_group_count,
            "independent_total_violations": independent_violation_count,
            "independent_structural_violations": independent_structural_count,
            "high_accept_structural_violations": high_accept_structural,
            "mean_by_ncf_epsilon": mean_by_ncf_epsilon.to_dict(orient="records"),
            "top_structural_datasets": hard_dataset_summary.to_dict(orient="records"),
            "structural_violations_by_ncf_w": {
                str(k): int(v) for k, v in structural_by_ncf_w.items()
            },
            "violations_by_ncf_w": {str(k): int(v) for k, v in violations_by_ncf_w.items()},
        },
    }
    # --- Extra sections ---
    extra_sections: list[str] = []

    if not df.empty:
        # Section: Coverage by NCF and epsilon
        by_ncf_eps = (
            df.groupby(["ncf", "epsilon"])
            .agg(
                mean_coverage=("coverage", "mean"),
                violation_rate=("violation", "mean"),
                structural_violation_rate=("structural_violation", "mean"),
                mean_accept_rate=("accept_rate", "mean"),
            )
            .reset_index()
        )
        extra_sections += [
            "## Coverage by NCF and epsilon",
            "",
            _markdown_table_from_df(by_ncf_eps),
            "",
        ]

        # Section: All datasets by structural violations (shows clean datasets too)
        by_dataset = (
            df.groupby("dataset")
            .agg(
                structural_violations=("structural_violation", "sum"),
                violations=("violation", "sum"),
                mean_coverage=("coverage", "mean"),
                mean_accept_rate=("accept_rate", "mean"),
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

    write_csv_json_md("scenario_7_ncf_coverage_validity", df, meta, extra_sections=extra_sections)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    arguments = parser.parse_args()
    run(RunConfig(seed=42, quick=arguments.quick))
