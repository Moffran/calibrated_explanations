"""Scenario 7: NCF coverage validity sweep (supplementary, empirical companion to Proposition 1).

Paper mapping: C1 / Proposition 1 companion (empirical, not formal).

Proposition 1 proves blended NCF preserves ICP marginal coverage at level 1-epsilon for any
fixed w in (0,1].  This scenario verifies that guarantee holds across the full (NCF, w, epsilon)
grid on the binary benchmark, and flags any cell that violates coverage.

This scenario is supplementary — it may yield misleading results until the RT-2
sigma-normalisation bug is resolved in the core CE library.  Run with --supplementary flag.

Grid:
  NCFs    : hinge, margin, ensured
  w       : 0.3, 0.5, 0.7, 1.0
  epsilon : 0.05, 0.10
  datasets: all binary benchmark datasets
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from calibrated_explanations import RejectPolicySpec

from .common_reject import (
    RunConfig,
    build_classification_bundle,
    clopper_pearson_interval,
    empirical_coverage,
    task_specs,
    write_csv_json_md,
)

_NCFS = ("hinge", "margin", "ensured")
_W_VALUES = (0.3, 0.5, 0.7, 1.0)
_EPSILONS = (0.05, 0.10)


def run(config: RunConfig) -> None:
    """Sweep (NCF, w, epsilon) grid and flag coverage violations on binary datasets."""
    rows: list[dict[str, float | str | int | bool]] = []

    for spec in task_specs("binary", quick=config.quick):
        bundle = build_classification_bundle(spec, config)
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
                    # Build a pseudo prediction-set from the accepted subset for coverage
                    # calculation: coverage is measured on the full test set prediction set,
                    # not on the accepted subset alone.
                    prediction_set_attr = getattr(result, "prediction_set", None)
                    if prediction_set_attr is not None:
                        prediction_set = np.asarray(prediction_set_attr, dtype=bool)
                        coverage = empirical_coverage(prediction_set, bundle.y_test)
                        successes = int(
                            np.sum(prediction_set[np.arange(len(bundle.y_test)), bundle.y_test])
                        )
                        lower_ci, upper_ci = clopper_pearson_interval(successes, len(bundle.y_test))
                    else:
                        coverage = float("nan")
                        lower_ci, upper_ci = float("nan"), float("nan")

                    rows.append(
                        {
                            "dataset": spec.name,
                            "ncf": ncf,
                            "w": float(w),
                            "epsilon": float(epsilon),
                            "n_cal": int(len(bundle.x_cal)),
                            "n_test": int(len(bundle.x_test)),
                            "coverage": coverage,
                            "lower_ci": lower_ci,
                            "upper_ci": upper_ci,
                            "violation": bool(coverage < 1.0 - epsilon) if np.isfinite(coverage) else False,
                            "structural_violation": bool(
                                np.isfinite(upper_ci) and upper_ci < 1.0 - epsilon
                            ),
                            "accept_rate": float(np.mean(accepted)),
                        }
                    )

    df = pd.DataFrame(rows)
    violations_by_ncf_w = (
        df.groupby(["ncf", "w"])["violation"].sum().to_dict()
        if not df.empty
        else {}
    )
    meta = {
        "scenario": "scenario_7_ncf_coverage_validity",
        "display_name": "Scenario 7 — NCF coverage validity sweep (supplementary)",
        "paper_contribution": "C1",
        "guarantee_status": "empirical_companion_to_proposition_1",
        "supplementary": True,
        "quick": config.quick,
        "highlights": [
            "SUPPLEMENTARY — may yield misleading results before the RT-2 fix.",
            "Empirical companion to Proposition 1: coverage >= 1-epsilon across (NCF, w, epsilon) grid.",
            "structural_violation: CI upper bound < 1-epsilon; cannot be attributed to finite-sample noise.",
        ],
        "outcome": {
            "datasets": int(df["dataset"].nunique()) if not df.empty else 0,
            "total_violations": int(df["violation"].sum()) if not df.empty else 0,
            "structural_violations": int(df["structural_violation"].sum()) if not df.empty else 0,
            "violations_by_ncf_w": {str(k): int(v) for k, v in violations_by_ncf_w.items()},
        },
    }
    write_csv_json_md("scenario_7_ncf_coverage_validity", df, meta)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    arguments = parser.parse_args()
    run(RunConfig(seed=42, quick=arguments.quick))
