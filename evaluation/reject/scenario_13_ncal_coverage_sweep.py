"""Scenario 13: n_cal sweep — arm A vs arm C structural coverage violations.

Paper mapping: supplementary (RT-3 follow-up).

Anomaly 1 from the red-team analysis (see reject_evaluation_suite_report.md) found
that arm C (experimental.difficulty_normalized) has 77% more structural coverage
violations than arm A (builtin.default) in Scenario 12 across 5 seeds and 26
binary datasets.

Hypothesis: difficulty normalization inflates the variance of the calibration score
distribution.  Higher variance degrades the empirical quantile estimate for small
calibration sets, producing finite-sample violations even though the formal coverage
guarantee holds asymptotically.  If the hypothesis is correct, structural violation
rates should DECREASE as n_cal grows and converge to arm A rates at large n_cal.

This scenario sweeps n_cal in {50, 100, 200, 400} with 5 seeds per dataset, tracking
structural violation rates for both arms at each n_cal level.  The key diagnostic is
whether arm C violation rates drop as n_cal increases.  If they do not, there is
evidence of a genuine exchangeability violation that requires a redesign.

Grid:
  n_cal   : 50, 100, 200, 400  (clipped to available data per dataset)
  epsilons: 0.05, 0.10
  seeds   : seed_grid (5)
  datasets: all binary benchmark datasets
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from calibrated_explanations import RejectPolicySpec, WrapCalibratedExplainer

from .common_reject import (
    RunConfig,
    _markdown_table_from_df,
    clopper_pearson_interval,
    empirical_coverage,
    load_dataset,
    seed_grid,
    singleton_precision_recall,
    split_dataset,
    task_specs,
    write_csv_json_md,
)

_W = 0.5
_N_CAL_TARGETS = (50, 100, 200, 400)
_EPSILONS = (0.05, 0.10)


@dataclass
class _DeterministicDifficultyEstimator:
    """Minimal deterministic estimator: Euclidean distance from training centroid."""

    center_: np.ndarray
    scale_: np.ndarray
    fitted: bool = True

    @classmethod
    def fit(cls, x: np.ndarray) -> _DeterministicDifficultyEstimator:
        x_arr = np.asarray(x, dtype=float)
        center = np.mean(x_arr, axis=0)
        scale = np.std(x_arr, axis=0)
        scale = np.where(np.isfinite(scale) & (scale > 1e-9), scale, 1.0)
        return cls(center_=center, scale_=scale, fitted=True)

    def apply(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        normalized = (x_arr - self.center_) / self.scale_
        squared_radius = np.mean(np.square(normalized), axis=1)
        scores = 1.0 + np.sqrt(np.maximum(squared_radius, 0.0))
        return np.where(np.isfinite(scores) & (scores > 0.0), scores, 1.0).astype(float)


def run(config: RunConfig) -> None:
    """Sweep n_cal and measure structural coverage violation rates for arm A and arm C."""
    rows: list[dict[str, Any]] = []

    for spec in task_specs("binary", quick=config.quick):
        _, x_all, y_all, feature_names = load_dataset(spec)

        for seed_offset in seed_grid(config):
            seed = int(config.seed + seed_offset)

            for n_cal_target in _N_CAL_TARGETS:
                x_fit, x_cal, x_test, y_fit, y_cal, y_test = split_dataset(
                    x_all, y_all, seed=seed, stratify=True, n_cal=n_cal_target
                )
                actual_n_cal = int(len(x_cal))

                # Skip degenerate splits: imbalanced datasets at large n_cal_target
                # can round the minority class to 0 samples in x_fit or x_cal,
                # causing predict_proba to return a single-column array.
                if len(np.unique(y_fit)) < 2 or len(np.unique(y_cal)) < 2:
                    continue

                difficulty_estimator = _DeterministicDifficultyEstimator.fit(x_fit)

                model = RandomForestClassifier(
                    n_estimators=60 if config.quick else 120,
                    random_state=seed,
                    max_depth=8 if config.quick else None,
                    n_jobs=1,
                )

                wrapper_a = WrapCalibratedExplainer(model)
                wrapper_a.fit(x_fit, y_fit)
                wrapper_a.calibrate(x_cal, y_cal, feature_names=list(feature_names))
                wrapper_a.set_difficulty_estimator(None, initialize=False)

                wrapper_c = WrapCalibratedExplainer(model)
                wrapper_c.fit(x_fit, y_fit)
                wrapper_c.calibrate(x_cal, y_cal, feature_names=list(feature_names))
                wrapper_c.set_difficulty_estimator(difficulty_estimator, initialize=False)

                for epsilon in _EPSILONS:
                    confidence = 1.0 - float(epsilon)
                    policy = RejectPolicySpec.flag(ncf="default", w=_W)

                    for arm_code, wrapper, strategy in [
                        ("A", wrapper_a, "builtin.default"),
                        ("C", wrapper_c, "experimental.difficulty_normalized"),
                    ]:
                        result = wrapper.predict(
                            x_test,
                            reject_policy=policy,
                            confidence=confidence,
                            strategy=strategy,
                        )
                        meta = getattr(result, "metadata", {}) or {}
                        prediction_set_raw = meta.get("prediction_set")
                        if prediction_set_raw is None:
                            continue
                        prediction_set = np.asarray(prediction_set_raw, dtype=bool)
                        if prediction_set.ndim != 2:
                            continue

                        cov = empirical_coverage(prediction_set, y_test)
                        singleton_metrics = singleton_precision_recall(prediction_set, y_test)
                        successes = int(
                            np.sum(prediction_set[np.arange(len(y_test)), y_test])
                        )
                        lower_ci, upper_ci = clopper_pearson_interval(successes, len(y_test))
                        structural_violation = bool(upper_ci < 1.0 - epsilon)
                        violation = bool(cov < 1.0 - epsilon)

                        rows.append(
                            {
                                "arm": arm_code,
                                "strategy": strategy,
                                "dataset": spec.name,
                                "seed": seed,
                                "n_cal_target": int(n_cal_target),
                                "n_cal": actual_n_cal,
                                "n_test": int(len(x_test)),
                                "epsilon": float(epsilon),
                                "confidence": confidence,
                                "coverage": cov,
                                "lower_ci": lower_ci,
                                "upper_ci": upper_ci,
                                "violation": violation,
                                "structural_violation": structural_violation,
                                "reject_rate": float(meta.get("reject_rate", float("nan"))),
                                **singleton_metrics,
                            }
                        )

    df = pd.DataFrame(rows)
    if df.empty:
        meta_out: dict[str, Any] = {
            "scenario": "scenario_13_ncal_coverage_sweep",
            "display_name": "Scenario 13 — n_cal sweep: arm A vs arm C structural violations",
            "guarantee_status": "empirical",
            "quick": config.quick,
            "highlights": ["No data generated."],
            "outcome": {},
        }
        write_csv_json_md("scenario_13_ncal_coverage_sweep", df, meta_out)
        return

    # Per arm × n_cal_target summary: structural violation rate
    sv_by_arm_ncal = (
        df.groupby(["arm", "n_cal_target"])
        .agg(
            structural_violation_rate=("structural_violation", "mean"),
            structural_violations=("structural_violation", "sum"),
            total_rows=("structural_violation", "count"),
            mean_coverage=("coverage", "mean"),
        )
        .reset_index()
    )

    # Check hypothesis: does arm C structural violation rate decrease with n_cal?
    def _is_decreasing(series: pd.Series) -> bool:
        vals = series.dropna().tolist()
        if len(vals) < 2:
            return False
        return all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))

    sv_c = sv_by_arm_ncal[sv_by_arm_ncal["arm"] == "C"].sort_values("n_cal_target")
    hypothesis_supported = bool(_is_decreasing(sv_c["structural_violation_rate"]))

    sv_a_total = int(df[df["arm"] == "A"]["structural_violation"].sum())
    sv_c_total = int(df[df["arm"] == "C"]["structural_violation"].sum())
    total_rows_a = int((df["arm"] == "A").shape[0])
    total_rows_c = int((df["arm"] == "C").shape[0])

    meta_out = {
        "scenario": "scenario_13_ncal_coverage_sweep",
        "display_name": "Scenario 13 — n_cal sweep: arm A vs arm C structural violations",
        "paper_contribution": "supplementary",
        "guarantee_status": "empirical",
        "quick": config.quick,
        "highlights": [
            "Sweeps n_cal in {50, 100, 200, 400} to test variance-inflation hypothesis for arm C.",
            "Hypothesis: arm C structural violation rate decreases as n_cal grows (finite-sample effect).",
            "Counter-evidence: if rate does not decrease, a genuine exchangeability violation is indicated.",
            f"Arm A structural violations: {sv_a_total}/{total_rows_a}.",
            f"Arm C structural violations: {sv_c_total}/{total_rows_c}.",
            f"Hypothesis (arm C violations decrease with n_cal): {'SUPPORTED' if hypothesis_supported else 'NOT SUPPORTED'}.",
        ],
        "outcome": {
            "rows": int(len(df)),
            "datasets": int(df["dataset"].nunique()),
            "seeds": int(df["seed"].nunique()),
            "n_cal_targets": list(_N_CAL_TARGETS),
            "arm_A_structural_violations": sv_a_total,
            "arm_A_total_rows": total_rows_a,
            "arm_C_structural_violations": sv_c_total,
            "arm_C_total_rows": total_rows_c,
            "hypothesis_supported": hypothesis_supported,
            "sv_by_arm_ncal": sv_by_arm_ncal.to_dict(orient="records"),
        },
    }

    # --- Extra sections ---
    extra_sections: list[str] = []

    # Section: Structural violation rate by arm and n_cal
    extra_sections += [
        "## Structural violation rate by arm and n_cal",
        "",
        _markdown_table_from_df(sv_by_arm_ncal),
        "",
    ]

    # Section: Hypothesis verdict
    verdict = pd.DataFrame([{
        "arm_C_hypothesis_supported": hypothesis_supported,
        "arm_A_total_sv": sv_a_total,
        "arm_C_total_sv": sv_c_total,
    }])
    extra_sections += [
        "## Hypothesis verdict",
        "",
        _markdown_table_from_df(verdict),
        "",
    ]

    write_csv_json_md("scenario_13_ncal_coverage_sweep", df, meta_out, extra_sections=extra_sections)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    arguments = parser.parse_args()
    run(RunConfig(seed=42, quick=arguments.quick))
