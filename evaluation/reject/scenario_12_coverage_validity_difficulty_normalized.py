"""Scenario 12: marginal coverage validity for arm C (difficulty-normalized).

Paper mapping: supplementary (RT-3 red-team obligation).

This scenario mirrors Scenario 1 (binary marginal coverage sweep) but runs both
arm A (`builtin.default`) and arm C (`experimental.difficulty_normalized`) side
by side. The conformal coverage guarantee is tied to exchangeability of the
nonconformity scores; difficulty normalization changes the score definition, so
coverage validity must be verified empirically for arm C.

A structural violation is reported when the Clopper-Pearson CI upper bound lies
below the nominal coverage level 1-epsilon, meaning finite-sample noise cannot
explain the shortfall.
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
    clopper_pearson_interval,
    empirical_coverage,
    load_dataset,
    seed_grid,
    split_dataset,
    task_specs,
    write_csv_json_md,
)

_W = 0.5


@dataclass
class _DeterministicDifficultyEstimator:
    """Minimal deterministic estimator for reproducible difficulty ablations."""

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
    """Measure binary marginal coverage for arm A and arm C side by side."""
    epsilons = (0.05, 0.10)
    rows: list[dict[str, Any]] = []

    for spec in task_specs("binary", quick=config.quick):
        _, x_all, y_all, feature_names = load_dataset(spec)

        for seed_offset in seed_grid(config):
            seed = int(config.seed + seed_offset)
            x_fit, x_cal, x_test, y_fit, y_cal, y_test = split_dataset(
                x_all, y_all, seed=seed, stratify=True
            )
            difficulty_estimator = _DeterministicDifficultyEstimator.fit(x_fit)

            model = RandomForestClassifier(
                n_estimators=60 if config.quick else 120,
                random_state=seed,
                max_depth=8 if config.quick else None,
                n_jobs=1,
            )
            # Arm A: builtin.default — no difficulty estimator
            wrapper_a = WrapCalibratedExplainer(model)
            wrapper_a.fit(x_fit, y_fit)
            wrapper_a.calibrate(x_cal, y_cal, feature_names=list(feature_names))
            wrapper_a.set_difficulty_estimator(None, initialize=False)

            # Arm C: experimental.difficulty_normalized — difficulty estimator required
            wrapper_c = WrapCalibratedExplainer(model)
            wrapper_c.fit(x_fit, y_fit)
            wrapper_c.calibrate(x_cal, y_cal, feature_names=list(feature_names))
            wrapper_c.set_difficulty_estimator(difficulty_estimator, initialize=False)

            for epsilon in epsilons:
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
                            "epsilon": float(epsilon),
                            "confidence": confidence,
                            "n_cal": int(len(x_cal)),
                            "n_test": int(len(x_test)),
                            "coverage": cov,
                            "lower_ci": lower_ci,
                            "upper_ci": upper_ci,
                            "violation": violation,
                            "structural_violation": structural_violation,
                            "reject_rate": float(meta.get("reject_rate", float("nan"))),
                        }
                    )

    df = pd.DataFrame(rows)
    if df.empty:
        meta_out: dict[str, Any] = {
            "scenario": "scenario_12_coverage_validity_difficulty_normalized",
            "display_name": "Scenario 12 — Coverage validity: arm A vs arm C",
            "guarantee_status": "empirical",
            "quick": config.quick,
            "highlights": ["No data generated."],
            "outcome": {},
        }
        write_csv_json_md("scenario_12_coverage_validity_difficulty_normalized", df, meta_out)
        return

    violation_count_a = int(df[df["arm"] == "A"]["violation"].sum())
    violation_count_c = int(df[df["arm"] == "C"]["violation"].sum())
    structural_count_a = int(df[df["arm"] == "A"]["structural_violation"].sum())
    structural_count_c = int(df[df["arm"] == "C"]["structural_violation"].sum())
    total_rows_a = int((df["arm"] == "A").sum())
    total_rows_c = int((df["arm"] == "C").sum())
    mean_cov_a = float(df[df["arm"] == "A"]["coverage"].mean())
    mean_cov_c = float(df[df["arm"] == "C"]["coverage"].mean())

    meta_out = {
        "scenario": "scenario_12_coverage_validity_difficulty_normalized",
        "display_name": "Scenario 12 — Coverage validity: arm A vs arm C",
        "paper_contribution": "supplementary",
        "guarantee_status": "empirical",
        "quick": config.quick,
        "highlights": [
            "Mirrors Scenario 1 but runs arm A (builtin.default) and arm C (experimental.difficulty_normalized) side by side.",
            "Coverage is standard label-set coverage from conformal prediction sets.",
            "A structural violation means CI upper bound < 1-epsilon; finite-sample noise cannot explain the shortfall.",
            f"Arm A violations: {violation_count_a}/{total_rows_a}; structural: {structural_count_a}/{total_rows_a}.",
            f"Arm C violations: {violation_count_c}/{total_rows_c}; structural: {structural_count_c}/{total_rows_c}.",
            f"Arm A mean coverage: {mean_cov_a:.4f}; Arm C mean coverage: {mean_cov_c:.4f}.",
        ],
        "outcome": {
            "rows": int(len(df)),
            "datasets": int(df["dataset"].nunique()),
            "seeds": int(df["seed"].nunique()),
            "arm_A_violations": violation_count_a,
            "arm_A_structural_violations": structural_count_a,
            "arm_A_total_rows": total_rows_a,
            "arm_A_mean_coverage": mean_cov_a,
            "arm_C_violations": violation_count_c,
            "arm_C_structural_violations": structural_count_c,
            "arm_C_total_rows": total_rows_c,
            "arm_C_mean_coverage": mean_cov_c,
        },
    }
    write_csv_json_md("scenario_12_coverage_validity_difficulty_normalized", df, meta_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    arguments = parser.parse_args()
    run(RunConfig(seed=42, quick=arguments.quick))
