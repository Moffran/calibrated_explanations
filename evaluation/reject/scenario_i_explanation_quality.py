"""Scenario I: explanation quality on accepted instances."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from calibrated_explanations import RejectPolicySpec

from .common_reject import (
    RunConfig,
    build_classification_bundle,
    expected_calibration_error,
    mean_feature_weight_variance,
    task_specs,
    write_csv_json_md,
)


def run(config: RunConfig) -> None:
    """Compare explanation quality on all vs accepted subsets."""
    rows: list[dict[str, float | str | int]] = []
    datasets = task_specs("binary", quick=config.quick) + task_specs("multiclass", quick=config.quick)
    for spec in datasets:
        bundle = build_classification_bundle(spec, config)
        confidence = 0.95
        result = bundle.wrapper.predict(bundle.x_test, reject_policy=RejectPolicySpec.flag(), confidence=confidence)
        rejected = np.asarray(result.rejected, dtype=bool)
        accepted = ~rejected
        all_explanations = bundle.wrapper.explain_factual(bundle.x_test)
        accepted_explanations = (
            bundle.wrapper.explain_factual(bundle.x_test[accepted]) if np.any(accepted) else []
        )
        if bundle.baseline_proba.shape[1] == 2:
            proba_all = bundle.baseline_proba[:, 1]
            proba_accepted = bundle.baseline_proba[accepted, 1] if np.any(accepted) else np.array([])
        else:
            proba_all = np.max(bundle.baseline_proba, axis=1)
            proba_accepted = np.max(bundle.baseline_proba[accepted], axis=1) if np.any(accepted) else np.array([])
        baseline_accuracy = float(np.mean(bundle.baseline_pred == bundle.y_test))
        accepted_accuracy = (
            float(np.mean(bundle.baseline_pred[accepted] == bundle.y_test[accepted]))
            if np.any(accepted)
            else float("nan")
        )
        baseline_ece = expected_calibration_error(
            (bundle.baseline_pred == bundle.y_test).astype(int),
            proba_all,
        )
        accepted_ece = (
            expected_calibration_error(
                (bundle.baseline_pred[accepted] == bundle.y_test[accepted]).astype(int),
                proba_accepted,
            )
            if np.any(accepted)
            else float("nan")
        )
        rows.append(
            {
                "dataset": spec.name,
                "n_test": int(len(bundle.x_test)),
                "confidence": confidence,
                "baseline_accuracy": baseline_accuracy,
                "accepted_accuracy": accepted_accuracy,
                "accuracy_delta": accepted_accuracy - baseline_accuracy if np.isfinite(accepted_accuracy) else float("nan"),
                "baseline_ece": baseline_ece,
                "accepted_ece": accepted_ece,
                "ece_delta": baseline_ece - accepted_ece if np.isfinite(accepted_ece) else float("nan"),
                "reject_rate": float(np.mean(rejected)),
                "weight_variance_all": mean_feature_weight_variance(all_explanations),
                "weight_variance_accepted": mean_feature_weight_variance(accepted_explanations),
            }
        )

    df = pd.DataFrame(rows)
    meta = {
        "scenario": "scenario_i_explanation_quality",
        "display_name": "Scenario I — Explanation quality",
        "quick": config.quick,
        "highlights": [
            "Explanation quality is evaluated only empirically; no conformal claim is attached to these metrics.",
            "Feature-weight stability is computed from per-instance explanation weight vectors.",
        ],
        "outcome": {
            "datasets": int(df["dataset"].nunique()) if not df.empty else 0,
            "mean_accuracy_delta": float(df["accuracy_delta"].mean()) if not df.empty else float("nan"),
            "mean_ece_delta": float(df["ece_delta"].mean()) if not df.empty else float("nan"),
        },
    }
    write_csv_json_md("scenario_i_explanation_quality", df, meta)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    arguments = parser.parse_args()
    run(RunConfig(seed=42, quick=arguments.quick))
