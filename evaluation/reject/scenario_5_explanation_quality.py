"""Scenario 5: explanation quality on accepted instances.

Paper mapping: C4 / RQ5 (empirical).

Paper finding (Section 6 RQ5): explanation-quality improvements (accuracy delta, ECE delta)
are reliable at low reject rates (<=15%) but degrade at high reject rates (>40%).  This scenario
produces regime-segmented summaries to reproduce that finding.

Reject-rate regimes:
  low      : reject_rate <= 0.15
  moderate : 0.15 < reject_rate <= 0.40
  high     : reject_rate > 0.40

mean_feature_weight_variance is NOT included in the primary output table — it is not a paper
metric.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from calibrated_explanations import RejectPolicySpec

from .common_reject import (
    RunConfig,
    build_classification_bundle,
    expected_calibration_error,
    task_specs,
    write_csv_json_md,
)

_LOW_REGIME = 0.15
_HIGH_REGIME = 0.40


def _regime_label(reject_rate: float) -> str:
    if reject_rate <= _LOW_REGIME:
        return "low"
    if reject_rate <= _HIGH_REGIME:
        return "moderate"
    return "high"


def run(config: RunConfig) -> None:
    """Compare explanation quality on all vs accepted subsets with regime segmentation."""
    rows: list[dict[str, float | str | int]] = []
    datasets = task_specs("binary", quick=config.quick) + task_specs("multiclass", quick=config.quick)
    for spec in datasets:
        bundle = build_classification_bundle(spec, config)
        confidence = 0.95
        result = bundle.wrapper.predict(bundle.x_test, reject_policy=RejectPolicySpec.flag(), confidence=confidence)
        rejected = np.asarray(result.rejected, dtype=bool)
        accepted = ~rejected
        reject_rate = float(np.mean(rejected))

        if bundle.baseline_proba.shape[1] == 2:
            proba_all = bundle.baseline_proba[:, 1]
            proba_accepted = bundle.baseline_proba[accepted, 1] if np.any(accepted) else np.array([])
        else:
            proba_all = np.max(bundle.baseline_proba, axis=1)
            proba_accepted = (
                np.max(bundle.baseline_proba[accepted], axis=1) if np.any(accepted) else np.array([])
            )

        baseline_accuracy = float(np.mean(bundle.baseline_pred == bundle.y_test))
        accepted_accuracy_val = (
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
                "task_type": spec.task_type,
                "n_test": int(len(bundle.x_test)),
                "confidence": confidence,
                "reject_rate": reject_rate,
                "regime": _regime_label(reject_rate),
                "baseline_accuracy": baseline_accuracy,
                "accepted_accuracy": accepted_accuracy_val,
                "accuracy_delta": (
                    accepted_accuracy_val - baseline_accuracy
                    if np.isfinite(accepted_accuracy_val)
                    else float("nan")
                ),
                "baseline_ece": baseline_ece,
                "accepted_ece": accepted_ece,
                "ece_delta": (
                    baseline_ece - accepted_ece if np.isfinite(accepted_ece) else float("nan")
                ),
            }
        )

    df = pd.DataFrame(rows)

    # Regime-segmented summary (the paper's key RQ5 finding)
    regime_summary: dict[str, dict[str, float]] = {}
    for regime in ("low", "moderate", "high"):
        sub = df[df["regime"] == regime]
        regime_summary[regime] = {
            "n": int(len(sub)),
            "mean_accuracy_delta": float(sub["accuracy_delta"].mean()) if not sub.empty else float("nan"),
            "mean_ece_delta": float(sub["ece_delta"].mean()) if not sub.empty else float("nan"),
            "mean_reject_rate": float(sub["reject_rate"].mean()) if not sub.empty else float("nan"),
        }

    meta = {
        "scenario": "scenario_5_explanation_quality",
        "display_name": "Scenario 5 — Explanation quality on accepted instances",
        "paper_contribution": "C4",
        "paper_rq": "RQ5",
        "guarantee_status": "empirical",
        "quick": config.quick,
        "regime_thresholds": {"low_max": _LOW_REGIME, "high_min": _HIGH_REGIME},
        "regime_summary": regime_summary,
        "highlights": [
            "Explanation quality is evaluated only empirically; no conformal claim is attached.",
            f"Regime boundaries: low (<={_LOW_REGIME:.0%}), "
            f"moderate ({_LOW_REGIME:.0%}–{_HIGH_REGIME:.0%}), "
            f"high (>{_HIGH_REGIME:.0%}) reject rate.",
            "Paper finding: accuracy_delta is most reliable in the low regime.",
            "mean_feature_weight_variance is not included — it is not a paper metric.",
        ],
        "outcome": {
            "datasets": int(df["dataset"].nunique()) if not df.empty else 0,
            "mean_accuracy_delta": float(df["accuracy_delta"].mean()) if not df.empty else float("nan"),
            "mean_ece_delta": float(df["ece_delta"].mean()) if not df.empty else float("nan"),
            "regime_summary": regime_summary,
        },
    }
    write_csv_json_md("scenario_5_explanation_quality", df, meta)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    arguments = parser.parse_args()
    run(RunConfig(seed=42, quick=arguments.quick))
