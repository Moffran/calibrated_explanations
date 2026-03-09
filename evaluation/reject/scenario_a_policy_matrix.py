"""Scenario A: compare active reject policies across confidence levels."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from calibrated_explanations.core.reject.policy import RejectPolicy

from .common_reject import (
    RunConfig,
    build_binary_bundle,
    confidence_grid,
    explanation_count,
    expected_calibration_error,
    reject_breakdown,
    save_plot,
    select_best_row,
    write_csv_json_md,
)


def run(config: RunConfig) -> None:
    """Run the policy-matrix sweep using the real reject integration path."""
    bundle = build_binary_bundle(config)
    confidences = confidence_grid(config.quick)
    policies = [RejectPolicy.FLAG, RejectPolicy.ONLY_REJECTED, RejectPolicy.ONLY_ACCEPTED]
    baseline_accuracy = float((bundle.baseline_pred == bundle.y_test).mean())
    baseline_ece = expected_calibration_error(bundle.y_test, bundle.baseline_proba[:, 1])
    rows: list[dict[str, float | str]] = []

    for policy in policies:
        for confidence in confidences:
            explanation = bundle.wrapper.explain_factual(
                bundle.x_test,
                reject_policy=policy,
                confidence=float(confidence),
            )
            breakdown = reject_breakdown(bundle.wrapper, bundle.x_test, confidence=float(confidence))
            rejected = np.asarray(breakdown["rejected"], dtype=bool)
            accepted = ~rejected
            explained = explanation_count(explanation)
            accepted_accuracy = (
                float((bundle.baseline_pred[accepted] == bundle.y_test[accepted]).mean())
                if np.any(accepted)
                else float("nan")
            )
            accepted_ece = (
                expected_calibration_error(
                    bundle.y_test[accepted],
                    bundle.baseline_proba[accepted, 1],
                )
                if np.any(accepted)
                else float("nan")
            )
            rows.append(
                {
                    "policy": policy.value,
                    "confidence": float(confidence),
                    "coverage": float(np.mean(accepted)),
                    "reject_rate": float(breakdown["reject_rate"]),
                    "error_rate": float(breakdown["error_rate"]),
                    "ambiguity_rate": float(breakdown.get("ambiguity_rate", np.nan)),
                    "novelty_rate": float(breakdown.get("novelty_rate", np.nan)),
                    "accepted_accuracy": accepted_accuracy,
                    "accepted_accuracy_delta": accepted_accuracy - baseline_accuracy,
                    "accepted_ece": accepted_ece,
                    "accepted_ece_delta": baseline_ece - accepted_ece,
                }
            )

    df = pd.DataFrame(rows)
    best_row = select_best_row(
        df[df["coverage"] >= 0.50] if (df["coverage"] >= 0.50).any() else df,
        sort_by=("accepted_accuracy_delta", "coverage"),
        ascending=(False, False),
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for policy_name, policy_df in df.groupby("policy"):
        axes[0].plot(policy_df["confidence"], policy_df["coverage"], marker="o", label=policy_name)
        axes[1].plot(
            policy_df["confidence"],
            policy_df["accepted_accuracy_delta"],
            marker="o",
            label=policy_name,
        )
    axes[0].set_title("Coverage vs confidence")
    axes[0].set_xlabel("confidence")
    axes[0].set_ylabel("coverage")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[1].set_title("Accepted accuracy uplift vs baseline")
    axes[1].set_xlabel("confidence")
    axes[1].set_ylabel("accuracy delta")
    axes[1].axhline(0.0, color="black", linewidth=1.0, alpha=0.4)
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    plot_name = save_plot("policy_matrix", fig, "tradeoffs")

    highlights = [
        f"Best accuracy uplift came from `{best_row['policy']}` at confidence {best_row['confidence']:.2f}, improving accepted accuracy by {best_row['accepted_accuracy_delta']:.4f} at coverage {best_row['coverage']:.4f}.",
        f"Baseline full-sample accuracy was {baseline_accuracy:.4f} and baseline ECE was {baseline_ece:.4f}.",
        "`flag` explains every instance, while `only_rejected` and `only_accepted` shift explanation volume to the rejected or accepted subset respectively.",
    ]
    meta = {
        "scenario": "policy_matrix",
        "display_name": "Scenario A — Policy matrix",
        "dataset": bundle.dataset_name,
        "quick": config.quick,
        "highlights": highlights,
        "outcome": {
            "best_policy": best_row["policy"],
            "best_confidence": float(best_row["confidence"]),
            "best_accuracy_delta": float(best_row["accepted_accuracy_delta"]),
            "best_coverage": float(best_row["coverage"]),
            "baseline_accuracy": baseline_accuracy,
            "baseline_ece": baseline_ece,
        },
        "plots": [plot_name],
    }
    write_csv_json_md("policy_matrix", df, meta)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    arguments = parser.parse_args()
    run(RunConfig(seed=42, quick=arguments.quick))
