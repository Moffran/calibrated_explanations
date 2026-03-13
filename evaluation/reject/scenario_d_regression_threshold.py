"""Scenario D: evaluate real thresholded regression rejection behavior."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common_reject import (
    RunConfig,
    binary_accuracy_from_threshold,
    build_regression_bundle,
    interval_outside_fraction,
    quantile_grid,
    regression_mse,
    save_plot,
    select_best_row,
    write_csv_json_md,
)


def run(config: RunConfig) -> None:
    """Sweep regression reject thresholds on the diabetes benchmark."""
    base_bundle = build_regression_bundle(config)
    rows: list[dict[str, float]] = []

    for quantile in quantile_grid(config.quick):
        threshold = float(np.quantile(base_bundle.y_cal, quantile))
        bundle = build_regression_bundle(config, seed_offset=int(quantile * 1000))
        bundle.wrapper.explainer.reject_orchestrator.initialize_reject_learner(
            threshold=threshold
        )
        breakdown = bundle.wrapper.explainer.reject_orchestrator.predict_reject_breakdown(
            bundle.x_test,
            confidence=0.95,
            threshold=threshold,
        )
        rejected = np.asarray(breakdown["rejected"], dtype=bool)
        accepted = ~rejected
        outside_all = interval_outside_fraction(bundle.y_test, bundle.baseline_low, bundle.baseline_high)
        outside_accepted = (
            interval_outside_fraction(
                bundle.y_test[accepted],
                bundle.baseline_low[accepted],
                bundle.baseline_high[accepted],
            )
            if np.any(accepted)
            else float("nan")
        )
        rows.append(
            {
                "threshold_quantile": float(quantile),
                "threshold_value": threshold,
                "coverage": float(np.mean(accepted)),
                "reject_rate": float(breakdown["reject_rate"]),
                "error_rate": float(breakdown["error_rate"]),
                "outside_fraction_all": outside_all,
                "outside_fraction_accepted": outside_accepted,
                "binary_accuracy_all": binary_accuracy_from_threshold(
                    bundle.y_test,
                    bundle.baseline_pred,
                    threshold=threshold,
                ),
                "binary_accuracy_accepted": binary_accuracy_from_threshold(
                    bundle.y_test[accepted],
                    bundle.baseline_pred[accepted],
                    threshold=threshold,
                )
                if np.any(accepted)
                else float("nan"),
                "mse_all": regression_mse(bundle.y_test, bundle.baseline_pred),
                "mse_accepted": regression_mse(bundle.y_test[accepted], bundle.baseline_pred[accepted])
                if np.any(accepted)
                else float("nan"),
            }
        )

    df = pd.DataFrame(rows)
    df["binary_accuracy_delta"] = df["binary_accuracy_accepted"] - df["binary_accuracy_all"]
    df["outside_fraction_delta"] = df["outside_fraction_all"] - df["outside_fraction_accepted"]
    best_row = select_best_row(
        df,
        sort_by=("binary_accuracy_delta", "outside_fraction_delta"),
        ascending=(False, False),
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].plot(df["threshold_quantile"], df["coverage"], marker="o", label="coverage")
    axes[0].plot(df["threshold_quantile"], df["reject_rate"], marker="s", label="reject rate")
    axes[0].set_title("Coverage and reject rate")
    axes[0].set_xlabel("threshold quantile")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[1].plot(
        df["threshold_quantile"],
        df["outside_fraction_accepted"],
        marker="o",
        label="accepted outside fraction",
    )
    axes[1].plot(
        df["threshold_quantile"],
        df["binary_accuracy_accepted"],
        marker="s",
        label="accepted binary accuracy",
    )
    axes[1].set_title("Accepted-set quality")
    axes[1].set_xlabel("threshold quantile")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    plot_name = save_plot("regression_threshold", fig, "tradeoffs")

    interval_delta = float(best_row["outside_fraction_delta"])
    interval_phrase = (
        f"Accepted-set interval miss fraction improved by {interval_delta:.4f}"
        if interval_delta >= 0.0
        else f"Accepted-set interval miss fraction changed by {interval_delta:.4f}"
    )
    highlights = [
        f"Best threshold quantile was {best_row['threshold_quantile']:.2f}, improving thresholded binary accuracy by {best_row['binary_accuracy_delta']:.4f}.",
        f"{interval_phrase} at that setting.",
        "This scenario evaluates real reject learner thresholds for probabilistic regression rather than synthetic interval scaling.",
    ]
    meta = {
        "scenario": "regression_threshold",
        "display_name": "Scenario D — Regression threshold sweep",
        "dataset": base_bundle.dataset_name,
        "quick": config.quick,
        "highlights": highlights,
        "outcome": {
            "best_threshold_quantile": float(best_row["threshold_quantile"]),
            "best_threshold_value": float(best_row["threshold_value"]),
            "best_binary_accuracy_delta": float(best_row["binary_accuracy_delta"]),
            "best_outside_fraction_delta": float(best_row["outside_fraction_delta"]),
        },
        "plots": [plot_name],
    }
    write_csv_json_md("regression_threshold", df, meta)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    arguments = parser.parse_args()
    run(RunConfig(seed=42, quick=arguments.quick))
