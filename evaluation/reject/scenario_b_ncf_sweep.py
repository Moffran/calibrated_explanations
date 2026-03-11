"""Scenario B: sweep reject NCF variants through `RejectPolicySpec`."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from calibrated_explanations import RejectPolicySpec

from .common_reject import (
    RunConfig,
    build_multiclass_bundle,
    explanation_count,
    save_plot,
    seed_grid,
    select_best_row,
    write_csv_json_md,
)

NCF_TYPES = ["default", "entropy", "ensured"]


def run(config: RunConfig) -> None:
    """Evaluate real reject behavior across the supported NCF variants."""
    rows: list[dict[str, float | int | str]] = []

    for repeat in seed_grid(config):
        bundle = build_multiclass_bundle(config, seed_offset=repeat * 11)
        baseline_accuracy = float((bundle.baseline_pred == bundle.y_test).mean())
        for ncf in NCF_TYPES:
            policy = RejectPolicySpec.flag(ncf=ncf, w=0.5)
            result = bundle.wrapper.predict(
                bundle.x_test,
                reject_policy=policy,
                confidence=0.95,
            )
            explanation = bundle.wrapper.explain_factual(
                bundle.x_test,
                reject_policy=policy,
                confidence=0.95,
            )
            metadata = result.metadata or {}
            rejected = np.asarray(result.rejected, dtype=bool)
            accepted = ~rejected
            accepted_accuracy = (
                float((bundle.baseline_pred[accepted] == bundle.y_test[accepted]).mean())
                if np.any(accepted)
                else float("nan")
            )
            prediction_set_size = np.asarray(metadata.get("prediction_set_size", []), dtype=float)
            rows.append(
                {
                    "ncf": ncf,
                    "repeat": int(repeat),
                    "coverage": float(np.mean(accepted)),
                    "reject_rate": float(metadata.get("reject_rate", np.nan)),
                    "error_rate": float(metadata.get("error_rate", np.nan)),
                    "ambiguity_rate": float(metadata.get("ambiguity_rate", np.nan)),
                    "novelty_rate": float(metadata.get("novelty_rate", np.nan)),
                    "accepted_accuracy": accepted_accuracy,
                    "accepted_accuracy_delta": accepted_accuracy - baseline_accuracy,
                    "mean_prediction_set_size": float(np.mean(prediction_set_size))
                    if prediction_set_size.size
                    else float("nan"),
                }
            )

    df = pd.DataFrame(rows)
    grouped = (
        df.groupby("ncf", as_index=False)[
            [
                "coverage",
                "reject_rate",
                "error_rate",
                "accepted_accuracy_delta",
                "mean_prediction_set_size",
            ]
        ]
        .mean()
        .sort_values("accepted_accuracy_delta", ascending=False)
    )
    best_row = select_best_row(
        grouped,
        sort_by=("accepted_accuracy_delta", "coverage"),
        ascending=(False, False),
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].bar(grouped["ncf"], grouped["accepted_accuracy_delta"], color="#4477aa")
    axes[0].axhline(0.0, color="black", linewidth=1.0, alpha=0.4)
    axes[0].set_title("Mean accepted-accuracy uplift")
    axes[0].set_ylabel("accuracy delta")
    axes[0].grid(axis="y", alpha=0.3)
    axes[1].bar(grouped["ncf"], grouped["coverage"], color="#66c2a5")
    axes[1].set_title("Mean coverage")
    axes[1].set_ylabel("coverage")
    axes[1].grid(axis="y", alpha=0.3)
    plot_name = save_plot("ncf_sweep", fig, "comparison")

    highlights = [
        f"Best mean accepted-accuracy uplift came from `{best_row['ncf']}`, with delta {best_row['accepted_accuracy_delta']:.4f} at mean coverage {best_row['coverage']:.4f}.",
        "This scenario uses `RejectPolicySpec.flag(...)` so the sweep exercises the public per-call policy+NCF contract rather than direct synthetic scoring.",
        "Prediction-set size highlights how aggressively each NCF concentrates accepted multiclass decisions.",
    ]
    meta = {
        "scenario": "ncf_sweep",
        "display_name": "Scenario B — NCF sweep",
        "dataset": "iris",
        "quick": config.quick,
        "highlights": highlights,
        "outcome": {
            "best_ncf": best_row["ncf"],
            "best_mean_accuracy_delta": float(best_row["accepted_accuracy_delta"]),
            "best_mean_coverage": float(best_row["coverage"]),
            "lowest_mean_error_rate": float(grouped["error_rate"].min()),
        },
        "plots": [plot_name],
    }
    write_csv_json_md("ncf_sweep", df, meta)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    arguments = parser.parse_args()
    run(RunConfig(seed=42, quick=arguments.quick))
