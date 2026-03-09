"""Scenario C: verify confidence/coverage behavior from the real reject learner."""

from __future__ import annotations

import argparse
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common_reject import (
    RunConfig,
    build_binary_bundle,
    confidence_from_matrix,
    confidence_grid,
    reject_breakdown,
    save_plot,
    write_csv_json_md,
)


def run(config: RunConfig) -> None:
    """Measure monotonicity under increasing reject confidence."""
    bundle = build_binary_bundle(config)
    accepted_conf = confidence_from_matrix(bundle.baseline_proba)
    rows: list[dict[str, float]] = []

    for confidence in confidence_grid(config.quick, start=0.80, stop=0.99):
        breakdown = reject_breakdown(bundle.wrapper, bundle.x_test, confidence=float(confidence))
        rejected = np.asarray(breakdown["rejected"], dtype=bool)
        accepted = ~rejected
        avg_confidence = float(np.mean(accepted_conf[accepted])) if np.any(accepted) else float("nan")
        rows.append(
            {
                "confidence": float(confidence),
                "coverage": float(np.mean(accepted)),
                "avg_confidence": avg_confidence,
                "accepted_accuracy": float((bundle.baseline_pred[accepted] == bundle.y_test[accepted]).mean())
                if np.any(accepted)
                else float("nan"),
                "ambiguity_rate": float(breakdown.get("ambiguity_rate", np.nan)),
                "novelty_rate": float(breakdown.get("novelty_rate", np.nan)),
            }
        )

    df = pd.DataFrame(rows).sort_values("confidence").reset_index(drop=True)
    coverage_diff = np.diff(df["coverage"].to_numpy())
    confidence_diff = np.diff(df["avg_confidence"].ffill().to_numpy())
    coverage_violations = int(np.sum(coverage_diff > 1e-9))
    confidence_violations = int(np.sum(confidence_diff < -1e-9))

    # One tie-induced violation per confidence transition is acceptable for
    # discrete conformal prediction on small test sets (< ~100 instances).
    # More than one consecutive violation indicates a genuine non-monotonicity.
    violation_tolerance = 1
    if coverage_violations > violation_tolerance:
        warnings.warn(
            f"Scenario C: coverage monotonicity has {coverage_violations} violations "
            f"(tolerance={violation_tolerance}). This exceeds the expected tie-artefact "
            "level — check for non-determinism or a conformal guarantee violation.",
            UserWarning,
            stacklevel=2,
        )

    fig, ax1 = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    ax2 = ax1.twinx()
    ax1.plot(df["confidence"], df["coverage"], marker="o", color="#4477aa", label="coverage")
    ax2.plot(
        df["confidence"],
        df["avg_confidence"],
        marker="s",
        color="#dd8452",
        label="avg accepted confidence",
    )
    ax1.set_xlabel("reject confidence")
    ax1.set_ylabel("coverage", color="#4477aa")
    ax2.set_ylabel("avg accepted confidence", color="#dd8452")
    ax1.set_title("Confidence monotonicity under rejection")
    ax1.grid(alpha=0.3)
    handles = ax1.get_lines() + ax2.get_lines()
    ax1.legend(handles, [line.get_label() for line in handles], loc="best")
    plot_name = save_plot("confidence_monotonicity", fig, "curve")

    highlights = [
        f"Coverage monotonicity violations: {coverage_violations}; accepted-confidence monotonicity violations: {confidence_violations}.",
        f"Coverage ranged from {df['coverage'].max():.4f} down to {df['coverage'].min():.4f} as confidence increased.",
        f"Average accepted confidence peaked at {df['avg_confidence'].max():.4f}.",
    ]
    meta = {
        "scenario": "confidence_monotonicity",
        "display_name": "Scenario C — Confidence monotonicity",
        "dataset": bundle.dataset_name,
        "quick": config.quick,
        "highlights": highlights,
        "outcome": {
            "coverage_monotonicity_violations": coverage_violations,
            "accepted_confidence_monotonicity_violations": confidence_violations,
            "violation_tolerance": violation_tolerance,
            "violations_within_tolerance": coverage_violations <= violation_tolerance,
            "max_avg_confidence": float(df["avg_confidence"].max()),
            "min_coverage": float(df["coverage"].min()),
        },
        "plots": [plot_name],
    }
    write_csv_json_md("confidence_monotonicity", df, meta)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    arguments = parser.parse_args()
    run(RunConfig(seed=42, quick=arguments.quick))
