"""Scenario B — OOD Detection Quality.

Question: Do the guard's p-values reliably separate out-of-distribution
perturbations from in-distribution perturbations?

This is the most fundamental test of the guarded feature. If p-values cannot
discriminate OOD from in-distribution, the guard filters arbitrarily — users
get "fewer rules" without the "better" guarantee from ADR-032.

Two metrics are reported:

  auroc
    AUROC of (1 − mean_p_value_per_instance) as a binary OOD classifier
    against the ground-truth label (0 = in-distribution, 1 = OOD).
    AUROC is threshold-free, so it does not depend on the chosen significance.
    Healthy: > 0.80 for moderate+ shift. Red flag: < 0.60 for extreme shift.

  fpr_at_significance
    Fraction of in-distribution instance intervals where p_value < significance
    (i.e., the guard wrongly rejects an in-distribution perturbation).
    By conformal prediction theory this must be ≤ α. Values materially larger
    than α indicate a calibration mismatch or implementation error.

Run with --quick for a fast smoke-test (2 seeds, reduced grid).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from calibrated_explanations.ce_agent_utils import ensure_ce_first_wrapper, fit_and_calibrate

from common_guarded import (
    GuardConfig,
    compute_ood_detection_metrics,
    extract_audit_rows,
    make_gaussian_classification,
    make_ood_shift,
    mean_p_value_per_instance,
    write_report,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_N_NEIGHBORS = (1, 5, 10, 20)
DEFAULT_N_DIM = (2, 5, 10, 20)
DEFAULT_SHIFT_LEVELS = {
    "mild": 1.0,
    "moderate": 2.0,
    "extreme": 5.0,
}
DEFAULT_NORMALIZE = (True, False)
DEFAULT_SIGNIFICANCE = 0.10  # fixed for FPR reporting; AUROC is threshold-free

N_CAL = 300
N_TRAIN = 500
N_ID_TEST = 100   # in-distribution test instances
N_OOD_TEST = 100  # OOD test instances per shift level


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_shift_vector(shift_magnitude: float, n_dim: int) -> np.ndarray:
    """Shift uniformly in all directions so no single feature dominates."""
    return np.ones(n_dim) * (shift_magnitude / np.sqrt(n_dim))


def _instance_p_values(
    guarded_expl: Any,
    n_instances: int,
) -> List[float]:
    """Mean guard p-value per instance from a batch guarded explanation."""
    audit_df = extract_audit_rows(guarded_expl)
    if audit_df.empty:
        return [float("nan")] * n_instances
    mean_p = mean_p_value_per_instance(audit_df)
    # Align by instance index (0..n_instances-1 within this batch).
    result: List[float] = []
    for i in range(n_instances):
        val = mean_p.get(i, float("nan"))
        result.append(float(val))
    return result


def _run_one_config(
    wrapper: Any,
    x_id: np.ndarray,
    x_ood: np.ndarray,
    cfg: GuardConfig,
    significance: float,
) -> Dict[str, Any]:
    """Explain both ID and OOD sets with cfg; return detection metrics dict."""
    x_all = np.vstack([x_id, x_ood])
    n_id = len(x_id)
    n_ood = len(x_ood)

    t0 = time.perf_counter()
    guarded_expl = wrapper.explain_guarded_factual(
        x_all,
        significance=cfg.significance,
        n_neighbors=cfg.n_neighbors,
        merge_adjacent=cfg.merge_adjacent,
        normalize_guard=cfg.normalize_guard,
    )
    runtime_ms = (time.perf_counter() - t0) * 1000.0

    audit_df = extract_audit_rows(guarded_expl)

    # Split audit rows by ID vs OOD instance index
    p_id_series = (
        audit_df[audit_df["instance_index"] < n_id]
        .dropna(subset=["p_value"])
        .groupby("instance_index")["p_value"]
        .mean()
        if not audit_df.empty else pd.Series(dtype=float)
    )
    p_ood_series = (
        audit_df[audit_df["instance_index"] >= n_id]
        .dropna(subset=["p_value"])
        .groupby("instance_index")["p_value"]
        .mean()
        if not audit_df.empty else pd.Series(dtype=float)
    )

    p_id = p_id_series.tolist()
    p_ood = p_ood_series.tolist()

    metrics = compute_ood_detection_metrics(p_id, p_ood, significance)
    metrics["runtime_ms_per_instance"] = runtime_ms / len(x_all) if len(x_all) > 0 else 0.0
    return metrics


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_auroc_by_dim_and_shift(df: pd.DataFrame, out_dir: Path) -> None:
    """Line plots: AUROC vs n_dim, one line per shift level, one panel per n_neighbors."""
    neighbors = sorted(df["n_neighbors"].unique())
    shift_levels = sorted(df["shift_level"].unique())
    fig, axes = plt.subplots(1, len(neighbors), figsize=(4 * len(neighbors), 4), sharey=True)
    if len(neighbors) == 1:
        axes = [axes]
    for ax, nn in zip(axes, neighbors):
        sub = df[(df["n_neighbors"] == nn) & (df["normalize_guard"])]
        for sl in shift_levels:
            grp = sub[sub["shift_level"] == sl].groupby("n_dim")["auroc"].mean().reset_index()
            ax.plot(grp["n_dim"], grp["auroc"], marker="o", label=sl)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="chance")
        ax.axhline(0.8, color="green", linestyle=":", linewidth=0.8, label="target")
        ax.set_xlabel("n_dim")
        ax.set_ylabel("AUROC")
        ax.set_title(f"n_neighbors={nn}")
        ax.set_ylim(0.4, 1.05)
        ax.legend(fontsize=7)
    fig.suptitle("AUROC by Dimensionality and Shift Level (normalize_guard=True)")
    fig.tight_layout()
    fig.savefig(out_dir / "auroc_by_dim_and_shift.png", dpi=160)
    plt.close(fig)


def _plot_fpr_vs_significance(df: pd.DataFrame, out_dir: Path) -> None:
    """FPR vs significance: shows whether the conformal validity guarantee holds."""
    # Compute FPR at a range of significance levels post-hoc from stored p-values
    # We use the fixed significance column available in the stored metrics
    fig, ax = plt.subplots(figsize=(6, 4))
    sub = df[df["normalize_guard"]].groupby(["n_dim", "shift_level"])["fpr_at_significance"].mean()
    sigs = df["significance"].unique()
    for (n_dim, sl), val in sub.items():
        ax.scatter([n_dim], [val], label=f"n_dim={n_dim}, {sl}", alpha=0.7)
    for sig in sigs:
        ax.axhline(sig, linestyle="--", linewidth=0.8, color="gray")
    ax.set_xlabel("n_dim")
    ax.set_ylabel(f"FPR @ significance={DEFAULT_SIGNIFICANCE}")
    ax.set_title("FPR on ID Intervals (should be ≤ significance)")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "fpr_vs_significance.png", dpi=160)
    plt.close(fig)


def _plot_normalize_comparison(df: pd.DataFrame, out_dir: Path) -> None:
    """AUROC with vs without normalize_guard, side-by-side."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for norm in [True, False]:
        sub = df[df["normalize_guard"] == norm].groupby("n_dim")["auroc"].mean().reset_index()
        ax.plot(sub["n_dim"], sub["auroc"], marker="o", label=f"normalize={norm}")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("n_dim")
    ax.set_ylabel("mean AUROC (all shifts, all n_neighbors)")
    ax.set_title("Effect of normalize_guard on Detection Quality")
    ax.set_ylim(0.4, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "normalize_comparison.png", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "artifacts" / "guarded" / "scenario_b",
    )
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--quick", action="store_true", help="Fast smoke-test mode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.num_seeds = 2
        n_neighbors_grid = (5,)
        n_dim_grid = (2, 10)
        shift_levels = {"moderate": 2.0, "extreme": 5.0}
        normalize_grid = (True,)
    else:
        n_neighbors_grid = DEFAULT_N_NEIGHBORS
        n_dim_grid = DEFAULT_N_DIM
        shift_levels = DEFAULT_SHIFT_LEVELS
        normalize_grid = DEFAULT_NORMALIZE

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []

    for seed in range(args.num_seeds):
        for n_dim in n_dim_grid:
            # Build calibration and training data from the base Gaussian distribution
            x_all, y_all = make_gaussian_classification(
                n=N_TRAIN + N_CAL + N_ID_TEST, n_dim=n_dim, seed=seed * 100 + n_dim
            )
            x_train = x_all[:N_TRAIN]
            y_train = y_all[:N_TRAIN]
            x_cal = x_all[N_TRAIN:N_TRAIN + N_CAL]
            y_cal = y_all[N_TRAIN:N_TRAIN + N_CAL]
            x_id_test = x_all[N_TRAIN + N_CAL:]

            model = RandomForestClassifier(
                n_estimators=100, max_depth=6, random_state=seed, n_jobs=-1
            )
            wrapper = ensure_ce_first_wrapper(model)
            fit_and_calibrate(wrapper, x_train, y_train, x_cal, y_cal)

            for shift_name, shift_magnitude in shift_levels.items():
                shift_vec = _build_shift_vector(shift_magnitude, n_dim)
                x_ood_test = make_ood_shift(x_id_test, shift_vec, seed=seed * 777 + n_dim)

                for nn in n_neighbors_grid:
                    for normalize in normalize_grid:
                        cfg = GuardConfig(
                            significance=DEFAULT_SIGNIFICANCE,
                            n_neighbors=min(nn, N_CAL - 1),
                            normalize_guard=normalize,
                        )
                        try:
                            metrics = _run_one_config(
                                wrapper, x_id_test, x_ood_test, cfg, DEFAULT_SIGNIFICANCE
                            )
                        except Exception as exc:  # noqa: BLE001
                            metrics = {
                                "auroc": float("nan"),
                                "fpr_at_significance": float("nan"),
                                "n_id": len(x_id_test),
                                "n_ood": len(x_ood_test),
                                "median_p_id": float("nan"),
                                "median_p_ood": float("nan"),
                                "runtime_ms_per_instance": float("nan"),
                                "error": str(exc),
                            }

                        rows.append({
                            "seed": seed,
                            "n_dim": n_dim,
                            "shift_level": shift_name,
                            "shift_magnitude": shift_magnitude,
                            "n_neighbors": cfg.n_neighbors,
                            "normalize_guard": normalize,
                            "significance": DEFAULT_SIGNIFICANCE,
                            **metrics,
                        })

    df = pd.DataFrame(rows)
    csv_path = out_dir / "ood_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote: {csv_path}")

    # Plots
    _plot_auroc_by_dim_and_shift(df, out_dir)
    _plot_normalize_comparison(df, out_dir)
    _plot_fpr_vs_significance(df, out_dir)
    print(f"Wrote plots to: {out_dir}")

    # Report
    by_shift = df.groupby(["shift_level", "n_dim"])[["auroc", "fpr_at_significance"]].mean()
    norm_comp = df.groupby("normalize_guard")["auroc"].mean()

    report_sections = [
        (
            "What this scenario tests",
            (
                "Scenario B asks: can the guard's p-values reliably distinguish "
                "out-of-distribution perturbations from in-distribution perturbations?\n\n"
                "Data: calibration and training from N(0, I_d). "
                "In-distribution test instances also from N(0, I_d). "
                "OOD instances are N(0, I_d) + shift_vector, with shift magnitude "
                "= 1σ (mild), 2σ (moderate), 5σ (extreme).\n\n"
                "For each test instance, explain_guarded_factual is called and the "
                "mean p-value across all audit intervals is extracted per instance. "
                "AUROC treats (1 − mean_p_value) as the anomaly score against the "
                "ground-truth OOD label. FPR is measured at the instance level: "
                "the fraction of in-distribution instances whose mean interval "
                "p-value falls below significance."
            ),
        ),
        (
            "AUROC by shift level and dimensionality",
            f"Mean AUROC (averaged over seeds and n_neighbors, normalize_guard=True):\n\n"
            f"{by_shift.to_markdown()}\n\n"
            "AUROC > 0.80: guard is reliably detecting OOD perturbations for this config.\n"
            "AUROC < 0.60: guard is near-random — flagging OOD and in-distribution alike.",
        ),
        (
            "Effect of normalize_guard",
            f"Mean AUROC by normalize_guard (all dims and shifts):\n\n"
            f"{norm_comp.to_markdown()}\n\n"
            "If normalize_guard=False substantially lowers AUROC, features at different "
            "scales dominate the KNN distance and destroy detection quality.",
        ),
        (
            "FPR at significance",
            f"Mean FPR@{DEFAULT_SIGNIFICANCE} across all configurations:\n\n"
            f"{df.groupby('n_dim')['fpr_at_significance'].mean().to_markdown()}\n\n"
            f"By conformal validity, FPR should be ≤ {DEFAULT_SIGNIFICANCE}. "
            "Values materially higher indicate the guard rejects more in-distribution "
            "perturbations than the theory guarantees.",
        ),
        (
            "Known blind spots exposed by this scenario",
            (
                "1. **Curse of dimensionality**: AUROC degrades as n_dim increases because "
                "KNN distances concentrate. This defines where the guard's design breaks down.\n"
                "2. **normalize_guard=False**: If a dominant-scale feature swamps the distance "
                "metric, AUROC will be near 0.5 even for large shifts.\n"
                "3. **n_neighbors=1 instability**: High AUROC variance across seeds signals "
                "unreliable guard behavior.\n"
                "4. **Minimum p-value granularity**: p-values are discrete with step 1/n_cal. "
                "With n_cal=300 the minimum possible p-value is ~0.003. Setting significance < "
                "1/n_cal means the guard can never reject anything — FPR will be exactly 0."
            ),
        ),
    ]
    write_report(
        out_dir / "report.md",
        "Scenario B: OOD Detection Quality",
        report_sections,
    )
    print(f"Wrote: {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
