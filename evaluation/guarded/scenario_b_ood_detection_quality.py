"""Scenario B: OOD Detection Quality.

Question: Do the guard's p-values reliably separate out-of-distribution
perturbations from in-distribution perturbations?

This is the most fundamental test of the guarded feature. If p-values cannot
discriminate OOD from in-distribution, the guard filters arbitrarily and users
get "fewer rules" without the "better" guarantee from ADR-032.

Two metrics are reported:

  auroc
    AUROC of (1 - fisher_combined_p_value_per_instance) as a binary OOD
    classifier against the ground-truth label (0 = in-distribution,
    1 = OOD). AUROC is threshold-free, so it does not depend on the chosen
    significance.
    Healthy: > 0.80 for moderate+ shift. Red flag: < 0.60 for extreme shift.

  fpr_at_significance
    Fraction of in-distribution audit intervals where raw p_value < significance.
    This is an interval-level rejection rate, not a rejection rate on combined
    instance scores. Values materially larger than significance indicate that
    the empirical interval-level behaviour is more aggressive than intended.

Run with --quick for a fast smoke-test (2 seeds, reduced grid).

Paper-focused execution:
    --paper-focused
        Restricts to paper-facing defaults (normalize_guard=True, n_neighbors=5),
        with the AUROC and interval-level FPR diagnostics used in the manuscript.
    --large
        Enables paper-focused mode and scales synthetic train/cal/test sizes and
        seeds for in-large evidence runs.
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
    fisher_p_value_per_instance,
    make_gaussian_classification,
    make_ood_shift,
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

# FPR is allowed to exceed significance by at most this factor before being
# flagged as a red-flag (conformal guarantee allows slight over-coverage on
# finite calibration sets, but material excess signals an implementation error).
FPR_TOLERANCE_FACTOR: float = 1.5

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

    # Split audit rows by ID vs OOD instance index; combine with Fisher's method.
    p_id_series = (
        fisher_p_value_per_instance(audit_df[audit_df["instance_index"] < n_id])
        if not audit_df.empty else pd.Series(dtype=float)
    )
    p_ood_series = (
        fisher_p_value_per_instance(audit_df[audit_df["instance_index"] >= n_id])
        if not audit_df.empty else pd.Series(dtype=float)
    )

    p_id = p_id_series.tolist()
    p_ood = p_ood_series.tolist()

    metrics = compute_ood_detection_metrics(p_id, p_ood)
    id_interval_p = (
        pd.to_numeric(
            audit_df.loc[audit_df["instance_index"] < n_id, "p_value"],
            errors="coerce",
        ).dropna().to_numpy()
        if not audit_df.empty and "p_value" in audit_df.columns
        else np.array([], dtype=float)
    )
    metrics["fpr_at_significance"] = (
        float(np.mean(id_interval_p < significance))
        if len(id_interval_p) > 0 else float("nan")
    )
    metrics["n_id_intervals"] = int(len(id_interval_p))
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


def _plot_fpr_vs_significance(df: pd.DataFrame, out_dir: Path, significance: float) -> None:
    """Plot interval-level rejection rate at the configured significance."""
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
    ax.set_ylabel(f"FPR @ significance={significance}")
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
    """Parse Scenario B command-line arguments.

    Parameters exposed to the caller
    --------------------------------
    --output-dir : pathlib.Path
        Destination for the OOD metrics CSV, plots, and report.
    --num-seeds : int, default=5
        Number of repeated synthetic draws per grid point. Useful range: 5-20
        for stable AUROC trends, 2 for smoke runs.
    --quick : bool
        Uses a reduced grid over dimensionality, shift size, and neighbors.
        Appropriate for CI smoke checks only.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "artifacts" / "guarded" / "scenario_b",
    )
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--n-train", type=int, default=N_TRAIN)
    parser.add_argument("--n-cal", type=int, default=N_CAL)
    parser.add_argument("--n-id-test", type=int, default=N_ID_TEST)
    parser.add_argument("--n-ood-test", type=int, default=N_OOD_TEST)
    parser.add_argument(
        "--paper-focused",
        action="store_true",
        help=(
            "Restrict grids and outputs to paper-facing defaults "
            "(normalize_guard=True, n_neighbors=5)."
        ),
    )
    parser.add_argument(
        "--large",
        action="store_true",
        help=(
            "Enable a large synthetic run profile and paper-focused mode for "
            "stronger in-large evidence."
        ),
    )
    parser.add_argument("--quick", action="store_true", help="Fast smoke-test mode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.large:
        args.paper_focused = True
        args.num_seeds = max(args.num_seeds, 10)
        args.n_train = max(args.n_train, 5000)
        args.n_cal = max(args.n_cal, 3000)
        args.n_id_test = max(args.n_id_test, 1000)
        args.n_ood_test = max(args.n_ood_test, 1000)

    if args.quick:
        args.num_seeds = 2
        n_neighbors_grid = (5,)
        n_dim_grid = (2, 10)
        shift_levels = {"moderate": 2.0, "extreme": 5.0}
        normalize_grid = (True,)
        n_train = min(args.n_train, 500)
        n_cal = min(args.n_cal, 300)
        n_id_test = min(args.n_id_test, 100)
        n_ood_test = min(args.n_ood_test, 100)
    elif args.paper_focused:
        n_neighbors_grid = (5,)
        n_dim_grid = DEFAULT_N_DIM
        shift_levels = DEFAULT_SHIFT_LEVELS
        normalize_grid = (True,)
        n_train = args.n_train
        n_cal = args.n_cal
        n_id_test = args.n_id_test
        n_ood_test = args.n_ood_test
    else:
        n_neighbors_grid = DEFAULT_N_NEIGHBORS
        n_dim_grid = DEFAULT_N_DIM
        shift_levels = DEFAULT_SHIFT_LEVELS
        normalize_grid = DEFAULT_NORMALIZE
        n_train = args.n_train
        n_cal = args.n_cal
        n_id_test = args.n_id_test
        n_ood_test = args.n_ood_test

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []

    for seed in range(args.num_seeds):
        for n_dim in n_dim_grid:
            # Build calibration and training data from the base Gaussian distribution
            x_all, y_all = make_gaussian_classification(
                n=n_train + n_cal + n_id_test, n_dim=n_dim, seed=seed * 100 + n_dim
            )
            x_train = x_all[:n_train]
            y_train = y_all[:n_train]
            x_cal = x_all[n_train:n_train + n_cal]
            y_cal = y_all[n_train:n_train + n_cal]
            x_id_test = x_all[n_train + n_cal:]

            model = RandomForestClassifier(
                n_estimators=100, max_depth=6, random_state=seed, n_jobs=1
            )
            wrapper = ensure_ce_first_wrapper(model)
            fit_and_calibrate(wrapper, x_train, y_train, x_cal, y_cal)

            for shift_name, shift_magnitude in shift_levels.items():
                shift_vec = _build_shift_vector(shift_magnitude, n_dim)
                x_ood_test = make_ood_shift(x_id_test, shift_vec, seed=seed * 777 + n_dim)
                if len(x_ood_test) > n_ood_test:
                    x_ood_test = x_ood_test[:n_ood_test]

                for nn in n_neighbors_grid:
                    for normalize in normalize_grid:
                        cfg = GuardConfig(
                            significance=DEFAULT_SIGNIFICANCE,
                            n_neighbors=min(nn, n_cal - 1),
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
                                "median_combined_p_id": float("nan"),
                                "median_combined_p_ood": float("nan"),
                                "n_id_intervals": 0,
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
    if not args.paper_focused:
        _plot_normalize_comparison(df, out_dir)
    _plot_fpr_vs_significance(df, out_dir, DEFAULT_SIGNIFICANCE)
    print(f"Wrote plots to: {out_dir}")

    paper_df = df[(df["normalize_guard"] == True) & (df["n_neighbors"] == 5)].copy()  # noqa: E712
    if paper_df.empty:
        paper_df = df[df["normalize_guard"] == True].copy()  # noqa: E712
    if paper_df.empty:
        paper_df = df.copy()

    # Interval-level rejection-rate check on the default paper-facing slice.
    fpr_threshold = DEFAULT_SIGNIFICANCE * FPR_TOLERANCE_FACTOR
    fpr_violations = paper_df[
        paper_df["fpr_at_significance"].notna()
        & (paper_df["fpr_at_significance"] > fpr_threshold)
    ]
    fpr_red_flag = len(fpr_violations) > 0

    # Report
    by_shift = paper_df.groupby(["shift_level", "n_dim"])[["auroc"]].mean()
    fpr_by_dim = paper_df.groupby("n_dim")["fpr_at_significance"].mean()
    paper_n_neighbors = sorted(paper_df["n_neighbors"].dropna().unique().tolist())

    report_sections = [
        (
            "Setup",
            (
                f"- Seeds: {args.num_seeds}\n"
                f"- Calibration size: {n_cal}\n"
                f"- Train size: {n_train}\n"
                f"- ID test size: {n_id_test}\n"
                f"- OOD test size per shift level: {n_ood_test}\n"
                f"- Paper-facing slice: normalize_guard=True, n_neighbors={paper_n_neighbors}, significance={DEFAULT_SIGNIFICANCE}"
            ),
        ),
        (
            "Purpose",
            (
                "Scenario B asks: can the guard's p-values reliably distinguish "
                "out-of-distribution perturbations from in-distribution perturbations?\n\n"
                "Data: calibration and training from N(0, I_d). "
                "In-distribution test instances also from N(0, I_d). "
                "OOD instances are N(0, I_d) + shift_vector, with shift magnitude "
                "= 1σ (mild), 2σ (moderate), 5σ (extreme).\n\n"
                "For each test instance, explain_guarded_factual is called and the "
                "interval-level guard p-values are combined into one Fisher "
                "p-value per instance. AUROC treats 1 - p_combined as the anomaly "
                "score against the ground-truth OOD label. The rejection-rate "
                "diagnostic is computed separately from raw interval-level p-values "
                "on in-distribution audit rows.\n\n"
                f"The paper-facing slice of this scenario uses normalize_guard=True "
                f"and n_neighbors={paper_n_neighbors}."
            ),
        ),
        (
            "Metric contract",
            (
                "The primary metric is AUROC computed from Fisher-combined per-instance "
                "guard p-values. This is the direct detection-quality result because it "
                "measures ranking quality without depending on a specific threshold.\n\n"
                "The secondary diagnostic is the interval-level rejection rate on "
                "in-distribution audit rows at the configured significance. It is not a "
                "valid statement about Fisher-combined instance scores and should be read "
                "only as a calibration-style sanity check for raw interval decisions."
            ),
        ),
        (
            "AUROC by shift level and dimensionality",
            f"Mean AUROC on the paper-facing slice:\n\n"
            f"{by_shift.to_markdown()}\n\n"
            "AUROC above 0.80 for moderate or extreme shift indicates useful "
            "separation. AUROC near 0.50 indicates that the guard is close to "
            "random on this synthetic shift task.",
        ),
        (
            "Interval-level rejection rate on in-distribution rows",
            f"Mean rejection rate at significance={DEFAULT_SIGNIFICANCE} on the "
            f"paper-facing slice:\n\n"
            f"{fpr_by_dim.to_markdown()}\n\n"
            f"We flag configurations where the empirical interval-level rejection "
            f"rate exceeds {FPR_TOLERANCE_FACTOR:.1f} times the nominal threshold "
            f"({fpr_threshold:.3f}). "
            + (
                f"{len(fpr_violations)} configuration(s) exceeded that bound in the "
                "paper-facing slice."
                if fpr_red_flag else
                "No configuration in the paper-facing slice exceeded that bound."
            ),
        ),
        (
            "Interpretation",
            (
                "Higher dimensionality makes KNN distance concentration more severe, so "
                "AUROC should be expected to degrade as n_dim grows. Mild shifts are "
                "allowed to be difficult; the important question is whether moderate and "
                "extreme shifts remain separable.\n\n"
                "normalize_guard=False and n_neighbors=1 remain in the CSV and plots as "
                "engineering stress tests. They are useful for understanding failure "
                "modes, but they should not be promoted to headline evidence because the "
                "paper claim is about the default guarded configuration.\n\n"
                "p-values are discrete with step 1/n_cal. Extremely small significance "
                "levels can therefore make the raw interval-level rejection diagnostic "
                "look artificially inactive even when AUROC remains informative."
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
