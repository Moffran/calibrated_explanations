"""Scenario C — Regression Invariants.

Question: Does the guard preserve the interval invariant (low ≤ predict ≤ high)
in regression tasks, where a different internal code path is exercised?

The guarded explain code has separate handling for regression vs. classification:
different discretizer criteria, different interval semantics, and crucially a
``warnings.warn`` path (instead of ``raise``) for interval violations. A
regression-specific implementation error would be invisible to Scenario A, which
is classification-only.

One metric is reported:

  n_invariant_violations
    Count of audit interval records where predict < low or predict > high
    (with ε = 1e-6 tolerance). This MUST be zero for every configuration.
    Any non-zero count is a bug — and because the regression code path uses
    warnings.warn instead of raise, violations silently pass in normal use.

Secondary diagnostic (recorded in CSV, not elevated as a finding):
  fraction_removed_ood vs fraction_removed_id
    For the synthetic dataset, instances with |x| > 3.5 are OOD relative
    to the calibration range |x| ≤ 3. The guard should remove more intervals
    for OOD instances than for in-distribution ones, confirming responsiveness
    to actual distributional shift in regression.

Run with --quick for a fast smoke-test (2 seeds, reduced grid).
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from calibrated_explanations.ce_agent_utils import ensure_ce_first_wrapper, fit_and_calibrate

from common_guarded import (
    GuardConfig,
    check_interval_invariant,
    extract_audit_rows,
    extract_audit_summary_rows,
    make_splits,
    write_report,
)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_SIGNIFICANCE = (0.05, 0.10, 0.20)
DEFAULT_N_NEIGHBORS = (3, 5, 10)


# ---------------------------------------------------------------------------
# Synthetic regression dataset
# ---------------------------------------------------------------------------

def make_regression_1d(
    n: int, seed: int, ood_boundary: float = 3.0
) -> Tuple[np.ndarray, np.ndarray]:
    """y = sin(x) + noise, x uniform in [-ood_boundary, ood_boundary]."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-ood_boundary, ood_boundary, size=(n, 1))
    y = np.sin(x[:, 0]) + rng.normal(0.0, 0.15, size=n)
    return x, y


def make_ood_regression_test(
    n_id: int, n_ood: int, seed: int, ood_boundary: float = 3.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (x_test, is_ood) where OOD instances have |x| > ood_boundary + 0.5."""
    rng = np.random.default_rng(seed)
    x_id = rng.uniform(-ood_boundary, ood_boundary, size=(n_id, 1))
    x_ood_pos = rng.uniform(ood_boundary + 0.5, ood_boundary + 2.0, size=(n_ood // 2, 1))
    x_ood_neg = rng.uniform(-ood_boundary - 2.0, -ood_boundary - 0.5, size=(n_ood - n_ood // 2, 1))
    x_test = np.vstack([x_id, x_ood_pos, x_ood_neg])
    is_ood = np.array([False] * n_id + [True] * n_ood)
    return x_test, is_ood


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def _evaluate_config(
    wrapper,
    x_test: np.ndarray,
    cfg: GuardConfig,
    dataset_name: str,
    model_name: str,
    seed: int,
    is_ood: np.ndarray | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run guarded factual explain and return (interval_df, summary_df).

    Captures any warnings emitted during the call (interval invariant warnings
    from the regression code path appear here).
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        guarded_expl = wrapper.explain_guarded_factual(
            x_test,
            significance=cfg.significance,
            n_neighbors=cfg.n_neighbors,
            merge_adjacent=cfg.merge_adjacent,
            normalize_guard=cfg.normalize_guard,
        )

    audit_df = extract_audit_rows(guarded_expl)
    summary_df = extract_audit_summary_rows(guarded_expl)

    # Attach metadata
    for df in (audit_df, summary_df):
        if not df.empty:
            df["dataset"] = dataset_name
            df["model"] = model_name
            df["seed"] = seed
            df["significance"] = cfg.significance
            df["n_neighbors"] = cfg.n_neighbors
            df["n_invariant_warnings"] = len([
                w for w in caught if issubclass(w.category, UserWarning)
                and "invariant" in str(w.message).lower()
            ])

    # Tag OOD flag for the secondary diagnostic
    if is_ood is not None and not summary_df.empty:
        summary_df["is_ood"] = summary_df["instance_index"].map(
            lambda i: bool(is_ood[i]) if i < len(is_ood) else False
        )

    return audit_df, summary_df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_fraction_removed(df: pd.DataFrame, out_dir: Path) -> None:
    """Show fraction_removed for ID vs OOD instances on synthetic dataset."""
    synth = df[df["dataset"] == "synthetic_1d"]
    if synth.empty or "is_ood" not in synth.columns or "intervals_removed_guard" not in synth.columns:
        return
    synth = synth.copy()
    synth["fraction_removed"] = (
        synth["intervals_removed_guard"] / synth["intervals_tested"].clip(lower=1)
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    for ood_flag, label in [(False, "in-distribution"), (True, "OOD (|x|>3.5)")]:
        sub = synth[synth["is_ood"] == ood_flag]
        means = sub.groupby("significance")["fraction_removed"].mean()
        ax.plot(means.index, means.values, marker="o", label=label)
    ax.set_xlabel("significance")
    ax.set_ylabel("mean fraction of intervals removed by guard")
    ax.set_title("Secondary Diagnostic: Guard Responsiveness (synthetic 1D)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fraction_removed_id_vs_ood.png", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "artifacts" / "guarded" / "scenario_c",
    )
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--quick", action="store_true", help="Fast smoke-test mode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.num_seeds = 2
        sig_grid = (0.10,)
        nn_grid = (5,)
    else:
        sig_grid = DEFAULT_SIGNIFICANCE
        nn_grid = DEFAULT_N_NEIGHBORS

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    all_audit_rows: List[pd.DataFrame] = []
    all_summary_rows: List[pd.DataFrame] = []

    configs = [
        GuardConfig(significance=s, n_neighbors=nn)
        for s in sig_grid
        for nn in nn_grid
    ]

    for seed in range(args.num_seeds):
        # ----------------------------------------------------------------
        # Dataset 1: synthetic 1D regression with known OOD boundary
        # ----------------------------------------------------------------
        x_syn, y_syn = make_regression_1d(n=2000, seed=seed * 10 + 1, ood_boundary=3.0)
        x_test_syn, is_ood_syn = make_ood_regression_test(
            n_id=60, n_ood=40, seed=seed * 10 + 2, ood_boundary=3.0
        )
        n_cal_syn = 400
        n_train_syn = 1200
        try:
            x_tr, y_tr, x_cal, y_cal, _, _ = make_splits(
                x_syn, y_syn, n_train=n_train_syn, n_cal=n_cal_syn, n_test=400,
                seed=seed, stratify=False
            )
        except ValueError:
            x_tr, y_tr = x_syn[:n_train_syn], y_syn[:n_train_syn]
            x_cal, y_cal = x_syn[n_train_syn:n_train_syn + n_cal_syn], y_syn[n_train_syn:n_train_syn + n_cal_syn]

        for model_name, model in [("rf_reg", RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)),
                                    ("ridge", Ridge(alpha=1.0))]:
            wrapper = ensure_ce_first_wrapper(model)
            fit_and_calibrate(wrapper, x_tr, y_tr, x_cal, y_cal)

            for cfg in configs:
                audit_df, summary_df = _evaluate_config(
                    wrapper, x_test_syn, cfg,
                    dataset_name="synthetic_1d", model_name=model_name,
                    seed=seed, is_ood=is_ood_syn,
                )
                all_audit_rows.append(audit_df)
                all_summary_rows.append(summary_df)

        # ----------------------------------------------------------------
        # Dataset 2: sklearn diabetes (real regression)
        # ----------------------------------------------------------------
        diabetes = load_diabetes()
        x_dia, y_dia = diabetes.data, diabetes.target
        n_dia = len(x_dia)
        n_cal_dia = min(100, n_dia // 4)
        n_train_dia = min(250, n_dia // 2)
        n_test_dia = min(80, n_dia - n_train_dia - n_cal_dia)
        try:
            x_tr_d, y_tr_d, x_cal_d, y_cal_d, x_test_d, _ = make_splits(
                x_dia, y_dia,
                n_train=n_train_dia, n_cal=n_cal_dia, n_test=n_test_dia,
                seed=seed, stratify=False,
            )
        except ValueError:
            # Fallback: manual slice
            x_tr_d, y_tr_d = x_dia[:n_train_dia], y_dia[:n_train_dia]
            x_cal_d = x_dia[n_train_dia:n_train_dia + n_cal_dia]
            y_cal_d = y_dia[n_train_dia:n_train_dia + n_cal_dia]
            x_test_d = x_dia[n_train_dia + n_cal_dia:n_train_dia + n_cal_dia + n_test_dia]

        for model_name, model in [("rf_reg", RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1))]:
            wrapper = ensure_ce_first_wrapper(model)
            fit_and_calibrate(wrapper, x_tr_d, y_tr_d, x_cal_d, y_cal_d)

            for cfg in configs:
                audit_df, summary_df = _evaluate_config(
                    wrapper, x_test_d, cfg,
                    dataset_name="diabetes", model_name=model_name,
                    seed=seed,
                )
                all_audit_rows.append(audit_df)
                all_summary_rows.append(summary_df)

    # ----------------------------------------------------------------
    # Combine and save
    # ----------------------------------------------------------------
    audit_all = pd.concat([df for df in all_audit_rows if not df.empty], ignore_index=True)
    summary_all = pd.concat([df for df in all_summary_rows if not df.empty], ignore_index=True)

    # Primary metric: interval invariant violations
    violations_df = check_interval_invariant(audit_all)
    violations_path = out_dir / "invariant_violations.csv"
    violations_df.to_csv(violations_path, index=False)
    print(f"Wrote: {violations_path} ({len(violations_df)} violations)")

    metrics_path = out_dir / "regression_metrics.csv"
    summary_all.to_csv(metrics_path, index=False)
    print(f"Wrote: {metrics_path}")

    _plot_fraction_removed(summary_all, out_dir)
    print(f"Wrote plots to: {out_dir}")

    # Build summary stats
    n_violations = len(violations_df)
    n_warnings = (
        int(summary_all["n_invariant_warnings"].sum())
        if "n_invariant_warnings" in summary_all.columns else 0
    )
    frac_removed_by_split = (
        summary_all.groupby(["dataset", "is_ood"])["intervals_removed_guard"].mean()
        if "is_ood" in summary_all.columns else None
    )

    report_sections = [
        (
            "What this scenario tests",
            (
                "Scenario C asks: does the regression-specific code path preserve "
                "the interval invariant (low ≤ predict ≤ high)?\n\n"
                "Two datasets: (1) synthetic sin(x) + noise with known OOD boundary "
                "at |x| > 3.5; (2) sklearn diabetes dataset.\n"
                "Models: RandomForestRegressor and Ridge.\n"
                "explain_guarded_factual is called for all configurations."
            ),
        ),
        (
            "Primary metric: interval invariant violations",
            (
                f"**n_invariant_violations = {n_violations}**\n\n"
                + (
                    "PASS: no violations found. The interval invariant holds everywhere."
                    if n_violations == 0
                    else (
                        f"FAIL: {n_violations} violation(s) found. "
                        "This is a bug — see invariant_violations.csv for details."
                    )
                )
                + f"\n\nInterval invariant warnings captured: {n_warnings}. "
                "The regression code path uses warnings.warn instead of raise, so "
                "violations would go unnoticed in normal use without this check."
            ),
        ),
        (
            "Secondary diagnostic: guard responsiveness to OOD",
            (
                "Mean intervals_removed_guard for ID vs OOD instances on synthetic dataset.\n\n"
                + (frac_removed_by_split.to_markdown() if frac_removed_by_split is not None else "N/A")
                + "\n\nOOD instances (|x| > 3.5) are expected to have more intervals "
                "removed than ID instances (|x| ≤ 3). A non-zero difference supports "
                "guard responsiveness; zero for both may indicate the OOD shift is "
                "insufficient relative to the calibration distribution at this significance."
            ),
        ),
    ]
    write_report(out_dir / "report.md", "Scenario C: Regression Invariants", report_sections)
    print(f"Wrote: {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
