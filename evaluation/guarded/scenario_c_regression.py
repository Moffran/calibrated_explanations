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
  guard_direct_ood_auroc
    Direct test of InDistributionGuard.is_conforming() on raw test instances
    (before any feature substitution).  For instances with |x| > 3.5 the guard
    should flag more as non-conforming than for |x| ≤ 3 instances.

    NOTE: the guarded *explain loop* tests perturbed instances (one feature
    replaced by a calibration representative), so it cannot detect OOD-ness
    when that replacement covers the only source of distributional shift.  For
    a 1-feature dataset the explain-loop removal rate is mathematically
    identical for OOD and ID instances regardless of guard correctness.
    The direct is_conforming() check bypasses this structural limitation and
    is the correct test for input-OOD sensitivity.

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
from calibrated_explanations.utils.distribution_guard import InDistributionGuard

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
# Direct guard OOD sensitivity check
# ---------------------------------------------------------------------------

def _direct_guard_ood_sensitivity(
    x_cal: np.ndarray,
    x_test: np.ndarray,
    is_ood: np.ndarray,
    significance: float,
    n_neighbors: int,
) -> Tuple[float, float, bool]:
    """Test InDistributionGuard.is_conforming() on raw (unperturbed) instances.

    The guarded explain loop tests perturbed instances (one feature replaced
    by a cal representative), so it cannot detect OOD-ness when that replacement
    covers the only source of distributional shift.  This function bypasses
    the explain loop and calls is_conforming() directly on the raw test vectors,
    which is the correct test for input-OOD sensitivity.

    Returns
    -------
    frac_conforming_ood : float
        Fraction of OOD instances accepted as in-distribution by the guard.
    frac_conforming_id : float
        Fraction of ID instances accepted as in-distribution by the guard.
    ood_is_stricter : bool
        True iff OOD instances are accepted at a lower rate than ID ones.
    """
    guard = InDistributionGuard(x_cal, n_neighbors=n_neighbors)
    conforming = guard.is_conforming(x_test, significance=significance)
    ood_mask = is_ood.astype(bool)
    frac_ood = float(conforming[ood_mask].mean()) if ood_mask.any() else float("nan")
    frac_id = float(conforming[~ood_mask].mean()) if (~ood_mask).any() else float("nan")
    ood_stricter = (
        (not pd.isna(frac_ood))
        and (not pd.isna(frac_id))
        and frac_ood < frac_id
    )
    return frac_ood, frac_id, ood_stricter


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
    """Parse Scenario C command-line arguments.

    Parameters exposed to the caller
    --------------------------------
    --output-dir : pathlib.Path
        Destination for invariant reports, CSVs, and plots.
    --num-seeds : int, default=5
        Number of repeated train/cal/test splits for both the synthetic and the
        diabetes dataset. Useful range: 5-10 for full checks, 2 for smoke runs.
    --quick : bool
        Reduces the significance and neighbor grids to a single representative
        value. Suitable for fast regression checks only.
    """
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
    direct_guard_rows: List[dict] = []

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

        # Direct guard OOD check — tests raw instances, not explain-loop probes.
        # Done once per seed per config (model-independent: uses x_cal directly).
        for cfg in configs:
            frac_ood, frac_id, ood_stricter = _direct_guard_ood_sensitivity(
                x_cal, x_test_syn, is_ood_syn,
                significance=cfg.significance,
                n_neighbors=cfg.n_neighbors,
            )
            direct_guard_rows.append({
                "seed": seed,
                "significance": cfg.significance,
                "n_neighbors": cfg.n_neighbors,
                "frac_conforming_ood": frac_ood,
                "frac_conforming_id": frac_id,
                "ood_is_stricter": ood_stricter,
            })

        for model_name, model in [("rf_reg", RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=1)),
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

        for model_name, model in [("rf_reg", RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=1))]:
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

    # Direct guard OOD sensitivity — aggregated across seeds and configs.
    direct_guard_df = pd.DataFrame(direct_guard_rows)
    direct_guard_path = out_dir / "direct_guard_ood.csv"
    direct_guard_df.to_csv(direct_guard_path, index=False)
    print(f"Wrote: {direct_guard_path}")

    if not direct_guard_df.empty:
        mean_frac_ood = float(direct_guard_df["frac_conforming_ood"].mean())
        mean_frac_id = float(direct_guard_df["frac_conforming_id"].mean())
        direct_ood_passes = bool(direct_guard_df["ood_is_stricter"].mean() >= 0.5)
    else:
        mean_frac_ood = mean_frac_id = float("nan")
        direct_ood_passes = False

    report_sections = [
        (
            "Setup",
            (
                f"- Seeds: {args.num_seeds}\n"
                f"- Datasets: synthetic_1d, diabetes\n"
                f"- Models: RandomForestRegressor, Ridge (synthetic only), RandomForestRegressor (diabetes)\n"
                f"- Guard grid: significance={list(sig_grid)}, n_neighbors={list(nn_grid)}"
            ),
        ),
        (
            "Purpose",
            (
                "Scenario C asks: does the regression-specific code path preserve "
                "the interval invariant (low ≤ predict ≤ high), AND does the guard "
                "respond to actual OOD shift?\n\n"
                "Two datasets: (1) synthetic sin(x) + noise with known OOD boundary "
                "at |x| > 3.5; (2) sklearn diabetes dataset.\n"
                "Models: RandomForestRegressor and Ridge.\n"
                "explain_guarded_factual is called for all configurations."
            ),
        ),
        (
            "Metric contract",
            (
                "The paper-relevant result in this scenario is binary: either the "
                "regression path preserves the interval invariant everywhere or it does "
                "not. Any non-zero invariant violation count is a correctness bug.\n\n"
                "The OOD-responsiveness check is retained as a secondary engineering "
                "diagnostic. It answers whether the regression guard is merely "
                "well-formed or also directionally sensible when test inputs move "
                "outside the calibration support."
            ),
        ),
        (
            "Interval invariant violations",
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
            "Secondary diagnostic: direct guard OOD sensitivity",
            (
                "Test: InDistributionGuard.is_conforming(x_raw) on synthetic_1d.\n"
                "OOD instances (|x| > 3.5) must be accepted at a lower rate than "
                "ID instances (|x| <= 3.0) in the majority of seed/config runs.\n\n"
                f"Mean frac conforming — OOD: {mean_frac_ood:.3f}, "
                f"ID: {mean_frac_id:.3f}\n\n"
                + (
                    "PASS: guard accepts OOD instances less often than ID instances "
                    "in the majority of runs — direct OOD sensitivity confirmed."
                    if direct_ood_passes else
                    "FAIL: guard does not consistently accept OOD instances less than "
                    "ID instances. Check n_neighbors, significance, and normalization."
                )
                + "\n\nNote: the explain-loop removal rate (intervals_removed_guard) "
                "is structurally identical for OOD and ID on 1D data because every "
                "non-factual probe replaces the single feature with a calibration "
                "representative, making it in-distribution by construction. "
                "See direct_guard_ood.csv for full per-seed breakdown."
            ),
        ),
        (
            "Interpretation",
            (
                "Scenario C is not meant to show that regression is a separate flagship "
                "result. Its role is to ensure the regression-specific implementation is "
                "not silently broken while the classification scenarios pass.\n\n"
                "A clean run here supports a narrow claim: the guarded regression path "
                "maintains interval semantics and reacts in the right direction under a "
                "simple synthetic OOD shift. It should be treated as appendix-strength "
                "validation unless the paper explicitly argues about regression."
            ),
        ),
    ]
    write_report(out_dir / "report.md", "Scenario C: Regression Invariants", report_sections)
    print(f"Wrote: {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
