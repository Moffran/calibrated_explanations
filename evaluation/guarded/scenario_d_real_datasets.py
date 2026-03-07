"""Scenario D — Real Dataset Correctness.

Question: Does the guard's API remain correct (no exceptions, complete audit
payloads) across the full variety of real-world task types — multiclass
classification, high-dimensional data, small calibration sets?

Synthetic scenarios are designed around the guard's assumptions. Real datasets
may violate those assumptions in ways we did not anticipate. This scenario is a
correctness sweep, not a performance benchmark.

Two metrics are reported:

  audit_field_completeness
    Every interval record in get_guarded_audit() must contain all fields
    defined in ADR-032 Addendum. A missing field means the payload contract is
    broken — likely a multiclass-specific bug where prediction["classes"] has a
    different shape than in the binary case.

  fraction_instances_fully_filtered
    Fraction of test instances with intervals_emitted = 0 (zero rules returned).
    The API must not crash on empty explanations. But if > 10% of instances get
    zero rules at significance=0.10, the guard is so aggressive it is impractical.
    Values > 0% should be documented and understood.

Run with --quick for a fast smoke-test (2 seeds, reduced grid, 2 datasets).
"""
from __future__ import annotations

import argparse
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.ensemble import RandomForestClassifier

from calibrated_explanations.ce_agent_utils import ensure_ce_first_wrapper, fit_and_calibrate

from common_guarded import (
    GuardConfig,
    check_audit_field_completeness,
    extract_audit_summary_rows,
    make_splits,
    write_report,
)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_SIGNIFICANCE = (0.05, 0.10, 0.20)
DEFAULT_N_NEIGHBORS = (3, 5)
DEFAULT_BONFERRONI = (False, True)


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

def load_datasets(quick: bool) -> List[Tuple[str, np.ndarray, np.ndarray, int, int, int]]:
    """Return list of (name, x, y, n_train, n_cal, n_test) tuples."""
    datasets = []

    # breast_cancer: 569 samples, 30 features, binary
    bc = load_breast_cancer()
    datasets.append(("breast_cancer", bc.data, bc.target, 280, 140, 100))

    # iris: 150 samples, 4 features, 3-class
    ir = load_iris()
    datasets.append(("iris", ir.data, ir.target, 70, 30, 30))

    if not quick:
        # wine: 178 samples, 13 features, 3-class
        wi = load_wine()
        datasets.append(("wine", wi.data, wi.target, 80, 40, 40))

        # digits (classes 0 vs 1 only): 360 samples, 64 features
        from sklearn.datasets import load_digits  # noqa: PLC0415
        dg = load_digits()
        mask = dg.target <= 1
        x_dg, y_dg = dg.data[mask], dg.target[mask]
        datasets.append(("digits_01", x_dg, y_dg, 170, 80, 60))

    return datasets


# ---------------------------------------------------------------------------
# Per-dataset evaluation
# ---------------------------------------------------------------------------

def _evaluate_dataset(
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    n_train: int,
    n_cal: int,
    n_test: int,
    configs: List[GuardConfig],
    seed: int,
) -> Tuple[List[Dict], List[Dict]]:
    """Run all configs on one dataset / seed pair.

    Returns (summary_rows, completeness_rows).
    """
    try:
        x_tr, y_tr, x_cal, y_cal, x_test, _ = make_splits(
            x, y, n_train=n_train, n_cal=n_cal, n_test=n_test,
            seed=seed, stratify=True,
        )
    except ValueError:
        # Dataset too small for requested split sizes; reduce proportionally
        n_total = len(x)
        frac_cal = n_cal / (n_train + n_cal + n_test)
        frac_test = n_test / (n_train + n_cal + n_test)
        n_cal_adj = max(5, int(n_total * frac_cal))
        n_test_adj = max(5, int(n_total * frac_test))
        n_train_adj = n_total - n_cal_adj - n_test_adj
        x_tr, y_tr, x_cal, y_cal, x_test, _ = make_splits(
            x, y, n_train=n_train_adj, n_cal=n_cal_adj, n_test=n_test_adj,
            seed=seed, stratify=True,
        )

    model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=seed, n_jobs=-1)
    wrapper = ensure_ce_first_wrapper(model)
    fit_and_calibrate(wrapper, x_tr, y_tr, x_cal, y_cal)

    n_features = x.shape[1]
    n_classes = len(np.unique(y))
    summary_rows: List[Dict] = []
    completeness_rows: List[Dict] = []

    for cfg in configs:
        # Cap n_neighbors at calibration set size
        safe_nn = min(cfg.n_neighbors, len(x_cal) - 1)
        safe_cfg = GuardConfig(
            significance=cfg.significance,
            n_neighbors=safe_nn,
            merge_adjacent=cfg.merge_adjacent,
            use_bonferroni=cfg.use_bonferroni,
            normalize_guard=cfg.normalize_guard,
        )

        try:
            guarded_expl = wrapper.explain_guarded_factual(
                x_test,
                significance=safe_cfg.significance,
                n_neighbors=safe_cfg.n_neighbors,
                merge_adjacent=safe_cfg.merge_adjacent,
                use_bonferroni=safe_cfg.use_bonferroni,
                normalize_guard=safe_cfg.normalize_guard,
            )

            # Field completeness check
            all_complete, missing_list = check_audit_field_completeness(guarded_expl)
            for missing_entry in missing_list:
                completeness_rows.append({
                    "dataset": name,
                    "seed": seed,
                    "significance": safe_cfg.significance,
                    "n_neighbors": safe_cfg.n_neighbors,
                    "use_bonferroni": safe_cfg.use_bonferroni,
                    **missing_entry,
                })

            # Summary metrics
            summary_df = extract_audit_summary_rows(guarded_expl)
            if summary_df.empty:
                continue

            n_test_actual = len(x_test)
            n_fully_filtered = int(
                (summary_df["intervals_emitted"] == 0).sum()
                if "intervals_emitted" in summary_df.columns else 0
            )
            fraction_fully_filtered = n_fully_filtered / max(1, n_test_actual)

            mean_removed = (
                summary_df["intervals_removed_guard"].mean()
                if "intervals_removed_guard" in summary_df.columns else float("nan")
            )

            summary_rows.append({
                "dataset": name,
                "n_features": n_features,
                "n_classes": n_classes,
                "n_cal": len(x_cal),
                "seed": seed,
                "significance": safe_cfg.significance,
                "n_neighbors": safe_cfg.n_neighbors,
                "use_bonferroni": safe_cfg.use_bonferroni,
                "audit_field_completeness": all_complete,
                "fraction_instances_fully_filtered": fraction_fully_filtered,
                "n_instances_fully_filtered": n_fully_filtered,
                "mean_intervals_removed_per_instance": mean_removed,
                "n_test_instances": n_test_actual,
                "error": None,
            })

        except Exception:  # noqa: BLE001
            summary_rows.append({
                "dataset": name,
                "n_features": n_features,
                "n_classes": n_classes,
                "n_cal": len(x_cal),
                "seed": seed,
                "significance": cfg.significance,
                "n_neighbors": cfg.n_neighbors,
                "use_bonferroni": cfg.use_bonferroni,
                "audit_field_completeness": False,
                "fraction_instances_fully_filtered": float("nan"),
                "n_instances_fully_filtered": -1,
                "mean_intervals_removed_per_instance": float("nan"),
                "n_test_instances": len(x_test),
                "error": traceback.format_exc().splitlines()[-1],
            })

    return summary_rows, completeness_rows


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_fraction_filtered(df: pd.DataFrame, out_dir: Path) -> None:
    """Bar chart: fraction_instances_fully_filtered by dataset at significance=0.10."""
    sub = df[(df["significance"] == 0.10) & (~df["use_bonferroni"])].copy()
    if sub.empty:
        return
    means = sub.groupby("dataset")["fraction_instances_fully_filtered"].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    means.plot(kind="bar", ax=ax, color="steelblue")
    ax.axhline(0.10, linestyle="--", color="orange", label="significance=0.10")
    ax.set_ylabel("fraction instances with 0 emitted rules")
    ax.set_title("Fraction Fully Filtered by Dataset (significance=0.10, no Bonferroni)")
    ax.set_xlabel("")
    ax.legend()
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "fraction_filtered_by_dataset.png", dpi=160)
    plt.close(fig)


def _plot_bonferroni_comparison(df: pd.DataFrame, out_dir: Path) -> None:
    """Side-by-side: mean_intervals_removed with vs without Bonferroni."""
    sub = df[df["significance"] == 0.10].copy()
    if sub.empty:
        return
    pivot = sub.groupby(["dataset", "use_bonferroni"])["mean_intervals_removed_per_instance"].mean().unstack()
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("mean intervals removed per instance")
    ax.set_title("Effect of use_bonferroni on Guard Aggressiveness (significance=0.10)")
    ax.set_xlabel("")
    ax.legend(title="use_bonferroni")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "bonferroni_comparison.png", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "artifacts" / "guarded" / "scenario_d",
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
        bonferroni_grid = (False,)
    else:
        sig_grid = DEFAULT_SIGNIFICANCE
        nn_grid = DEFAULT_N_NEIGHBORS
        bonferroni_grid = DEFAULT_BONFERRONI

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = load_datasets(quick=args.quick)
    configs = [
        GuardConfig(significance=s, n_neighbors=nn, use_bonferroni=b)
        for s in sig_grid
        for nn in nn_grid
        for b in bonferroni_grid
    ]

    all_summary: List[Dict] = []
    all_completeness: List[Dict] = []

    for seed in range(args.num_seeds):
        for name, x, y, n_train, n_cal, n_test in datasets:
            summary_rows, completeness_rows = _evaluate_dataset(
                name, x, y, n_train, n_cal, n_test, configs, seed
            )
            all_summary.extend(summary_rows)
            all_completeness.extend(completeness_rows)

    summary_df = pd.DataFrame(all_summary)
    completeness_df = pd.DataFrame(all_completeness)

    metrics_path = out_dir / "real_dataset_metrics.csv"
    summary_df.to_csv(metrics_path, index=False)
    print(f"Wrote: {metrics_path}")

    completeness_path = out_dir / "audit_completeness_details.csv"
    completeness_df.to_csv(completeness_path, index=False)
    if completeness_df.empty:
        print(f"Wrote: {completeness_path} (empty — all fields present)")
    else:
        print(f"Wrote: {completeness_path} ({len(completeness_df)} missing-field records)")

    _plot_fraction_filtered(summary_df, out_dir)
    _plot_bonferroni_comparison(summary_df, out_dir)
    print(f"Wrote plots to: {out_dir}")

    # Build dataset summary string from what was actually loaded
    dataset_desc = ", ".join(
        f"{name} ({x.shape[1]} features, {len(np.unique(y))}-class)"
        for name, x, y, *_ in datasets
    )

    # Summary stats for report
    n_missing_fields = len(completeness_df)
    error_rows = summary_df[summary_df["error"].notna()] if "error" in summary_df.columns else pd.DataFrame()
    fully_filtered_table = (
        summary_df[summary_df["significance"] == 0.10]
        .groupby("dataset")[["fraction_instances_fully_filtered", "audit_field_completeness"]]
        .mean()
        if not summary_df.empty else pd.DataFrame()
    )

    report_sections = [
        (
            "What this scenario tests",
            (
                "Scenario D asks: does the guard's API remain correct across the full "
                "variety of real-world task types — multiclass classification, "
                "high-dimensional data, and small calibration sets?\n\n"
                f"Datasets: {dataset_desc}.\n"
                "Model: RandomForestClassifier.\n"
                "Grid includes use_bonferroni=True, which is untested in Scenario A."
            ),
        ),
        (
            "Primary metric 1: audit_field_completeness",
            (
                f"Total interval records with missing fields: **{n_missing_fields}**\n\n"
                + (
                    "PASS: all required fields present in every audit interval record."
                    if n_missing_fields == 0
                    else (
                        f"FAIL: {n_missing_fields} record(s) with missing fields. "
                        "See audit_completeness_details.csv."
                    )
                )
            ),
        ),
        (
            "Primary metric 2: fraction_instances_fully_filtered",
            (
                "At significance=0.10, fraction of instances with 0 emitted rules:\n\n"
                + (fully_filtered_table.to_markdown() if not fully_filtered_table.empty else "N/A")
                + "\n\nValues > 0.10 mean the guard is more aggressive than the significance "
                "level implies — likely due to small calibration sets or high dimensionality. "
                "Values > 0.10 should be flagged in documentation."
            ),
        ),
        (
            "Crashes and API errors",
            (
                f"API exception count: **{len(error_rows)}**\n\n"
                + (
                    "PASS: no exceptions raised across any dataset or configuration."
                    if error_rows.empty
                    else f"FAIL: {len(error_rows)} exception(s). See real_dataset_metrics.csv column 'error'."
                )
            ),
        ),
        (
            "Known blind spots exposed by this scenario",
            (
                "1. **Multiclass payload shape**: iris and wine exercise the multiclass "
                "code path. Any bug in how prediction[\"classes\"] is stored in audit "
                "interval records will appear as a missing field.\n"
                "2. **use_bonferroni=True**: with max_depth=3 and many bins per feature, "
                "Bonferroni divides significance by n_bins, making the guard dramatically "
                "stricter. The bonferroni_comparison.png plot quantifies this effect.\n"
                "3. **Tiny calibration sets**: iris with ~30 calibration instances has "
                "p-value granularity of 1/30 ≈ 0.033. At significance=0.05 only 1-2 "
                "discrete p-value levels separate filtered from kept intervals.\n"
                "4. **High-dimensional data**: digits_01 with 64 features tests KNN "
                "guard behavior in high dimensions with a small calibration set."
            ),
        ),
    ]
    write_report(out_dir / "report.md", "Scenario D: Real Dataset Correctness", report_sections)
    print(f"Wrote: {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
