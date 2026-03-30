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
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from calibrated_explanations.ce_agent_utils import ensure_ce_first_wrapper, fit_and_calibrate

from common_guarded import (
    GuardConfig,
    check_audit_field_completeness,
    extract_audit_summary_rows,
    write_report,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.ensure.common_ensure import (  # noqa: E402
    EnsureRunConfig,
    _safe_calibration_pool_size,
    _safe_test_size,
    can_run_dataset,
    subsample_calibration,
)
from evaluation.ensure.datasets_ensure import (  # noqa: E402
    BINARY_CLASSIFICATION_DATASETS,
    MULTICLASS_DATASETS,
    list_regression_txt_datasets,
    load_binary_dataset,
    load_multiclass_dataset,
    load_regression_dataset_from_txt,
)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_SIGNIFICANCE = (0.05, 0.10, 0.20)
DEFAULT_N_NEIGHBORS = (3, 5)
DEFAULT_TEST_SIZE = EnsureRunConfig.test_size
DEFAULT_CALIBRATION_SIZES = EnsureRunConfig.calibration_sizes


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

def load_dataset_specs(quick: bool) -> List[Tuple[str, str]]:
    """Return ``(task, dataset_name)`` pairs from the ensured dataset registry."""
    binary = list(BINARY_CLASSIFICATION_DATASETS)
    multiclass = list(MULTICLASS_DATASETS)
    regression = list_regression_txt_datasets()

    if quick:
        binary = binary[:2]
        multiclass = multiclass[:2]
        regression = regression[:2]

    specs: List[Tuple[str, str]] = []
    specs.extend([("binary", name) for name in binary])
    specs.extend([("multiclass", name) for name in multiclass])
    specs.extend([("regression", name) for name in regression])
    return specs


def _load_dataset(task: str, name: str) -> Any:
    if task == "binary":
        return load_binary_dataset(name)
    if task == "multiclass":
        return load_multiclass_dataset(name)
    if task == "regression":
        return load_regression_dataset_from_txt(name)
    raise ValueError(f"Unsupported task: {task}")


# ---------------------------------------------------------------------------
# Per-dataset evaluation
# ---------------------------------------------------------------------------

def _evaluate_dataset(
    task: str,
    name: str,
    configs: List[GuardConfig],
    seed: int,
    ensure_config: EnsureRunConfig,
) -> Tuple[List[Dict], List[Dict]]:
    """Run all configs on one dataset / seed pair.

    Returns (summary_rows, completeness_rows).
    """
    ds = _load_dataset(task, name)
    n_classes = int(len(np.unique(ds.y))) if task != "regression" else None
    ok, reason = can_run_dataset(
        n_samples=int(ds.X.shape[0]),
        task=task,
        config=ensure_config,
        n_classes=n_classes,
    )
    if not ok:
        return (
            [{
                "task": task,
                "dataset": name,
                "n_features": int(ds.X.shape[1]),
                "n_classes": n_classes,
                "n_cal": 0,
                "seed": seed,
                "significance": np.nan,
                "n_neighbors": np.nan,
                "audit_field_completeness": False,
                "fraction_instances_fully_filtered": float("nan"),
                "n_instances_fully_filtered": -1,
                "mean_intervals_removed_per_instance": float("nan"),
                "n_test_instances": 0,
                "task_skipped": True,
                "skip_reason": reason,
                "error": None,
            }],
            [],
        )

    test_size = _safe_test_size(int(ds.X.shape[0]), ensure_config.test_size)
    split_kwargs: Dict[str, Any] = {
        "test_size": test_size,
        "random_state": seed,
    }
    if task != "regression":
        split_kwargs["stratify"] = ds.y
    try:
        x_train, x_test, y_train, y_test = train_test_split(
            ds.X,
            ds.y,
            **split_kwargs,
        )
    except ValueError:
        return (
            [{
                "task": task,
                "dataset": name,
                "n_features": int(ds.X.shape[1]),
                "n_classes": n_classes,
                "n_cal": 0,
                "seed": seed,
                "significance": np.nan,
                "n_neighbors": np.nan,
                "audit_field_completeness": False,
                "fraction_instances_fully_filtered": float("nan"),
                "n_instances_fully_filtered": -1,
                "mean_intervals_removed_per_instance": float("nan"),
                "n_test_instances": 0,
                "task_skipped": True,
                "skip_reason": "train/test split failed",
                "error": traceback.format_exc().splitlines()[-1],
            }],
            [],
        )

    min_remaining = max(100, int(n_classes)) if task != "regression" and n_classes is not None else 100
    cal_pool_size = _safe_calibration_pool_size(
        len(x_train),
        max(DEFAULT_CALIBRATION_SIZES),
        min_remaining=min_remaining,
    )
    cal_split_kwargs: Dict[str, Any] = {
        "test_size": cal_pool_size,
        "random_state": seed,
    }
    if task != "regression":
        cal_split_kwargs["stratify"] = y_train
    x_tr, x_cal_pool, y_tr, y_cal_pool = train_test_split(
        x_train,
        y_train,
        **cal_split_kwargs,
    )

    if task != "regression":
        assert n_classes is not None
        effective_cal_sizes = [
            size
            for size in DEFAULT_CALIBRATION_SIZES
            if size <= len(x_cal_pool) and int(size) >= n_classes
        ]
        if not effective_cal_sizes and len(x_cal_pool) >= n_classes:
            effective_cal_sizes = [len(x_cal_pool)]
    else:
        effective_cal_sizes = [
            size for size in DEFAULT_CALIBRATION_SIZES if size <= len(x_cal_pool)
        ]
        if not effective_cal_sizes:
            effective_cal_sizes = [len(x_cal_pool)]

    if not effective_cal_sizes:
        return (
            [{
                "task": task,
                "dataset": name,
                "n_features": int(ds.X.shape[1]),
                "n_classes": n_classes,
                "n_cal": 0,
                "seed": seed,
                "significance": np.nan,
                "n_neighbors": np.nan,
                "audit_field_completeness": False,
                "fraction_instances_fully_filtered": float("nan"),
                "n_instances_fully_filtered": -1,
                "mean_intervals_removed_per_instance": float("nan"),
                "n_test_instances": len(x_test),
                "task_skipped": True,
                "skip_reason": "calibration pool too small",
                "error": None,
            }],
            [],
        )

    n_features = ds.X.shape[1]
    summary_rows: List[Dict] = []
    completeness_rows: List[Dict] = []

    for cal_size in effective_cal_sizes:
        x_cal, y_cal = subsample_calibration(
            x_cal_pool,
            y_cal_pool,
            cal_size,
            random_state=seed + int(cal_size),
        )
        for cfg in configs:
            safe_nn = min(cfg.n_neighbors, max(1, len(x_cal) - 1))
            safe_cfg = GuardConfig(
                significance=cfg.significance,
                n_neighbors=safe_nn,
                merge_adjacent=cfg.merge_adjacent,
                normalize_guard=cfg.normalize_guard,
            )
            model = (
                RandomForestClassifier(n_estimators=100, max_depth=8, random_state=seed, n_jobs=1)
                if task != "regression"
                else RandomForestRegressor(n_estimators=100, max_depth=8, random_state=seed, n_jobs=1)
            )
            wrapper = ensure_ce_first_wrapper(model)
            calibrate_mode = "classification" if task != "regression" else "regression"
            try:
                fit_and_calibrate(
                    wrapper,
                    x_tr,
                    y_tr,
                    x_cal,
                    y_cal,
                    explainer={
                        "mode": calibrate_mode,
                        "feature_names": ds.feature_names,
                        "categorical_features": ds.categorical_features,
                    },
                )
                guarded_expl = wrapper.explain_guarded_factual(
                    x_test,
                    significance=safe_cfg.significance,
                    n_neighbors=safe_cfg.n_neighbors,
                    merge_adjacent=safe_cfg.merge_adjacent,
                    normalize_guard=safe_cfg.normalize_guard,
                )

                all_complete, missing_list = check_audit_field_completeness(guarded_expl)
                for missing_entry in missing_list:
                    completeness_rows.append({
                        "task": task,
                        "dataset": name,
                        "seed": seed,
                        "n_cal": len(x_cal),
                        "significance": safe_cfg.significance,
                        "n_neighbors": safe_cfg.n_neighbors,
                        **missing_entry,
                    })

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
                    "task": task,
                    "dataset": name,
                    "n_features": n_features,
                    "n_classes": n_classes,
                    "n_cal": len(x_cal),
                    "seed": seed,
                    "significance": safe_cfg.significance,
                    "n_neighbors": safe_cfg.n_neighbors,
                    "audit_field_completeness": all_complete,
                    "fraction_instances_fully_filtered": fraction_fully_filtered,
                    "n_instances_fully_filtered": n_fully_filtered,
                    "mean_intervals_removed_per_instance": mean_removed,
                    "n_test_instances": n_test_actual,
                    "task_skipped": False,
                    "skip_reason": None,
                    "error": None,
                })

            except Exception:  # noqa: BLE001
                summary_rows.append({
                    "task": task,
                    "dataset": name,
                    "n_features": n_features,
                    "n_classes": n_classes,
                    "n_cal": len(x_cal),
                    "seed": seed,
                    "significance": cfg.significance,
                    "n_neighbors": cfg.n_neighbors,
                    "audit_field_completeness": False,
                    "fraction_instances_fully_filtered": float("nan"),
                    "n_instances_fully_filtered": -1,
                    "mean_intervals_removed_per_instance": float("nan"),
                    "n_test_instances": len(x_test),
                    "task_skipped": False,
                    "skip_reason": None,
                    "error": traceback.format_exc().splitlines()[-1],
                })

    return summary_rows, completeness_rows


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_fraction_filtered(df: pd.DataFrame, out_dir: Path) -> None:
    """Bar chart: fraction_instances_fully_filtered by task at significance=0.10."""
    sub = df[df["significance"] == 0.10].copy()
    if sub.empty:
        return
    sub = sub[sub["task_skipped"] == False]  # noqa: E712
    means = sub.groupby("task")["fraction_instances_fully_filtered"].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    means.plot(kind="bar", ax=ax, color="steelblue")
    ax.axhline(0.10, linestyle="--", color="orange", label="significance=0.10")
    ax.set_ylabel("fraction instances with 0 emitted rules")
    ax.set_title("Fraction Fully Filtered by Task (significance=0.10)")
    ax.set_xlabel("")
    ax.legend()
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "fraction_filtered_by_dataset.png", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse Scenario D command-line arguments.

    Parameters exposed to the caller
    --------------------------------
    --output-dir : pathlib.Path
        Destination for real-dataset metrics, completeness details, plots, and
        report.
    --num-seeds : int, default=5
        Number of repeated split draws per dataset. Useful range: 1-3 for the
        full ensured-style dataset universe; higher values increase runtime
        materially because Scenario D now covers binary, multiclass, and
        regression datasets.
    --quick : bool
        Restricts the dataset list and configuration grid to a smoke-test
        subset. Useful for CI and local sanity checks.
    """
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
    else:
        sig_grid = DEFAULT_SIGNIFICANCE
        nn_grid = DEFAULT_N_NEIGHBORS

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ensure_config = EnsureRunConfig(
        test_size=DEFAULT_TEST_SIZE,
        calibration_sizes=DEFAULT_CALIBRATION_SIZES,
    )
    dataset_specs = load_dataset_specs(quick=args.quick)
    configs = [
        GuardConfig(significance=s, n_neighbors=nn)
        for s in sig_grid
        for nn in nn_grid
    ]

    all_summary: List[Dict] = []
    all_completeness: List[Dict] = []

    for seed in range(args.num_seeds):
        for task, name in dataset_specs:
            summary_rows, completeness_rows = _evaluate_dataset(
                task, name, configs, seed, ensure_config
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
    print(f"Wrote plots to: {out_dir}")

    task_counts = pd.Series([task for task, _ in dataset_specs]).value_counts().to_dict()
    dataset_desc = (
        f"{task_counts.get('binary', 0)} binary, "
        f"{task_counts.get('multiclass', 0)} multiclass, "
        f"{task_counts.get('regression', 0)} regression datasets"
    )

    # Summary stats for report
    n_missing_fields = len(completeness_df)
    error_rows = summary_df[summary_df["error"].notna()] if "error" in summary_df.columns else pd.DataFrame()
    skipped_rows = summary_df[summary_df["task_skipped"] == True] if "task_skipped" in summary_df.columns else pd.DataFrame()  # noqa: E712
    skipped_unique = (
        skipped_rows[["task", "dataset", "skip_reason"]].drop_duplicates()
        if not skipped_rows.empty else pd.DataFrame()
    )
    fully_filtered_table = (
        summary_df[
            (summary_df["significance"] == 0.10)
            & (summary_df["task_skipped"] == False)  # noqa: E712
        ]
        .groupby(["task", "dataset", "n_cal"])[["fraction_instances_fully_filtered", "audit_field_completeness"]]
        .mean()
        if not summary_df.empty else pd.DataFrame()
    )

    report_sections = [
        (
            "Setup",
            (
                f"- Seeds: {args.num_seeds}\n"
                f"- Dataset universe: {dataset_desc}\n"
                f"- Ensure-style split policy: test_size={ensure_config.test_size}, calibration_sizes={list(ensure_config.calibration_sizes)}\n"
                f"- Models: RandomForestClassifier (binary/multiclass), RandomForestRegressor (regression)\n"
                f"- Guard grid: significance={list(sig_grid)}, n_neighbors={list(nn_grid)}"
            ),
        ),
        (
            "Purpose",
            (
                "Scenario D asks: does the guard's API remain correct across the full "
                "variety of real-world task types — binary classification, "
                "multiclass classification, regression, high-dimensional inputs, "
                "and small calibration sets?\n\n"
                f"This run draws from the same dataset registry used by the ensured "
                f"evaluation suite: {dataset_desc}."
            ),
        ),
        (
            "Metric contract",
            (
                "This scenario is about API correctness and usability, not about proving "
                "that guarded explanations improve scientific quality on real datasets. "
                "The two key questions are whether the audit payload is structurally "
                "complete and whether the guard remains usable rather than filtering away "
                "nearly every explanation.\n\n"
                "Accordingly, missing audit fields are correctness failures, while high "
                "fully-filtered rates are practicality warnings. Neither metric should be "
                "overstated as a standalone scientific result."
            ),
        ),
        (
            "Audit field completeness",
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
            "Dataset coverage",
            (
                f"Unique datasets skipped by the ensured-style safety checks: **{len(skipped_unique)}**\n\n"
                + (
                    skipped_unique.to_markdown(index=False)
                    if not skipped_unique.empty else
                    "No datasets were skipped."
                )
            ),
        ),
        (
            "Fraction of instances fully filtered",
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
            "Interpretation",
            (
                "Binary and multiclass datasets stress the payload shape, class handling, "
                "and empty-rule behavior under guarded filtering. Regression datasets stress "
                "the separate guarded regression path and confirm the audit contract holds "
                "outside classification.\n\n"
                "Scenario D supports an engineering claim: the guarded API behaves "
                "correctly across realistic dataset shapes. It does not by itself justify "
                "a broad real-world effectiveness claim."
            ),
        ),
    ]
    write_report(out_dir / "report.md", "Scenario D: Real Dataset Correctness", report_sections)
    print(f"Wrote: {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
