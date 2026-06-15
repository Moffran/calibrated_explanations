"""Scenario C: Rule-count and retention benchmarks on real datasets.

Scientific question
-------------------
How does conformal guard filtering affect the number of emitted factual rules
across the full dataset universe (binary classification, multiclass
classification, regression)?  Are guard retention rates stable and meaningful
enough to be practically useful across diverse real-world datasets?

Primary metrics (representative-level, factual mode)
----------------------------------------------------
- guard_retention_rate
    intervals_emitted / (intervals_emitted + guard_removed), computed over
    factual bins only (design_excluded non-factual bins are excluded from both
    numerator and denominator).  At significance=0.1 the guard should retain
    at least 90 % of in-distribution candidates by construction; lower values
    indicate regions of sparse calibration coverage or correlated features.

- mean_guarded_rules_per_instance
    Mean intervals_emitted per test instance across the test set.  Directly
    comparable to mean_standard_rules_per_instance.

- mean_standard_rules_per_instance
    Mean factual-rule count per test instance from standard Calibrated
    Explanations (max_depth=1, no guard).  Approximately equal to n_features
    modulo zero-weight filtering.

- fraction_instances_fully_filtered
    Fraction of test instances that receive zero guarded rules (all features
    filtered).  Values above 0.10 at significance=0.10 indicate that the guard
    is too aggressive for practical deployment on this dataset.

Coverage preservation is NOT a benchmark metric: it is structurally invariant
under guard filtering; see main.tex §2.3.

Execution modes
---------------
--quick         Fast smoke-test (1 seed, 2 datasets per task, ε=0.10 only).
--paper-focused Paper-facing defaults (3 seeds, full grid, ε∈{0.05,0.10,0.20}).
(default)       As paper-focused.

Run with --quick for a fast local sanity check.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure the terminal can render Unicode characters (e.g. ε) on Windows.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Repo-root path fix so the module works when executed from the guarded/ dir.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from calibrated_explanations.ce_agent_utils import ensure_ce_first_wrapper, fit_and_calibrate

from common_guarded import (  # type: ignore[import]
    GuardConfig,
    ProgressTracker,
    append_intermediate_rows,
    extract_audit_summary_rows,
    reset_intermediate_outputs,
    write_progress_snapshot,
    write_report,
)

from evaluation.ensure.common_ensure import (  # noqa: E402
    EnsureRunConfig,
    _safe_calibration_pool_size,
    _safe_test_size,
    can_run_dataset,
    compute_thresholds_from_non_test_targets,
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

DEFAULT_SIGNIFICANCE: Tuple[float, ...] = (0.05, 0.10, 0.20)
DEFAULT_N_NEIGHBORS: int = 5
DEFAULT_NORMALIZE_GUARD: bool = True
DEFAULT_MERGE_ADJACENT: bool = False
DEFAULT_NUM_SEEDS: int = 3
DEFAULT_CAL_SIZES: Tuple[int, ...] = EnsureRunConfig.calibration_sizes  # (100, 300, 500)
DEFAULT_TEST_SIZE: int = EnsureRunConfig.test_size  # 100

# Columns for the intermediate CSV
_RESULT_COLUMNS: Tuple[str, ...] = (
    "task",
    "dataset",
    "n_features",
    "n_classes",
    "n_cal",
    "seed",
    "significance",
    "n_neighbors",
    "normalize_guard",
    "regression_mode",
    "threshold_value",
    "explanation_type",
    "n_test_instances",
    "mean_standard_rules_per_instance",
    "mean_guarded_rules_per_instance",
    "guard_retention_rate",
    "fraction_instances_fully_filtered",
    "n_instances_fully_filtered",
    "task_skipped",
    "skip_reason",
    "error",
)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _load_dataset(task: str, name: str) -> Any:
    """Dispatch to the appropriate dataset loader."""
    if task == "binary":
        return load_binary_dataset(name)
    if task == "multiclass":
        return load_multiclass_dataset(name)
    if task == "regression":
        return load_regression_dataset_from_txt(name)
    raise ValueError(f"Unsupported task: {task!r}")


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
    specs.extend(("binary", name) for name in binary)
    specs.extend(("multiclass", name) for name in multiclass)
    specs.extend(("regression", name) for name in regression)
    return specs


# ---------------------------------------------------------------------------
# Per-dataset evaluation
# ---------------------------------------------------------------------------

def _count_standard_rules(explanations: Any) -> float:
    """Return mean rule count per instance from a factual or alternative explanation batch.

    get_rules() returns the ConjunctionState payload dict (16 fixed schema keys).
    len(dict) is always 16 — the actual per-instance rule list is rules["rule"].
    """
    counts: List[int] = []
    for expl in explanations:
        try:
            rules = expl.get_rules()
            if rules is None:
                counts.append(0)
            elif isinstance(rules, dict):
                counts.append(len(rules.get("rule", [])))
            else:
                counts.append(len(rules))
        except Exception:  # noqa: BLE001
            counts.append(0)
    return float(np.mean(counts)) if counts else float("nan")


def _extract_guard_metrics(
    instances: List[Dict[str, Any]],
    n_test: int,
    *,
    explanation_type: str,
) -> Tuple[float, float, float, int]:
    """Compute guarded metrics from audit instance records.

    For factual explanations the relevant bins are ``is_factual=True``; for
    alternative explanations they are ``is_factual=False``.

    Returns (mean_guarded_rules, guard_retention_rate,
             fraction_fully_filtered, n_fully_filtered).
    """
    is_target = (explanation_type == "factual")  # True → factual bins; False → alt bins

    total_emitted = 0
    total_target_emitted = 0
    total_target_guard_removed = 0
    n_test_actual = len(instances) if instances else n_test
    n_fully_filtered = 0

    for inst in instances:
        inst_emitted = inst["summary"].get("intervals_emitted", 0)
        total_emitted += inst_emitted
        if inst_emitted == 0:
            n_fully_filtered += 1
        for rec in inst.get("intervals", []):
            if rec.get("is_factual", False) == is_target:
                if rec.get("emitted", False):
                    total_target_emitted += 1
                elif rec.get("emission_reason") == "removed_guard":
                    total_target_guard_removed += 1

    mean_guarded_rules = total_emitted / max(1, n_test_actual)
    factual_denom = total_target_emitted + total_target_guard_removed
    guard_retention = (
        total_target_emitted / factual_denom if factual_denom > 0 else float("nan")
    )
    fraction_filtered = n_fully_filtered / max(1, n_test_actual)
    return mean_guarded_rules, guard_retention, fraction_filtered, n_fully_filtered


def _evaluate_dataset(
    task: str,
    name: str,
    *,
    guard_configs: List[GuardConfig],
    num_seeds: int,
    ensure_config: EnsureRunConfig,
) -> List[Dict[str, Any]]:
    """Run all configurations on one dataset and return a list of result rows.

    Each row corresponds to one (seed, calibration_size, significance) triple.
    Skip rows are inserted when the dataset cannot be run.
    """
    try:
        ds = _load_dataset(task, name)
    except Exception:  # noqa: BLE001
        return [{
            "task": task, "dataset": name,
            "n_features": None, "n_classes": None, "n_cal": None,
            "seed": None, "significance": None, "n_neighbors": None,
            "normalize_guard": None, "n_test_instances": 0,
            "mean_standard_rules_per_instance": float("nan"),
            "mean_guarded_rules_per_instance": float("nan"),
            "guard_retention_rate": float("nan"),
            "fraction_instances_fully_filtered": float("nan"),
            "n_instances_fully_filtered": -1,
            "task_skipped": True,
            "skip_reason": f"load failed: {traceback.format_exc().splitlines()[-1]}",
            "error": None,
        }]

    n_classes: Optional[int] = (
        int(len(np.unique(ds.y))) if task != "regression" else None
    )
    ok, reason = can_run_dataset(
        n_samples=int(ds.X.shape[0]),
        task=task,
        config=ensure_config,
        n_classes=n_classes,
    )
    if not ok:
        return [{
            "task": task, "dataset": name,
            "n_features": int(ds.X.shape[1]), "n_classes": n_classes,
            "n_cal": 0, "seed": None, "significance": None, "n_neighbors": None,
            "normalize_guard": None, "n_test_instances": 0,
            "mean_standard_rules_per_instance": float("nan"),
            "mean_guarded_rules_per_instance": float("nan"),
            "guard_retention_rate": float("nan"),
            "fraction_instances_fully_filtered": float("nan"),
            "n_instances_fully_filtered": -1,
            "task_skipped": True, "skip_reason": reason, "error": None,
        }]

    n_features = int(ds.X.shape[1])
    test_size = _safe_test_size(int(ds.X.shape[0]), ensure_config.test_size)

    rows: List[Dict[str, Any]] = []

    for seed in range(num_seeds):
        split_kwargs: Dict[str, Any] = {"test_size": test_size, "random_state": seed}
        if task != "regression":
            split_kwargs["stratify"] = ds.y
        try:
            x_train, x_test, y_train, y_test = train_test_split(ds.X, ds.y, **split_kwargs)
        except ValueError:
            rows.append(_skip_row(task, name, n_features, n_classes, seed, "train_test_split failed"))
            continue

        # Build calibration pool from the training portion
        min_rem = max(100, int(n_classes)) if task != "regression" and n_classes else 100
        cal_pool_size = _safe_calibration_pool_size(
            len(x_train), max(ensure_config.calibration_sizes), min_remaining=min_rem
        )
        cal_split_kw: Dict[str, Any] = {"test_size": cal_pool_size, "random_state": seed}
        if task != "regression":
            cal_split_kw["stratify"] = y_train
        try:
            x_tr, x_cal_pool, y_tr, y_cal_pool = train_test_split(x_train, y_train, **cal_split_kw)
        except ValueError:
            rows.append(_skip_row(task, name, n_features, n_classes, seed, "cal pool split failed"))
            continue

        if task != "regression":
            assert n_classes is not None
            effective_cal_sizes = [
                s for s in ensure_config.calibration_sizes
                if s <= len(x_cal_pool) and s >= n_classes
            ]
            if not effective_cal_sizes and len(x_cal_pool) >= n_classes:
                effective_cal_sizes = [len(x_cal_pool)]
        else:
            effective_cal_sizes = [
                s for s in ensure_config.calibration_sizes if s <= len(x_cal_pool)
            ]
            if not effective_cal_sizes:
                effective_cal_sizes = [len(x_cal_pool)]

        if not effective_cal_sizes:
            rows.append(_skip_row(task, name, n_features, n_classes, seed, "no valid cal size"))
            continue

        # For regression: evaluate plain conformal + probabilistic (p25/p50/p75) using
        # the same trained model but passing each threshold at explain time — exactly
        # mirroring the ensure regression evaluation (compute_thresholds_from_non_test_targets).
        if task == "regression":
            thresh_dict = compute_thresholds_from_non_test_targets(
                y_train, percentiles=(25, 50, 75)
            )
            regression_modes: List[Tuple[str, Optional[float]]] = [
                ("plain", None),
                ("p25", float(thresh_dict["p25"])),
                ("p50", float(thresh_dict["p50"])),
                ("p75", float(thresh_dict["p75"])),
            ]
        else:
            regression_modes = [("cls", None)]

        for cal_size in effective_cal_sizes:
            x_cal, y_cal = subsample_calibration(
                x_cal_pool, y_cal_pool, cal_size, random_state=seed + int(cal_size)
            )

            # One model per (seed, cal_size) triple — shared across regression modes
            # and guard configs.  For regression, calibrate once with mode="regression";
            # the threshold is only passed at explain time.
            model_cls = RandomForestRegressor if task == "regression" else RandomForestClassifier
            raw_model = model_cls(n_estimators=100, max_depth=8, random_state=seed, n_jobs=1)
            wrapper = ensure_ce_first_wrapper(raw_model)
            calibrate_mode = "regression" if task == "regression" else "classification"
            try:
                fit_and_calibrate(
                    wrapper, x_tr, y_tr, x_cal, y_cal,
                    explainer={
                        "mode": calibrate_mode,
                        "feature_names": ds.feature_names,
                        "categorical_features": ds.categorical_features,
                    },
                )
            except Exception:  # noqa: BLE001
                rows.append(_error_row(
                    task, name, n_features, n_classes, cal_size, seed,
                    traceback.format_exc().splitlines()[-1],
                    n_test=len(x_test),
                ))
                continue

            for reg_mode, threshold_val in regression_modes:
                # Standard CE rules: factual and alternative (shared model, same threshold).
                try:
                    std_factual_expl = wrapper.explain_factual(
                        x_test, threshold=threshold_val
                    )
                    mean_std_factual = _count_standard_rules(std_factual_expl)
                except Exception:  # noqa: BLE001
                    mean_std_factual = float("nan")

                try:
                    std_alt_expl = wrapper.explore_alternatives(
                        x_test, threshold=threshold_val
                    )
                    mean_std_alt = _count_standard_rules(std_alt_expl)
                except Exception:  # noqa: BLE001
                    mean_std_alt = float("nan")

                # Guarded CE — one run per (significance, n_neighbors) config,
                # both factual and alternative.
                for cfg in guard_configs:
                    safe_nn = min(cfg.n_neighbors, max(1, len(x_cal) - 1))
                    safe_cfg = GuardConfig(
                        significance=cfg.significance,
                        n_neighbors=safe_nn,
                        merge_adjacent=cfg.merge_adjacent,
                        normalize_guard=cfg.normalize_guard,
                    )
                    _shared = dict(
                        task=task, dataset=name, n_features=n_features, n_classes=n_classes,
                        n_cal=int(cal_size), seed=seed, significance=safe_cfg.significance,
                        n_neighbors=safe_cfg.n_neighbors, normalize_guard=safe_cfg.normalize_guard,
                        regression_mode=reg_mode, threshold_value=threshold_val,
                        task_skipped=False, skip_reason=None, error=None,
                    )

                    # --- factual ---
                    try:
                        guarded_factual = wrapper.explain_factual(
                            x_test,
                            guarded=True,
                            threshold=threshold_val,
                            significance=safe_cfg.significance,
                            n_neighbors=safe_cfg.n_neighbors,
                            merge_adjacent=safe_cfg.merge_adjacent,
                            normalize_guard=safe_cfg.normalize_guard,
                        )
                        instances_f = guarded_factual.get_guarded_audit().get("instances", [])
                        mg_f, gr_f, ff_f, nff_f = _extract_guard_metrics(
                            instances_f, len(x_test), explanation_type="factual"
                        )
                        rows.append({
                            **_shared,
                            "explanation_type": "factual",
                            "n_test_instances": len(instances_f) if instances_f else len(x_test),
                            "mean_standard_rules_per_instance": mean_std_factual,
                            "mean_guarded_rules_per_instance": mg_f,
                            "guard_retention_rate": gr_f,
                            "fraction_instances_fully_filtered": ff_f,
                            "n_instances_fully_filtered": nff_f,
                        })
                    except Exception:  # noqa: BLE001
                        rows.append(_error_row(
                            task, name, n_features, n_classes, cal_size, seed,
                            traceback.format_exc().splitlines()[-1],
                            n_test=len(x_test),
                            significance=cfg.significance,
                            n_neighbors=cfg.n_neighbors,
                            regression_mode=reg_mode,
                            threshold_val=threshold_val,
                            explanation_type="factual",
                        ))

                    # --- alternative ---
                    try:
                        guarded_alt = wrapper.explore_alternatives(
                            x_test,
                            guarded=True,
                            threshold=threshold_val,
                            significance=safe_cfg.significance,
                            n_neighbors=safe_cfg.n_neighbors,
                            merge_adjacent=safe_cfg.merge_adjacent,
                            normalize_guard=safe_cfg.normalize_guard,
                        )
                        instances_a = guarded_alt.get_guarded_audit().get("instances", [])
                        mg_a, gr_a, ff_a, nff_a = _extract_guard_metrics(
                            instances_a, len(x_test), explanation_type="alternative"
                        )
                        rows.append({
                            **_shared,
                            "explanation_type": "alternative",
                            "n_test_instances": len(instances_a) if instances_a else len(x_test),
                            "mean_standard_rules_per_instance": mean_std_alt,
                            "mean_guarded_rules_per_instance": mg_a,
                            "guard_retention_rate": gr_a,
                            "fraction_instances_fully_filtered": ff_a,
                            "n_instances_fully_filtered": nff_a,
                        })
                    except Exception:  # noqa: BLE001
                        rows.append(_error_row(
                            task, name, n_features, n_classes, cal_size, seed,
                            traceback.format_exc().splitlines()[-1],
                            n_test=len(x_test),
                            significance=cfg.significance,
                            n_neighbors=cfg.n_neighbors,
                            regression_mode=reg_mode,
                            threshold_val=threshold_val,
                            explanation_type="alternative",
                        ))

    return rows


def _skip_row(
    task: str,
    name: str,
    n_features: Optional[int],
    n_classes: Optional[int],
    seed: int,
    reason: str,
) -> Dict[str, Any]:
    return {
        "task": task, "dataset": name, "n_features": n_features,
        "n_classes": n_classes, "n_cal": 0, "seed": seed,
        "significance": None, "n_neighbors": None, "normalize_guard": None,
        "regression_mode": None, "threshold_value": None, "explanation_type": None,
        "n_test_instances": 0,
        "mean_standard_rules_per_instance": float("nan"),
        "mean_guarded_rules_per_instance": float("nan"),
        "guard_retention_rate": float("nan"),
        "fraction_instances_fully_filtered": float("nan"),
        "n_instances_fully_filtered": -1,
        "task_skipped": True, "skip_reason": reason, "error": None,
    }


def _error_row(
    task: str,
    name: str,
    n_features: Optional[int],
    n_classes: Optional[int],
    cal_size: int,
    seed: int,
    error_msg: str,
    *,
    n_test: int = 0,
    significance: Optional[float] = None,
    n_neighbors: Optional[int] = None,
    regression_mode: Optional[str] = None,
    threshold_val: Optional[float] = None,
    explanation_type: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "task": task, "dataset": name, "n_features": n_features,
        "n_classes": n_classes, "n_cal": int(cal_size), "seed": seed,
        "significance": significance, "n_neighbors": n_neighbors,
        "normalize_guard": DEFAULT_NORMALIZE_GUARD,
        "regression_mode": regression_mode, "threshold_value": threshold_val,
        "explanation_type": explanation_type,
        "n_test_instances": n_test,
        "mean_standard_rules_per_instance": float("nan"),
        "mean_guarded_rules_per_instance": float("nan"),
        "guard_retention_rate": float("nan"),
        "fraction_instances_fully_filtered": float("nan"),
        "n_instances_fully_filtered": -1,
        "task_skipped": False, "skip_reason": None, "error": error_msg,
    }


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _aggregate_results(df: pd.DataFrame, significance: float) -> pd.DataFrame:
    """Return per-dataset aggregates at a fixed significance level.

    Averages over seeds and calibration sizes for non-skipped rows.
    Each (task, dataset, regression_mode, explanation_type) combination
    yields one row.
    """
    sub = df[
        (df["task_skipped"] == False)  # noqa: E712
        & (df["significance"] == significance)
        & df["error"].isna()
        & df["explanation_type"].notna()
    ].copy()
    if sub.empty:
        return pd.DataFrame()

    agg = (
        sub.groupby(["task", "dataset", "n_features", "regression_mode", "explanation_type"])[
            [
                "mean_standard_rules_per_instance",
                "mean_guarded_rules_per_instance",
                "guard_retention_rate",
                "fraction_instances_fully_filtered",
            ]
        ]
        .mean()
        .reset_index()
    )
    agg = agg.sort_values(["task", "dataset", "regression_mode", "explanation_type"])
    return agg


def _summary_by_task(agg: pd.DataFrame) -> pd.DataFrame:
    """Collapse the per-dataset aggregate to a per-task / per-regression-mode /
    per-explanation-type summary."""
    if agg.empty:
        return pd.DataFrame()
    return (
        agg.groupby(["task", "regression_mode", "explanation_type"])[
            [
                "mean_standard_rules_per_instance",
                "mean_guarded_rules_per_instance",
                "guard_retention_rate",
                "fraction_instances_fully_filtered",
            ]
        ]
        .mean()
        .reset_index()
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_retention_by_task(df: pd.DataFrame, sig: float, out_dir: Path) -> None:
    """Bar chart: mean guard retention rate by task at a fixed significance."""
    sub = df[
        (df["task_skipped"] == False)  # noqa: E712
        & (df["significance"] == sig)
        & df["error"].isna()
    ]
    if sub.empty:
        return
    means = sub.groupby("task")["guard_retention_rate"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(means["task"], means["guard_retention_rate"], color="steelblue")
    ax.axhline(1.0 - sig, linestyle="--", color="orange",
               label=f"expected floor (1−ε={1.0 - sig:.2f})")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean guard retention rate")
    ax.set_title(f"Guard retention rate by task (ε={sig})")
    ax.legend(fontsize=8)
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{bar.get_height():.2f}",
            ha="center", va="bottom", fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(out_dir / f"retention_by_task_eps{sig:.2f}.png", dpi=160)
    plt.close(fig)


def _plot_rules_by_task(df: pd.DataFrame, sig: float, out_dir: Path) -> None:
    """Grouped bar chart: mean standard vs guarded rules per instance by task."""
    sub = df[
        (df["task_skipped"] == False)  # noqa: E712
        & (df["significance"] == sig)
        & df["error"].isna()
    ]
    if sub.empty:
        return
    means = sub.groupby("task")[
        ["mean_standard_rules_per_instance", "mean_guarded_rules_per_instance"]
    ].mean().reset_index()

    tasks = means["task"].tolist()
    x = np.arange(len(tasks))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, means["mean_standard_rules_per_instance"],
           width, label="Standard CE", color="orange", alpha=0.8)
    ax.bar(x + width / 2, means["mean_guarded_rules_per_instance"],
           width, label=f"Guarded CE (ε={sig})", color="steelblue", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.set_ylabel("Mean rules per instance")
    ax.set_title(f"Standard vs guarded rule count by task (ε={sig})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"rules_per_instance_by_task_eps{sig:.2f}.png", dpi=160)
    plt.close(fig)


def _plot_retention_vs_significance(df: pd.DataFrame, out_dir: Path) -> None:
    """Line chart: mean guard retention rate vs significance, one line per task."""
    sub = df[
        (df["task_skipped"] == False)  # noqa: E712
        & df["error"].isna()
        & df["significance"].notna()
    ]
    if sub.empty:
        return
    means = sub.groupby(["task", "significance"])["guard_retention_rate"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(6, 4))
    for task in sorted(means["task"].unique()):
        grp = means[means["task"] == task].sort_values("significance")
        ax.plot(grp["significance"], grp["guard_retention_rate"], marker="o", label=task)
    ax.set_xlabel("Guard significance ε")
    ax.set_ylabel("Mean guard retention rate")
    ax.set_title("Guard retention rate vs significance by task")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out_dir / "retention_vs_significance.png", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse Scenario C command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "artifacts" / "guarded" / "scenario_c",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=DEFAULT_NUM_SEEDS,
        help="Number of random splits per dataset (default: 3).",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=DEFAULT_TEST_SIZE,
        help="Test set size per split (default: 100).",
    )
    parser.add_argument(
        "--calibration-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_CAL_SIZES),
        help="Calibration sizes to sweep (default: 100 300 500).",
    )
    parser.add_argument(
        "--significance",
        type=float,
        nargs="+",
        default=list(DEFAULT_SIGNIFICANCE),
        help="Guard significance levels to evaluate (default: 0.05 0.10 0.20).",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=DEFAULT_N_NEIGHBORS,
        help="KNN neighbours for the guard (default: 5).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke-test: 1 seed, 2 datasets per task, ε=0.10 only.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Suppress the advisory "predict not in [low, high]" UserWarning that CE emits
    # when a perturbed-feature representative sits fractionally outside its conformal
    # interval.  This is expected for probabilistic-regression threshold modes (p25/
    # p50/p75) and for OOD perturbations that the guard already filters — it carries
    # no information beyond what the guard metrics themselves capture.
    warnings.filterwarnings(
        "ignore",
        message="Prediction invariant violated",
        category=UserWarning,
        module=r"calibrated_explanations",
    )

    args = parse_args()

    if args.quick:
        args.num_seeds = 1
        args.significance = [0.10]
        args.calibration_sizes = [100]

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ensure_config = EnsureRunConfig(
        test_size=args.test_size,
        calibration_sizes=tuple(int(s) for s in args.calibration_sizes),
    )
    dataset_specs = load_dataset_specs(quick=args.quick)
    guard_configs = [
        GuardConfig(
            significance=float(sig),
            n_neighbors=args.n_neighbors,
            merge_adjacent=DEFAULT_MERGE_ADJACENT,
            normalize_guard=DEFAULT_NORMALIZE_GUARD,
        )
        for sig in args.significance
    ]

    intermediate_csv = out_dir / "scenario_c_raw.csv"
    progress_json = out_dir / "progress.json"
    reset_intermediate_outputs([intermediate_csv, progress_json])

    tracker = ProgressTracker("scenario_c", total=len(dataset_specs))
    tracker.start(f"{len(dataset_specs)} datasets × {args.num_seeds} seeds × "
                  f"{len(args.calibration_sizes)} cal sizes × "
                  f"{len(args.significance)} sig levels")

    for task, name in dataset_specs:
        tracker.advance(detail=f"{task}/{name}", force=True)
        write_progress_snapshot(progress_json, tracker, detail=f"{task}/{name}")

        rows = _evaluate_dataset(
            task, name,
            guard_configs=guard_configs,
            num_seeds=args.num_seeds,
            ensure_config=ensure_config,
        )
        append_intermediate_rows(rows, intermediate_csv, columns=list(_RESULT_COLUMNS))

    tracker.finish("all datasets processed")

    # -------------------------------------------------------------------
    # Load full result frame for reporting
    # -------------------------------------------------------------------
    if not intermediate_csv.exists():
        print("[WARN] No results written — all datasets were skipped or failed.")
        return

    df = pd.read_csv(intermediate_csv)

    # Pandas treats "N/A" as NaN by default, so classification rows written with
    # regression_mode="N/A" (legacy) or rows that stored None arrive as NaN here.
    # Restore the sentinel for non-regression, non-skipped rows so groupby includes them.
    cls_mask = (
        df["task"].isin(["binary", "multiclass"])
        & (df["task_skipped"] == False)  # noqa: E712
        & df["regression_mode"].isna()
    )
    df.loc[cls_mask, "regression_mode"] = "cls"

    # Summary for the primary significance (0.10)
    primary_sig = 0.10
    agg = _aggregate_results(df, primary_sig)
    task_summary = _summary_by_task(agg)

    # -------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------
    for sig in args.significance:
        _plot_retention_by_task(df, sig, out_dir)
        _plot_rules_by_task(df, sig, out_dir)
    _plot_retention_vs_significance(df, out_dir)

    # -------------------------------------------------------------------
    # Report
    # -------------------------------------------------------------------
    def _fmt(v: Any) -> str:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v)

    # Task-level summary table
    if not task_summary.empty:
        task_lines = [
            "| Task | Reg mode | Expl type | Std rules | Guarded rules | Retention | Fully filtered |",
            "|---|---|---|---|---|---|---|",
        ]
        for _, row in task_summary.iterrows():
            task_lines.append(
                f"| {row['task']} "
                f"| {row.get('regression_mode', 'N/A')} "
                f"| {row.get('explanation_type', '?')} "
                f"| {_fmt(row['mean_standard_rules_per_instance'])} "
                f"| {_fmt(row['mean_guarded_rules_per_instance'])} "
                f"| {_fmt(row['guard_retention_rate'])} "
                f"| {_fmt(row['fraction_instances_fully_filtered'])} |"
            )
        task_body = "\n".join(task_lines)
    else:
        task_body = "No runnable datasets at primary significance."

    # Per-dataset table
    if not agg.empty:
        ds_lines = [
            "| Task | Dataset | Reg mode | Expl type | d | Std rules | Guarded rules | Retention | Fully filtered |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for _, row in agg.iterrows():
            ds_lines.append(
                f"| {row['task']} "
                f"| {row['dataset']} "
                f"| {row.get('regression_mode', 'N/A')} "
                f"| {row.get('explanation_type', '?')} "
                f"| {int(row['n_features']) if pd.notna(row['n_features']) else '?'} "
                f"| {_fmt(row['mean_standard_rules_per_instance'])} "
                f"| {_fmt(row['mean_guarded_rules_per_instance'])} "
                f"| {_fmt(row['guard_retention_rate'])} "
                f"| {_fmt(row['fraction_instances_fully_filtered'])} |"
            )
        ds_body = "\n".join(ds_lines)
    else:
        ds_body = "No runnable datasets at primary significance."

    # Skip/error summary
    n_total = len(dataset_specs)
    n_skipped = int(df[df["task_skipped"] == True]["dataset"].nunique())  # noqa: E712
    n_errors = int(df[df["error"].notna() & (df["task_skipped"] == False)]["dataset"].nunique())  # noqa: E712
    status_body = (
        f"- Total datasets evaluated: {n_total}\n"
        f"- Skipped (too small): {n_skipped}\n"
        f"- Errors during execution: {n_errors}\n"
        f"\n**Seeds:** {args.num_seeds}  "
        f"**Cal sizes:** {list(args.calibration_sizes)}  "
        f"**k:** {args.n_neighbors}  "
        f"**normalize_guard:** {DEFAULT_NORMALIZE_GUARD}  "
        f"**merge_adjacent:** {DEFAULT_MERGE_ADJACENT}"
    )

    write_report(
        out_dir / "report.md",
        title="Scenario C — Real-Dataset Guard Retention Benchmark",
        sections=[
            ("Scientific Question", (
                "How does conformal guard filtering affect the number of emitted factual and "
                "alternative rules across the full dataset universe?  Are guard retention rates "
                "stable across diverse real-world datasets at ε=0.10?\n\n"
                "**Coverage preservation is not a metric here**: it is structurally invariant "
                "under guard filtering (§2.3 of the paper)."
            )),
            (f"Task-level summary (ε={primary_sig})", task_body),
            (f"Per-dataset results (ε={primary_sig})", ds_body),
            ("Execution summary", status_body),
            ("Metric definitions", (
                "| Metric | Definition |\n"
                "|---|---|\n"
                "| `mean_standard_rules_per_instance` | Mean rule count from `explain_factual` / `explore_alternatives` |\n"
                "| `mean_guarded_rules_per_instance` | Mean `intervals_emitted` per test instance |\n"
                "| `guard_retention_rate` | `intervals_emitted / (intervals_emitted + guard_removed)` over factual bins only |\n"
                "| `fraction_instances_fully_filtered` | Fraction of test instances with 0 guarded rules |\n"
            )),
        ],
    )

    # -------------------------------------------------------------------
    # JSON summary
    # -------------------------------------------------------------------
    summary: Dict[str, Any] = {
        "scenario": "C",
        "description": "Real-dataset guard retention benchmark",
        "primary_significance": primary_sig,
        "n_datasets_total": n_total,
        "n_datasets_skipped": n_skipped,
        "n_datasets_errored": n_errors,
        "config": {
            "num_seeds": args.num_seeds,
            "calibration_sizes": list(args.calibration_sizes),
            "significance_levels": list(args.significance),
            "n_neighbors": args.n_neighbors,
            "normalize_guard": DEFAULT_NORMALIZE_GUARD,
            "merge_adjacent": DEFAULT_MERGE_ADJACENT,
        },
        "task_summary": task_summary.to_dict(orient="records") if not task_summary.empty else [],
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print(f"\nScenario C complete.")
    print(f"  Raw results : {intermediate_csv}")
    print(f"  Report      : {out_dir / 'report.md'}")
    print(f"  Summary JSON: {summary_path}")

    if not task_summary.empty:
        print(f"\nTask-level summary at ε={primary_sig}:")
        print(task_summary.to_string(index=False))


if __name__ == "__main__":
    main()
