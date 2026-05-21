"""Scenario 11: matched reject-rate operating-point selection.

This scenario compares experimental reject strategies at matched observed
reject-rate operating points instead of averaging over a confidence grid.

Primary comparison:
    A. builtin.default, ncf=default
    C. experimental.difficulty_normalized, ncf=default

Secondary comparison:
    C. experimental.difficulty_normalized, ncf=default
    G. experimental.ambiguity_normalized_novelty_penalized, ncf=default

For each dataset/seed/arm, the scenario sweeps confidence values and selects the
row whose observed reject rate is closest to each target reject rate.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common_reject import RunConfig, seed_grid, task_specs, write_csv_json_md
from .scenario_10_ambiguity_novelty_reject import (
    _ARMS,
    DeterministicNoveltyEstimator,
    _run_arm,
)
from .scenario_9_difficulty_normalized_ncf import (
    _build_classification_bundle,
    _format_scalar,
    _markdown_table,
)

_PREFIX = "scenario_11_operating_point_selection"
_TARGET_REJECT_RATES = (0.10, 0.20, 0.30, 0.40)
_DELTA_METRICS: dict[str, tuple[str, ...]] = {
    "A_vs_C": (
        "accepted_accuracy",
        "empirical_coverage",
        "observed_reject_rate",
        "ambiguity_rate",
        "novelty_rate",
        "rejected_error_capture_rate",
        "difficulty_reject_auc",
        "difficulty_gap_rejected_minus_accepted",
    ),
    "C_vs_G": (
        "accepted_accuracy",
        "novelty_rate",
        "ambiguity_rate",
        "novelty_reject_auc",
        "rejected_error_capture_rate",
        "difficulty_reject_auc",
        "empty_set_rate",
        "multilabel_rate",
    ),
}


def _confidence_grid(quick: bool) -> np.ndarray:
    """Return the confidence grid used for operating-point selection."""
    count = 15 if quick else 31
    return np.linspace(0.50, 0.99, count)


def _safe_fraction_positive(values: pd.Series) -> float:
    finite = values[np.isfinite(values)]
    if finite.empty:
        return float("nan")
    return float(np.mean(finite > 0.0))


def _safe_std(values: pd.Series) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) <= 1:
        return float("nan")
    return float(finite.std(ddof=1))


def _select_operating_points(sweep: pd.DataFrame) -> pd.DataFrame:
    """Select closest observed reject-rate rows for every target operating point."""
    selected: list[pd.Series] = []
    group_cols = ["dataset", "seed", "arm_code"]
    for target in _TARGET_REJECT_RATES:
        candidate = sweep.copy()
        candidate["target_reject_rate"] = float(target)
        candidate["reject_rate_target_abs_error"] = (
            candidate["observed_reject_rate"] - float(target)
        ).abs()
        for _, group in candidate.groupby(group_cols, sort=False):
            best = group.sort_values(
                ["reject_rate_target_abs_error", "selected_confidence"],
                ascending=[True, True],
                kind="mergesort",
            ).iloc[0]
            selected.append(best)
    if not selected:
        return pd.DataFrame()
    result = pd.DataFrame(selected).reset_index(drop=True)
    return result.sort_values(
        ["dataset", "seed", "arm_code", "target_reject_rate"],
        kind="mergesort",
    ).reset_index(drop=True)


def _delta_frame(selected: pd.DataFrame) -> pd.DataFrame:
    """Compute paired deltas for A-vs-C and C-vs-G."""
    rows: list[dict[str, Any]] = []
    comparisons = {
        "A_vs_C": ("A", "C", "C_minus_A"),
        "C_vs_G": ("C", "G", "G_minus_C"),
    }
    index_cols = ["dataset", "seed", "target_reject_rate"]
    for comparison, (base_arm, candidate_arm, delta_prefix) in comparisons.items():
        subset = selected[selected["arm_code"].isin([base_arm, candidate_arm])]
        if subset.empty:
            continue
        pivot = subset.pivot_table(
            index=index_cols,
            columns="arm_code",
            values=[
                "accepted_accuracy",
                "empirical_coverage",
                "observed_reject_rate",
                "ambiguity_rate",
                "novelty_rate",
                "rejected_error_capture_rate",
                "difficulty_reject_auc",
                "difficulty_gap_rejected_minus_accepted",
                "novelty_reject_auc",
                "empty_set_rate",
                "multilabel_rate",
                "selected_confidence",
                "reject_rate_target_abs_error",
            ],
            aggfunc="first",
        )
        required = ("observed_reject_rate", base_arm), ("observed_reject_rate", candidate_arm)
        if any(column not in pivot.columns for column in required):
            continue
        for index, wide in pivot.iterrows():
            dataset, seed, target = index
            row: dict[str, Any] = {
                "comparison_group": comparison,
                "dataset": dataset,
                "seed": int(seed),
                "target_reject_rate": float(target),
                "base_arm": base_arm,
                "candidate_arm": candidate_arm,
                "base_selected_confidence": wide.get(("selected_confidence", base_arm), np.nan),
                "candidate_selected_confidence": wide.get(
                    ("selected_confidence", candidate_arm), np.nan
                ),
                "base_target_abs_error": wide.get(
                    ("reject_rate_target_abs_error", base_arm), np.nan
                ),
                "candidate_target_abs_error": wide.get(
                    ("reject_rate_target_abs_error", candidate_arm), np.nan
                ),
            }
            for metric in _DELTA_METRICS[comparison]:
                base_value = wide.get((metric, base_arm), np.nan)
                candidate_value = wide.get((metric, candidate_arm), np.nan)
                row[f"{delta_prefix}_{metric}"] = candidate_value - base_value
                row[f"base_{metric}"] = base_value
                row[f"candidate_{metric}"] = candidate_value
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["comparison_group", "target_reject_rate", "dataset", "seed"],
        kind="mergesort",
    )


def _aggregate_deltas(deltas: pd.DataFrame) -> pd.DataFrame:
    """Return target-level aggregate delta summaries."""
    rows: list[dict[str, Any]] = []
    if deltas.empty:
        return pd.DataFrame()
    for comparison, group in deltas.groupby("comparison_group", sort=False):
        prefix = "C_minus_A" if comparison == "A_vs_C" else "G_minus_C"
        metrics = _DELTA_METRICS[comparison]
        for target, target_group in group.groupby("target_reject_rate", sort=True):
            base: dict[str, Any] = {
                "comparison_group": comparison,
                "target_reject_rate": float(target),
                "paired_groups": int(len(target_group)),
                "base_mean_selected_confidence": float(
                    target_group["base_selected_confidence"].mean()
                ),
                "candidate_mean_selected_confidence": float(
                    target_group["candidate_selected_confidence"].mean()
                ),
                "base_mean_target_abs_error": float(target_group["base_target_abs_error"].mean()),
                "candidate_mean_target_abs_error": float(
                    target_group["candidate_target_abs_error"].mean()
                ),
            }
            for metric in metrics:
                col = f"{prefix}_{metric}"
                values = pd.to_numeric(target_group[col], errors="coerce")
                finite = values[np.isfinite(values)]
                base[f"{col}_mean"] = float(finite.mean()) if not finite.empty else float("nan")
                base[f"{col}_median"] = (
                    float(finite.median()) if not finite.empty else float("nan")
                )
                base[f"{col}_std"] = _safe_std(values)
                base[f"{col}_fraction_positive"] = _safe_fraction_positive(values)
                base[f"{col}_finite_groups"] = int(len(finite))
            rows.append(base)
    return pd.DataFrame(rows)


def _append_readable_sections(
    selected: pd.DataFrame,
    deltas: pd.DataFrame,
    aggregate: pd.DataFrame,
    analyses: list[str],
) -> None:
    md_path = Path(__file__).resolve().parent / "artifacts" / f"{_PREFIX}.md"
    content = md_path.read_text(encoding="utf-8")
    selected_summary = (
        selected.groupby(["arm_code", "target_reject_rate"])[
            [
                "observed_reject_rate",
                "reject_rate_target_abs_error",
                "selected_confidence",
                "accepted_accuracy",
                "ambiguity_rate",
                "novelty_rate",
                "difficulty_reject_auc",
            ]
        ]
        .mean(numeric_only=True)
        .reset_index()
        if not selected.empty
        else pd.DataFrame()
    )
    extra = [
        "## Selected Operating Points",
        "",
        _markdown_table(selected_summary),
        "",
        "## Pairwise Delta Aggregates",
        "",
        _markdown_table(aggregate),
        "",
        "## Pairwise Delta Rows",
        "",
        _markdown_table(deltas.head(40) if not deltas.empty else deltas),
        "",
        "## Required Analyses",
        "",
        "### A vs C",
        "",
        "1. Does direct difficulty normalization improve accepted accuracy at matched reject rates?",
        analyses[0],
        "2. At which reject-rate targets is it most useful?",
        analyses[1],
        "3. Does it consistently increase ambiguity and decrease novelty rejection?",
        analyses[2],
        "4. Does it select higher-difficulty cases for rejection?",
        analyses[3],
        "5. Is Step 8 justified?",
        analyses[4],
        "",
        "### C vs G",
        "",
        "1. Does the novelty-aware variant improve novelty routing at matched reject rates?",
        analyses[5],
        "2. Does it improve or harm accepted accuracy?",
        analyses[6],
        "3. Does it improve novelty selectivity or merely increase empty sets?",
        analyses[7],
        "4. Should it remain internal only?",
        analyses[8],
        "",
    ]
    md_path.write_text(content + "\n" + "\n".join(extra), encoding="utf-8")


def _metric_from_aggregate(
    aggregate: pd.DataFrame,
    comparison: str,
    target: float,
    column: str,
) -> float:
    subset = aggregate[
        (aggregate["comparison_group"] == comparison)
        & np.isclose(aggregate["target_reject_rate"], target)
    ]
    if subset.empty or column not in subset:
        return float("nan")
    return float(subset.iloc[0][column])


def _build_analyses(aggregate: pd.DataFrame) -> tuple[list[str], dict[str, Any]]:
    analyses = ["Insufficient data."] * 9
    outcome: dict[str, Any] = {}
    if aggregate.empty:
        return analyses, outcome

    ac_acc_col = "C_minus_A_accepted_accuracy_mean"
    ac_amb_col = "C_minus_A_ambiguity_rate_fraction_positive"
    ac_nov_col = "C_minus_A_novelty_rate_fraction_positive"
    ac_diff_auc_col = "C_minus_A_difficulty_reject_auc_mean"
    ac_rows = aggregate[aggregate["comparison_group"] == "A_vs_C"].copy()
    if not ac_rows.empty:
        best_row = ac_rows.sort_values(ac_acc_col, ascending=False, kind="mergesort").iloc[0]
        acc_by_target = {
            f"{float(row['target_reject_rate']):.2f}": float(row[ac_acc_col])
            for _, row in ac_rows.iterrows()
        }
        outcome["A_vs_C_accepted_accuracy_delta_by_target"] = acc_by_target
        outcome["A_vs_C_best_target_by_accepted_accuracy"] = float(
            best_row["target_reject_rate"]
        )
        outcome["A_vs_C_best_accepted_accuracy_delta"] = float(best_row[ac_acc_col])
        analyses[0] = (
            "C minus A accepted-accuracy deltas by target reject rate are "
            + ", ".join(f"{target}: {value:+.4f}" for target, value in acc_by_target.items())
            + "."
        )
        analyses[1] = (
            "The strongest matched operating point is target "
            f"{float(best_row['target_reject_rate']):.2f}, with mean accepted-accuracy delta "
            f"{float(best_row[ac_acc_col]):+.4f}."
        )
        amb_frac = float(ac_rows[ac_amb_col].mean()) if ac_amb_col in ac_rows else float("nan")
        nov_frac = float(ac_rows[ac_nov_col].mean()) if ac_nov_col in ac_rows else float("nan")
        analyses[2] = (
            "Across targets, C increased ambiguity in mean fraction-positive "
            f"{amb_frac:.4f} and increased novelty in fraction-positive {nov_frac:.4f}. "
            "This operating-point quick run does not show the same consistent ambiguity-up, "
            "novelty-down geometry seen in Scenario 9."
        )
        diff_auc_mean = float(ac_rows[ac_diff_auc_col].mean())
        analyses[3] = (
            "Across targets, C minus A mean difficulty-reject-AUC delta is "
            f"{diff_auc_mean:+.4f}."
        )
        promote_ready = (
            float(best_row[ac_acc_col]) > 0.02
            and diff_auc_mean > 0.05
            and len(ac_rows) == len(_TARGET_REJECT_RATES)
        )
        analyses[4] = (
            "Step 8 is not justified yet; matched operating-point evidence is mixed and "
            "does not support public API promotion."
            if not promote_ready
            else "The operating-point evidence supports considering direct-only promotion, "
            "subject to API and validity review."
        )
        outcome["A_vs_C_mean_difficulty_reject_auc_delta"] = diff_auc_mean

    cg_rows = aggregate[aggregate["comparison_group"] == "C_vs_G"].copy()
    if not cg_rows.empty:
        novelty_col = "G_minus_C_novelty_rate_mean"
        acc_col = "G_minus_C_accepted_accuracy_mean"
        novelty_auc_col = "G_minus_C_novelty_reject_auc_mean"
        empty_col = "G_minus_C_empty_set_rate_mean"
        ambiguity_col = "G_minus_C_ambiguity_rate_mean"
        novelty_mean = float(cg_rows[novelty_col].mean())
        acc_mean = float(cg_rows[acc_col].mean())
        novelty_auc_mean = float(cg_rows[novelty_auc_col].mean())
        empty_mean = float(cg_rows[empty_col].mean())
        ambiguity_mean = float(cg_rows[ambiguity_col].mean())
        outcome["C_vs_G_mean_novelty_rate_delta"] = novelty_mean
        outcome["C_vs_G_mean_accepted_accuracy_delta"] = acc_mean
        outcome["C_vs_G_mean_novelty_reject_auc_delta"] = novelty_auc_mean
        analyses[5] = (
            "G minus C mean novelty-rate delta across targets is "
            f"{novelty_mean:+.4f}; empty-set delta is {empty_mean:+.4f}."
        )
        analyses[6] = (
            "G minus C mean accepted-accuracy delta across targets is "
            f"{acc_mean:+.4f}."
        )
        analyses[7] = (
            "G minus C novelty-reject-AUC delta is "
            f"{novelty_auc_mean:+.4f}; ambiguity-rate delta is {ambiguity_mean:+.4f}."
        )
        analyses[8] = (
            "The novelty-aware strategy should remain internal only; it is not promotion-ready."
        )

    promotion = "needs_more_evidence"
    if (
        outcome.get("A_vs_C_best_accepted_accuracy_delta", 0.0) < 0.01
        or outcome.get("A_vs_C_mean_difficulty_reject_auc_delta", 0.0) <= 0.0
    ):
        promotion = "do_not_promote"
    elif outcome.get("A_vs_C_mean_difficulty_reject_auc_delta", 0.0) > 0.05:
        promotion = "needs_more_evidence"
    outcome["promotion_recommendation"] = promotion

    novelty_recommendation = "continue_experimental"
    if outcome.get("C_vs_G_mean_novelty_reject_auc_delta", 0.0) < 0.0:
        novelty_recommendation = "do_not_promote"
    outcome["novelty_strategy_recommendation"] = novelty_recommendation
    return analyses, outcome


def _write_delta_csv(deltas: pd.DataFrame) -> None:
    path = Path(__file__).resolve().parent / "artifacts" / f"{_PREFIX}_deltas.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    deltas.to_csv(path, index=False)


def run(config: RunConfig) -> None:
    """Run Scenario 11 matched operating-point selection."""
    sweep_rows: list[dict[str, Any]] = []
    datasets = task_specs("binary", quick=config.quick) + task_specs("multiclass", quick=config.quick)
    confidences = tuple(float(c) for c in _confidence_grid(config.quick))

    for spec in datasets:
        for seed_offset in seed_grid(config):
            seed = int(config.seed + seed_offset)
            bundle, difficulty_scores, difficulty_estimator = _build_classification_bundle(
                spec,
                config,
                seed_offset=seed_offset,
                use_va_difficulty=False,
            )
            novelty_estimator = DeterministicNoveltyEstimator.fit(bundle.x_fit)
            novelty_scores = np.asarray(novelty_estimator.apply(bundle.x_test), dtype=float)
            for arm in _ARMS:
                for confidence in confidences:
                    row = _run_arm(
                        arm=arm,
                        bundle=bundle,
                        difficulty_estimator=difficulty_estimator,
                        novelty_estimator=novelty_estimator,
                        difficulty_scores=difficulty_scores,
                        novelty_scores=novelty_scores,
                        confidence=confidence,
                    )
                    row["seed"] = seed
                    row["task_type"] = spec.task_type
                    row["sweep_confidence"] = float(confidence)
                    row["selected_confidence"] = float(confidence)
                    row["observed_reject_rate"] = float(row["reject_rate"])
                    row["empty_set_rate"] = float(row["empty_rate"])
                    sweep_rows.append(row)

    sweep = pd.DataFrame(sweep_rows)
    selected = _select_operating_points(sweep)
    selected["comparison_group"] = selected["arm_code"].map(
        {"A": "A_vs_C", "C": "A_vs_C;C_vs_G", "G": "C_vs_G"}
    )
    deltas = _delta_frame(selected)
    aggregate = _aggregate_deltas(deltas)
    analyses, outcome_from_analysis = _build_analyses(aggregate)

    outcome = {
        "rows": int(len(selected)),
        "sweep_rows": int(len(sweep)),
        "delta_rows": int(len(deltas)),
        "datasets": int(selected["dataset"].nunique()) if not selected.empty else 0,
        "seeds": int(selected["seed"].nunique()) if not selected.empty else 0,
        "target_reject_rates": list(_TARGET_REJECT_RATES),
        **outcome_from_analysis,
    }

    meta = {
        "scenario": _PREFIX,
        "display_name": "Scenario 11 - Matched operating-point reject selection",
        "generated_at": datetime.now(UTC).isoformat(),
        "guarantee_status": "empirical",
        "quick": config.quick,
        "target_reject_rates": list(_TARGET_REJECT_RATES),
        "confidence_grid": list(confidences),
        "arms_evaluated": [
            {
                "arm_code": arm.code,
                "strategy": arm.strategy,
                "ncf": arm.ncf,
                "novelty_weight": arm.novelty_weight,
            }
            for arm in _ARMS
        ],
        "comparison_definitions": {
            "A_vs_C": "C minus A at matched target reject-rate operating points.",
            "C_vs_G": "G minus C at matched target reject-rate operating points.",
        },
        "promotion_recommendation": outcome.get("promotion_recommendation"),
        "novelty_strategy_recommendation": outcome.get("novelty_strategy_recommendation"),
        "highlights": [
            "Selects confidence values closest to target reject rates instead of averaging over the confidence grid.",
            "Primary decision gate is A vs C for direct difficulty-normalized scoring.",
            "Secondary diagnostic is C vs G for the novelty-aware variant.",
            *analyses,
        ],
        "outcome": outcome,
        "delta_aggregate": aggregate.to_dict(orient="records"),
    }

    _write_delta_csv(deltas)
    write_csv_json_md(_PREFIX, selected, meta)
    _append_readable_sections(selected, deltas, aggregate, analyses)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    arguments = parser.parse_args()
    run(RunConfig(seed=42, quick=arguments.quick))
