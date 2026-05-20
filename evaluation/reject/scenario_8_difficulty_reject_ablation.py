"""Scenario 8: difficulty-estimator reject ablation on the existing classification path.

This scenario measures the already-implemented indirect path:

    difficulty_estimator -> VennAbers probability scaling -> reject NCF -> ConformalClassifier

It does not modify reject scoring. Instead, it compares reject behavior with and
without a deterministic fitted difficulty estimator under the public `default`
and `ensured` reject NCF modes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from calibrated_explanations.ce_agent_utils import ensure_ce_first_wrapper, fit_and_calibrate
from calibrated_explanations.core.reject.policy import RejectPolicy
from calibrated_explanations.explanations.reject import RejectPolicySpec

from .common_reject import (
    ClassificationBundle,
    DatasetSpec,
    RunConfig,
    accepted_accuracy,
    breakdown_from_reject_output,
    confidence_grid,
    empirical_coverage,
    load_dataset,
    seed_grid,
    split_dataset,
    task_specs,
    write_csv_json_md,
)

_NCFS = ("default", "ensured")
_W = 0.5


@dataclass
class DeterministicDifficultyEstimator:
    """Small evaluation-only difficulty estimator.

    The estimator is deterministic, exposes ``fitted=True`` and ``apply(X)``,
    and returns positive finite scores suitable for the current CE validation
    and Venn-Abers scaling path.
    """

    center_: np.ndarray
    scale_: np.ndarray
    fitted: bool = True

    @classmethod
    def fit(cls, x: np.ndarray) -> DeterministicDifficultyEstimator:
        """Fit a deterministic scale model from training features."""
        x_arr = np.asarray(x, dtype=float)
        center = np.mean(x_arr, axis=0)
        scale = np.std(x_arr, axis=0)
        scale = np.where(np.isfinite(scale) & (scale > 1e-9), scale, 1.0)
        return cls(center_=center, scale_=scale, fitted=True)

    def apply(self, x: np.ndarray) -> np.ndarray:
        """Return positive finite per-instance difficulty scores."""
        x_arr = np.asarray(x, dtype=float)
        normalized = (x_arr - self.center_) / self.scale_
        squared_radius = np.mean(np.square(normalized), axis=1)
        scores = 1.0 + np.sqrt(np.maximum(squared_radius, 0.0))
        scores = np.where(np.isfinite(scores) & (scores > 0.0), scores, 1.0)
        return scores.astype(float)


def _build_classification_bundle(
    spec: DatasetSpec,
    config: RunConfig,
    *,
    seed_offset: int,
    difficulty_estimator: Any | None,
) -> ClassificationBundle:
    """Mirror the common classification builder while allowing calibrate kwargs."""
    dataset_name, x_all, y_all, feature_names = load_dataset(spec)
    seed = config.seed + seed_offset
    x_fit, x_cal, x_test, y_fit, y_cal, y_test = split_dataset(
        x_all,
        y_all,
        seed=seed,
        stratify=True,
    )
    model = RandomForestClassifier(
        n_estimators=60 if config.quick else 120,
        random_state=seed,
        max_depth=8 if config.quick else None,
        n_jobs=1,
    )
    wrapper = ensure_ce_first_wrapper(model)
    fit_and_calibrate(
        wrapper,
        x_fit,
        y_fit,
        x_cal,
        y_cal,
        explainer_kwargs={"difficulty_estimator": difficulty_estimator},
    )
    baseline_pred = model.predict(x_test)
    baseline_proba = model.predict_proba(x_test)
    return ClassificationBundle(
        dataset_name=dataset_name,
        feature_names=feature_names,
        wrapper=wrapper,
        x_fit=x_fit,
        y_fit=y_fit,
        x_cal=x_cal,
        y_cal=y_cal,
        x_test=x_test,
        y_test=y_test,
        baseline_pred=np.asarray(baseline_pred),
        baseline_proba=np.asarray(baseline_proba),
    )


def _mean_or_nan(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


def _safe_rate(mask: np.ndarray) -> float:
    if mask.size == 0:
        return float("nan")
    return float(np.mean(mask))


def _singleton_error_rate(breakdown: dict[str, Any]) -> float:
    if not bool(breakdown.get("error_rate_defined", False)):
        return float("nan")
    return float(breakdown.get("error_rate", float("nan")))


def _empirical_coverage_or_nan(prediction_set: Any, y_true: np.ndarray) -> float:
    """Return empirical coverage only when set columns align with label indices."""
    if prediction_set is None:
        return float("nan")
    prediction_set_arr = np.asarray(prediction_set, dtype=bool)
    if prediction_set_arr.ndim != 2 or prediction_set_arr.shape[0] != len(y_true):
        return float("nan")
    y_arr = np.asarray(y_true, dtype=int).reshape(-1)
    if y_arr.size == 0:
        return float("nan")
    if np.min(y_arr) < 0 or np.max(y_arr) >= prediction_set_arr.shape[1]:
        return float("nan")
    return empirical_coverage(prediction_set_arr, y_arr)


def _pp(value: float) -> str:
    """Format a rate delta as percentage points."""
    return f"{value * 100.0:+.1f} pp"


def _pairwise_row(table: pd.DataFrame, *, ncf: str, use_difficulty: bool) -> pd.Series | None:
    subset = table[(table["ncf"] == ncf) & (table["use_difficulty"] == use_difficulty)]
    if subset.empty:
        return None
    return subset.iloc[0]


def _format_md_scalar(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return "nan"
        return f"{float(value):.4f}"
    return str(value)


def _markdown_table(table: pd.DataFrame) -> str:
    if table.empty:
        return "_No rows generated._"
    headers = list(table.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(_format_md_scalar(row[col]) for col in headers) + " |")
    return "\n".join(lines)


def _append_readable_sections(
    prefix: str,
    arm_summary: pd.DataFrame,
    confidence_arm_summary: pd.DataFrame,
    confidence_pairwise_table: pd.DataFrame,
    pairwise_table: pd.DataFrame,
    integrity_table: pd.DataFrame,
) -> None:
    md_path = Path(__file__).resolve().parent / "artifacts" / f"{prefix}.md"
    content = md_path.read_text(encoding="utf-8")
    extra_sections = [
        "## Arm Summary",
        "",
        "This table averages each arm across all datasets, seeds, and confidence levels.",
        "",
        _markdown_table(arm_summary),
        "",
        "## By Confidence And Arm",
        "",
        "This table keeps the reject operating point visible instead of averaging across the whole epsilon sweep.",
        "`empirical_coverage` is computed from the returned prediction sets; `coverage_gap = empirical_coverage - confidence`.",
        "",
        _markdown_table(confidence_arm_summary),
        "",
        "## Difficulty Effect By Confidence And NCF",
        "",
        "This table compares `use_difficulty=yes` against `use_difficulty=no` at the same confidence and NCF.",
        "Negative `coverage_gap_delta` means the difficulty-enabled arm covered fewer true labels at that operating point.",
        "",
        _markdown_table(confidence_pairwise_table),
        "",
        "## Difficulty Effect By NCF",
        "",
        "This table compares `use_difficulty=yes` against `use_difficulty=no` within each public reject NCF.",
        "Negative `accept_rate_delta` means the difficulty-enabled arm rejects more aggressively.",
        "Positive `rejected_error_capture_rate_delta` means the difficulty-enabled arm captures more mistakes in the rejected subset.",
        "",
        _markdown_table(pairwise_table),
        "",
        "## Integrity Audit",
        "",
        "These checks flag impossible reject geometry and whether ambiguity could be coming from a fallback path without prediction sets.",
        "All residuals should stay near zero; positive `positive_ambiguity_without_prediction_set_rows` would be suspicious.",
        "",
        _markdown_table(integrity_table),
        "",
    ]
    md_path.write_text(content + "\n" + "\n".join(extra_sections), encoding="utf-8")


def run(config: RunConfig) -> None:
    """Measure the existing effect of difficulty on reject decisions."""
    rows: list[dict[str, float | str | int | bool]] = []
    datasets = task_specs("binary", quick=config.quick) + task_specs(
        "multiclass", quick=config.quick
    )
    confidences = tuple(float(c) for c in confidence_grid(config.quick))

    for spec in datasets:
        for seed_offset in seed_grid(config):
            actual_seed = int(config.seed + seed_offset)
            reference_bundle = _build_classification_bundle(
                spec,
                config,
                seed_offset=seed_offset,
                difficulty_estimator=None,
            )
            reference_difficulty = DeterministicDifficultyEstimator.fit(reference_bundle.x_fit)
            difficulty_scores = np.asarray(reference_difficulty.apply(reference_bundle.x_test))

            for use_difficulty in (False, True):
                bundle = _build_classification_bundle(
                    spec,
                    config,
                    seed_offset=seed_offset,
                    difficulty_estimator=reference_difficulty if use_difficulty else None,
                )
                full_accuracy = float(np.mean(bundle.baseline_pred == bundle.y_test))
                errors = np.asarray(bundle.baseline_pred != bundle.y_test, dtype=bool)

                for ncf in _NCFS:
                    policy = RejectPolicySpec(RejectPolicy.FLAG, ncf=ncf, w=_W)
                    for confidence in confidences:
                        result = bundle.wrapper.predict(
                            bundle.x_test,
                            reject_policy=policy,
                            confidence=confidence,
                        )
                        breakdown = breakdown_from_reject_output(
                            result,
                            default_confidence=float(confidence),
                        )
                        rejected = np.asarray(breakdown["rejected"], dtype=bool)
                        accepted = ~rejected
                        set_sizes = np.asarray(breakdown["prediction_set_size"], dtype=int)
                        prediction_set = breakdown.get("prediction_set")

                        accepted_acc = accepted_accuracy(
                            bundle.y_test,
                            bundle.baseline_pred,
                            accepted,
                        )
                        rejected_errors = errors & rejected
                        total_errors = int(np.sum(errors))
                        rejected_error_capture_rate = (
                            float(np.sum(rejected_errors) / total_errors)
                            if total_errors > 0
                            else float("nan")
                        )
                        reject_rate = float(breakdown["reject_rate"])
                        ambiguity_rate = float(breakdown["ambiguity_rate"])
                        novelty_rate = float(breakdown["novelty_rate"])
                        accept_rate = float(np.mean(accepted))
                        empirical_cov = _empirical_coverage_or_nan(prediction_set, bundle.y_test)
                        coverage_gap = (
                            empirical_cov - float(confidence)
                            if np.isfinite(empirical_cov)
                            else float("nan")
                        )
                        ambiguity_equals_novelty = bool(
                            np.isclose(ambiguity_rate, novelty_rate, atol=1e-12)
                        )
                        ambiguity_equals_novelty_positive = bool(
                            ambiguity_equals_novelty and max(ambiguity_rate, novelty_rate) > 1e-12
                        )

                        rows.append(
                            {
                                "task_type": spec.task_type,
                                "dataset": spec.name,
                                "seed": actual_seed,
                                "confidence": float(confidence),
                                "epsilon": float(breakdown["epsilon"]),
                                "n_train": int(len(bundle.x_fit)),
                                "n_cal": int(len(bundle.x_cal)),
                                "n_test": int(len(bundle.x_test)),
                                "ncf": ncf,
                                "use_difficulty": bool(use_difficulty),
                                "arm": f"{ncf}|difficulty={int(use_difficulty)}",
                                "accept_rate": accept_rate,
                                "reject_rate": reject_rate,
                                "ambiguity_rate": ambiguity_rate,
                                "novelty_rate": novelty_rate,
                                "accepted_accuracy": accepted_acc,
                                "full_accuracy": full_accuracy,
                                "accuracy_delta": (
                                    accepted_acc - full_accuracy
                                    if np.isfinite(accepted_acc)
                                    else float("nan")
                                ),
                                "singleton_error_rate": _singleton_error_rate(breakdown),
                                "error_rate_defined": bool(breakdown["error_rate_defined"]),
                                "rejected_error_capture_rate": rejected_error_capture_rate,
                                "mean_difficulty_all": _mean_or_nan(difficulty_scores),
                                "mean_difficulty_accepted": _mean_or_nan(difficulty_scores[accepted]),
                                "mean_difficulty_rejected": _mean_or_nan(difficulty_scores[rejected]),
                                "empty_rate": _safe_rate(set_sizes == 0),
                                "singleton_rate": _safe_rate(set_sizes == 1),
                                "multilabel_rate": _safe_rate(set_sizes >= 2),
                                "empirical_coverage": empirical_cov,
                                "coverage_gap": coverage_gap,
                                "coverage_defined": bool(np.isfinite(empirical_cov)),
                                "has_prediction_set": bool(prediction_set is not None),
                                "reject_partition_residual": reject_rate - ambiguity_rate - novelty_rate,
                                "accept_singleton_residual": accept_rate - _safe_rate(set_sizes == 1),
                                "ambiguity_multilabel_residual": ambiguity_rate - _safe_rate(set_sizes >= 2),
                                "novelty_empty_residual": novelty_rate - _safe_rate(set_sizes == 0),
                                "ambiguity_equals_novelty": ambiguity_equals_novelty,
                                "ambiguity_equals_novelty_positive": ambiguity_equals_novelty_positive,
                                "positive_ambiguity_without_prediction_set": bool(
                                    (prediction_set is None) and (ambiguity_rate > 1e-12)
                                ),
                            }
                        )

    df = pd.DataFrame(rows)
    paired_summary: dict[str, dict[str, float]] = {}
    interpretive_highlights: list[str] = []
    outcome_summary: dict[str, Any] = {
        "rows": int(len(df)),
        "datasets": int(df["dataset"].nunique()) if not df.empty else 0,
        "seeds": int(df["seed"].nunique()) if not df.empty else 0,
        "mean_accept_rate": float(df["accept_rate"].mean()) if not df.empty else float("nan"),
        "mean_accuracy_delta": float(df["accuracy_delta"].mean()) if not df.empty else float("nan"),
    }
    if not df.empty:
        grouped = (
            df.groupby(["ncf", "use_difficulty"])[
                [
                    "accept_rate",
                    "accepted_accuracy",
                    "accuracy_delta",
                    "empirical_coverage",
                    "coverage_gap",
                    "singleton_error_rate",
                    "rejected_error_capture_rate",
                    "mean_difficulty_accepted",
                    "mean_difficulty_rejected",
                ]
            ]
            .mean(numeric_only=True)
            .reset_index()
        )
        for ncf in _NCFS:
            base = grouped[(grouped["ncf"] == ncf) & (~grouped["use_difficulty"])]
            diff = grouped[(grouped["ncf"] == ncf) & (grouped["use_difficulty"])]
            if base.empty or diff.empty:
                continue
            paired_summary[ncf] = {
                "accept_rate_delta": float(diff.iloc[0]["accept_rate"] - base.iloc[0]["accept_rate"]),
                "accepted_accuracy_delta": float(
                    diff.iloc[0]["accepted_accuracy"] - base.iloc[0]["accepted_accuracy"]
                ),
                "accuracy_delta_delta": float(
                    diff.iloc[0]["accuracy_delta"] - base.iloc[0]["accuracy_delta"]
                ),
                "empirical_coverage_delta": float(
                    diff.iloc[0]["empirical_coverage"] - base.iloc[0]["empirical_coverage"]
                ),
                "coverage_gap_delta": float(
                    diff.iloc[0]["coverage_gap"] - base.iloc[0]["coverage_gap"]
                ),
                "singleton_error_rate_delta": float(
                    diff.iloc[0]["singleton_error_rate"] - base.iloc[0]["singleton_error_rate"]
                ),
                "rejected_error_capture_rate_delta": float(
                    diff.iloc[0]["rejected_error_capture_rate"]
                    - base.iloc[0]["rejected_error_capture_rate"]
                ),
                "mean_difficulty_rejected_delta": float(
                    diff.iloc[0]["mean_difficulty_rejected"]
                    - base.iloc[0]["mean_difficulty_rejected"]
                ),
            }

        for ncf in _NCFS:
            base_row = _pairwise_row(grouped, ncf=ncf, use_difficulty=False)
            diff_row = _pairwise_row(grouped, ncf=ncf, use_difficulty=True)
            if base_row is None or diff_row is None:
                continue

            accept_rate_delta = float(diff_row["accept_rate"] - base_row["accept_rate"])
            accepted_accuracy_delta = float(
                diff_row["accepted_accuracy"] - base_row["accepted_accuracy"]
            )
            accuracy_delta_delta = float(diff_row["accuracy_delta"] - base_row["accuracy_delta"])
            reject_capture_delta = float(
                diff_row["rejected_error_capture_rate"] - base_row["rejected_error_capture_rate"]
            )
            singleton_error_delta = float(
                diff_row["singleton_error_rate"] - base_row["singleton_error_rate"]
            )
            diff_gap = float(diff_row["mean_difficulty_rejected"] - diff_row["mean_difficulty_accepted"])

            outcome_summary[f"{ncf}_accept_rate_no_difficulty"] = float(base_row["accept_rate"])
            outcome_summary[f"{ncf}_accept_rate_with_difficulty"] = float(diff_row["accept_rate"])
            outcome_summary[f"{ncf}_accept_rate_delta"] = accept_rate_delta
            outcome_summary[f"{ncf}_accepted_accuracy_delta"] = accepted_accuracy_delta
            outcome_summary[f"{ncf}_accuracy_delta_delta"] = accuracy_delta_delta
            outcome_summary[f"{ncf}_rejected_error_capture_rate_delta"] = reject_capture_delta
            outcome_summary[f"{ncf}_singleton_error_rate_delta"] = singleton_error_delta
            outcome_summary[f"{ncf}_difficulty_gap_with_difficulty"] = diff_gap
            outcome_summary[f"{ncf}_empirical_coverage_no_difficulty"] = float(
                base_row["empirical_coverage"]
            )
            outcome_summary[f"{ncf}_empirical_coverage_with_difficulty"] = float(
                diff_row["empirical_coverage"]
            )
            outcome_summary[f"{ncf}_coverage_gap_delta"] = float(
                diff_row["coverage_gap"] - base_row["coverage_gap"]
            )

            interpretive_highlights.append(
                f"With `{ncf}`, enabling difficulty changed accept_rate by {_pp(accept_rate_delta)}, "
                f"rejected_error_capture_rate by {_pp(reject_capture_delta)}, and accepted_accuracy by {_pp(accepted_accuracy_delta)}."
            )
            interpretive_highlights.append(
                f"With `{ncf}`, mean empirical coverage shifted by {_pp(float(diff_row['empirical_coverage'] - base_row['empirical_coverage']))} across the swept confidence grid."
            )

            if diff_gap > 0:
                interpretive_highlights.append(
                    f"With `{ncf}` and difficulty enabled, rejected instances were harder than accepted ones by {diff_gap:.3f} mean difficulty units."
                )

            if accept_rate_delta < 0.0 and reject_capture_delta > 0.0 and accepted_accuracy_delta < 0.0:
                interpretive_highlights.append(
                    f"For `{ncf}`, the current difficulty path acts mainly as a stricter reject gate: it captures more errors, but at the cost of accepting far fewer instances and lowering accepted accuracy."
                )
            elif accept_rate_delta < 0.0 and reject_capture_delta > 0.0:
                interpretive_highlights.append(
                    f"For `{ncf}`, the current difficulty path increases selectivity and error capture without a clean accuracy gain signal."
                )

        overall_gap = df[df["use_difficulty"]].copy()
        if not overall_gap.empty:
            overall_gap["difficulty_gap"] = (
                overall_gap["mean_difficulty_rejected"] - overall_gap["mean_difficulty_accepted"]
            )
            outcome_summary["mean_difficulty_gap_with_difficulty"] = float(
                overall_gap["difficulty_gap"].mean()
            )

    arm_summary = pd.DataFrame()
    confidence_arm_summary = pd.DataFrame()
    confidence_pairwise_table = pd.DataFrame()
    pairwise_table = pd.DataFrame()
    integrity_table = pd.DataFrame()
    if not df.empty:
        arm_summary = (
            df.groupby(["ncf", "use_difficulty"])[
                [
                    "accept_rate",
                    "accepted_accuracy",
                    "accuracy_delta",
                    "empirical_coverage",
                    "coverage_gap",
                    "rejected_error_capture_rate",
                    "mean_difficulty_accepted",
                    "mean_difficulty_rejected",
                    "singleton_rate",
                    "empty_rate",
                    "multilabel_rate",
                ]
            ]
            .mean(numeric_only=True)
            .reset_index()
        )
        arm_summary["use_difficulty"] = arm_summary["use_difficulty"].map(
            lambda value: "yes" if bool(value) else "no"
        )
        confidence_arm_summary = (
            df.groupby(["confidence", "epsilon", "ncf", "use_difficulty"])[
                [
                    "accept_rate",
                    "accepted_accuracy",
                    "rejected_error_capture_rate",
                    "empirical_coverage",
                    "coverage_gap",
                    "ambiguity_rate",
                    "novelty_rate",
                    "singleton_rate",
                ]
            ]
            .mean(numeric_only=True)
            .reset_index()
            .sort_values(["ncf", "confidence", "use_difficulty"], kind="mergesort")
        )
        confidence_arm_summary["use_difficulty"] = confidence_arm_summary["use_difficulty"].map(
            lambda value: "yes" if bool(value) else "no"
        )

        pairwise_rows: list[dict[str, float | str]] = []
        for ncf, metrics in paired_summary.items():
            pairwise_rows.append(
                {
                    "ncf": ncf,
                    "accept_rate_delta": metrics["accept_rate_delta"],
                    "accepted_accuracy_delta": metrics["accepted_accuracy_delta"],
                    "accuracy_delta_delta": metrics["accuracy_delta_delta"],
                    "empirical_coverage_delta": metrics["empirical_coverage_delta"],
                    "coverage_gap_delta": metrics["coverage_gap_delta"],
                    "rejected_error_capture_rate_delta": metrics[
                        "rejected_error_capture_rate_delta"
                    ],
                    "singleton_error_rate_delta": metrics["singleton_error_rate_delta"],
                    "mean_difficulty_rejected_delta": metrics[
                        "mean_difficulty_rejected_delta"
                    ],
                }
            )
        pairwise_table = pd.DataFrame(pairwise_rows)

        confidence_pairwise_rows: list[dict[str, float | str]] = []
        by_confidence = (
            df.groupby(["confidence", "epsilon", "ncf", "use_difficulty"])[
                [
                    "accept_rate",
                    "accepted_accuracy",
                    "rejected_error_capture_rate",
                    "empirical_coverage",
                    "coverage_gap",
                ]
            ]
            .mean(numeric_only=True)
            .reset_index()
        )
        for confidence in sorted(by_confidence["confidence"].unique()):
            confidence_slice = by_confidence[by_confidence["confidence"] == confidence]
            for ncf in _NCFS:
                base = confidence_slice[
                    (confidence_slice["ncf"] == ncf) & (~confidence_slice["use_difficulty"])
                ]
                diff = confidence_slice[
                    (confidence_slice["ncf"] == ncf) & (confidence_slice["use_difficulty"])
                ]
                if base.empty or diff.empty:
                    continue
                confidence_pairwise_rows.append(
                    {
                        "confidence": float(confidence),
                        "epsilon": float(base.iloc[0]["epsilon"]),
                        "ncf": ncf,
                        "accept_rate_delta": float(
                            diff.iloc[0]["accept_rate"] - base.iloc[0]["accept_rate"]
                        ),
                        "accepted_accuracy_delta": float(
                            diff.iloc[0]["accepted_accuracy"] - base.iloc[0]["accepted_accuracy"]
                        ),
                        "rejected_error_capture_rate_delta": float(
                            diff.iloc[0]["rejected_error_capture_rate"]
                            - base.iloc[0]["rejected_error_capture_rate"]
                        ),
                        "empirical_coverage_delta": float(
                            diff.iloc[0]["empirical_coverage"] - base.iloc[0]["empirical_coverage"]
                        ),
                        "coverage_gap_delta": float(
                            diff.iloc[0]["coverage_gap"] - base.iloc[0]["coverage_gap"]
                        ),
                    }
                )
        confidence_pairwise_table = pd.DataFrame(confidence_pairwise_rows).sort_values(
            ["ncf", "confidence"], kind="mergesort"
        )

        integrity_table = (
            df.groupby(["ncf", "use_difficulty"])
            .agg(
                rows=("confidence", "size"),
                max_abs_reject_partition_residual=(
                    "reject_partition_residual",
                    lambda values: float(np.max(np.abs(values))),
                ),
                max_abs_accept_singleton_residual=(
                    "accept_singleton_residual",
                    lambda values: float(np.max(np.abs(values))),
                ),
                max_abs_ambiguity_multilabel_residual=(
                    "ambiguity_multilabel_residual",
                    lambda values: float(np.max(np.abs(values))),
                ),
                max_abs_novelty_empty_residual=(
                    "novelty_empty_residual",
                    lambda values: float(np.max(np.abs(values))),
                ),
                equal_ambiguity_novelty_rows=("ambiguity_equals_novelty", "sum"),
                equal_positive_ambiguity_novelty_rows=(
                    "ambiguity_equals_novelty_positive",
                    "sum",
                ),
                coverage_defined_rows=("coverage_defined", "sum"),
                positive_ambiguity_without_prediction_set_rows=(
                    "positive_ambiguity_without_prediction_set",
                    "sum",
                ),
                min_coverage_gap=("coverage_gap", "min"),
                max_coverage_gap=("coverage_gap", "max"),
            )
            .reset_index()
            .sort_values(["ncf", "use_difficulty"], kind="mergesort")
        )
        integrity_table["use_difficulty"] = integrity_table["use_difficulty"].map(
            lambda value: "yes" if bool(value) else "no"
        )

        outcome_summary["unique_confidences"] = int(df["confidence"].nunique())
        outcome_summary["min_epsilon"] = float(df["epsilon"].min())
        outcome_summary["max_epsilon"] = float(df["epsilon"].max())
        outcome_summary["max_abs_reject_partition_residual"] = float(
            np.max(np.abs(df["reject_partition_residual"]))
        )
        outcome_summary["max_abs_accept_singleton_residual"] = float(
            np.max(np.abs(df["accept_singleton_residual"]))
        )
        outcome_summary["positive_ambiguity_without_prediction_set_rows"] = int(
            df["positive_ambiguity_without_prediction_set"].sum()
        )
        outcome_summary["equal_positive_ambiguity_novelty_rows"] = int(
            df["ambiguity_equals_novelty_positive"].sum()
        )
        outcome_summary["coverage_defined_rows"] = int(df["coverage_defined"].sum())
        outcome_summary["min_empirical_coverage_gap"] = float(df["coverage_gap"].min())
        outcome_summary["max_empirical_coverage_gap"] = float(df["coverage_gap"].max())
        interpretive_highlights.append(
            "The markdown now includes a by-confidence table so the headline summary is no longer averaged over hidden epsilon values."
        )
        interpretive_highlights.append(
            "Integrity checks verify reject_rate = ambiguity_rate + novelty_rate, accepted instances match singleton prediction sets, and no positive ambiguity appears without prediction sets."
        )
        interpretive_highlights.append(
            "Empirical coverage is reported only for rows whose prediction-set columns are label-index aligned; unsupported rows stay `nan` instead of inventing a value."
        )

    meta = {
        "scenario": "scenario_8_difficulty_reject_ablation",
        "display_name": "Scenario 8 — Difficulty estimator reject ablation",
        "guarantee_status": "empirical",
        "quick": config.quick,
        "highlights": [
            "Measures the current indirect difficulty effect through Venn-Abers scaling only; reject scoring itself is unchanged.",
            "Arms compare use_difficulty in {False, True} crossed with reject NCF in {default, ensured}.",
            "Difficulty summary columns use the same deterministic reference estimator in all arms so selection differences are comparable.",
            "This scenario does not test difficulty-normalized reject NCFs; it quantifies the baseline before that experiment.",
            *interpretive_highlights,
        ],
        "outcome": outcome_summary,
        "pairwise_summary": paired_summary,
    }
    write_csv_json_md("scenario_8_difficulty_reject_ablation", df, meta)
    _append_readable_sections(
        "scenario_8_difficulty_reject_ablation",
        arm_summary,
        confidence_arm_summary,
        confidence_pairwise_table,
        pairwise_table,
        integrity_table,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    arguments = parser.parse_args()
    run(RunConfig(seed=42, quick=arguments.quick))
