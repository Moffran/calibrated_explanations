"""Scenario 9: direct difficulty-normalized reject scoring ablation.

This scenario compares the current indirect difficulty path (difficulty-aware
Venn-Abers probabilities) against the experimental direct strategy that
normalizes reject nonconformity scores by per-instance difficulty before
conformal p-values / prediction sets are computed.

Classification arms:
    A. no VA difficulty, builtin strategy, ncf=default
    B. VA difficulty, builtin strategy, ncf=default
    C. no VA difficulty, experimental strategy, ncf=default
    D. VA difficulty, experimental strategy, ncf=default
    E. no VA difficulty, experimental strategy, ncf=ensured
    F. VA difficulty, experimental strategy, ncf=ensured

Primary scientific contrast: A vs C.
Diagnostic double-count checks: D and F.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from calibrated_explanations import RejectPolicySpec, WrapCalibratedExplainer

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

_W = 0.5


@dataclass(frozen=True)
class ArmSpec:
    """One ablation arm for Scenario 9."""

    code: str
    use_va_difficulty: bool
    strategy: str
    ncf: str

    @property
    def difficulty_normalized(self) -> bool:
        return self.strategy == "experimental.difficulty_normalized"

    @property
    def double_count_difficulty(self) -> bool:
        return self.use_va_difficulty and self.difficulty_normalized


@dataclass
class DeterministicDifficultyEstimator:
    """Small deterministic estimator used for controlled difficulty ablations."""

    center_: np.ndarray
    scale_: np.ndarray
    fitted: bool = True

    @classmethod
    def fit(cls, x: np.ndarray) -> DeterministicDifficultyEstimator:
        x_arr = np.asarray(x, dtype=float)
        center = np.mean(x_arr, axis=0)
        scale = np.std(x_arr, axis=0)
        scale = np.where(np.isfinite(scale) & (scale > 1e-9), scale, 1.0)
        return cls(center_=center, scale_=scale, fitted=True)

    def apply(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        normalized = (x_arr - self.center_) / self.scale_
        squared_radius = np.mean(np.square(normalized), axis=1)
        scores = 1.0 + np.sqrt(np.maximum(squared_radius, 0.0))
        scores = np.where(np.isfinite(scores) & (scores > 0.0), scores, 1.0)
        return scores.astype(float)


_ARMS: tuple[ArmSpec, ...] = (
    ArmSpec("A", use_va_difficulty=False, strategy="builtin.default", ncf="default"),
    ArmSpec("B", use_va_difficulty=True, strategy="builtin.default", ncf="default"),
    ArmSpec(
        "C",
        use_va_difficulty=False,
        strategy="experimental.difficulty_normalized",
        ncf="default",
    ),
    ArmSpec(
        "D",
        use_va_difficulty=True,
        strategy="experimental.difficulty_normalized",
        ncf="default",
    ),
    ArmSpec(
        "E",
        use_va_difficulty=False,
        strategy="experimental.difficulty_normalized",
        ncf="ensured",
    ),
    ArmSpec(
        "F",
        use_va_difficulty=True,
        strategy="experimental.difficulty_normalized",
        ncf="ensured",
    ),
)


def _build_classification_bundle(
    spec: DatasetSpec,
    config: RunConfig,
    *,
    seed_offset: int,
    use_va_difficulty: bool,
) -> tuple[ClassificationBundle, np.ndarray, DeterministicDifficultyEstimator]:
    """Build one calibrated classification bundle with deterministic splits."""
    dataset_name, x_all, y_all, feature_names = load_dataset(spec)
    seed = config.seed + seed_offset
    x_fit, x_cal, x_test, y_fit, y_cal, y_test = split_dataset(
        x_all,
        y_all,
        seed=seed,
        stratify=True,
    )

    difficulty_estimator = DeterministicDifficultyEstimator.fit(x_fit)
    difficulty_scores_test = np.asarray(difficulty_estimator.apply(x_test), dtype=float)

    model = RandomForestClassifier(
        n_estimators=60 if config.quick else 120,
        random_state=seed,
        max_depth=8 if config.quick else None,
        n_jobs=1,
    )
    wrapper = WrapCalibratedExplainer(model)
    wrapper.fit(x_fit, y_fit)
    wrapper.calibrate(
        x_cal,
        y_cal,
        feature_names=feature_names,
        difficulty_estimator=difficulty_estimator if use_va_difficulty else None,
    )

    baseline_pred = model.predict(x_test)
    baseline_proba = model.predict_proba(x_test)
    return (
        ClassificationBundle(
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
        ),
        difficulty_scores_test,
        difficulty_estimator,
    )


def _safe_mean(values: np.ndarray) -> float:
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


def _difficulty_reject_auc(difficulty: np.ndarray, rejected: np.ndarray) -> float:
    y_true = np.asarray(rejected, dtype=int).reshape(-1)
    scores = np.asarray(difficulty, dtype=float).reshape(-1)
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, scores))
    except ValueError:
        return float("nan")


def _format_scalar(value: Any) -> str:
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
        lines.append("| " + " | ".join(_format_scalar(row[col]) for col in headers) + " |")
    return "\n".join(lines)


def _arm_row(table: pd.DataFrame, arm_code: str) -> pd.Series | None:
    subset = table[table["arm_code"] == arm_code]
    if subset.empty:
        return None
    return subset.iloc[0]


def _matched_reject_bin_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-arm and A-vs-C matched-bin accepted-accuracy tables."""
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    bin_edges = np.linspace(0.0, 1.000001, 11)
    with_bins = df.copy()
    with_bins["reject_rate_bin"] = pd.cut(
        with_bins["reject_rate"],
        bins=bin_edges,
        include_lowest=True,
        right=False,
    )

    per_arm = (
        with_bins.groupby(["reject_rate_bin", "arm_code"])
        .agg(
            n_rows=("accepted_accuracy", "size"),
            mean_reject_rate=("reject_rate", "mean"),
            mean_accepted_accuracy=("accepted_accuracy", "mean"),
        )
        .reset_index()
        .sort_values(["reject_rate_bin", "arm_code"], kind="mergesort")
    )

    ac = per_arm[per_arm["arm_code"].isin(["A", "C"])].copy()
    if ac.empty:
        return per_arm, pd.DataFrame()
    pivot = ac.pivot_table(
        index="reject_rate_bin",
        columns="arm_code",
        values=["mean_accepted_accuracy", "mean_reject_rate", "n_rows"],
        aggfunc="first",
    )
    if ("mean_accepted_accuracy", "A") not in pivot.columns or (
        "mean_accepted_accuracy",
        "C",
    ) not in pivot.columns:
        return per_arm, pd.DataFrame()

    matched = pd.DataFrame(
        {
            "reject_rate_bin": pivot.index.astype(str),
            "A_mean_accepted_accuracy": pivot[("mean_accepted_accuracy", "A")],
            "C_mean_accepted_accuracy": pivot[("mean_accepted_accuracy", "C")],
            "accepted_accuracy_delta_C_minus_A": (
                pivot[("mean_accepted_accuracy", "C")]
                - pivot[("mean_accepted_accuracy", "A")]
            ),
            "A_mean_reject_rate": pivot[("mean_reject_rate", "A")],
            "C_mean_reject_rate": pivot[("mean_reject_rate", "C")],
            "A_rows": pivot[("n_rows", "A")],
            "C_rows": pivot[("n_rows", "C")],
        }
    ).reset_index(drop=True)
    matched = matched.sort_values("reject_rate_bin", kind="mergesort")
    return per_arm, matched


def _append_readable_sections(
    prefix: str,
    arm_summary: pd.DataFrame,
    confidence_arm_summary: pd.DataFrame,
    matched_bins_table: pd.DataFrame,
    per_arm_bins_table: pd.DataFrame,
    analyses: list[str],
) -> None:
    md_path = Path(__file__).resolve().parent / "artifacts" / f"{prefix}.md"
    content = md_path.read_text(encoding="utf-8")
    extra = [
        "## Arm Summary",
        "",
        _markdown_table(arm_summary),
        "",
        "## By Confidence And Arm",
        "",
        _markdown_table(confidence_arm_summary),
        "",
        "## Accepted Accuracy At Matched Reject-Rate Bins",
        "",
        "A-vs-C matched-bin comparison (primary contrast).",
        "",
        _markdown_table(matched_bins_table),
        "",
        "## Per-Arm Reject-Rate Bin Aggregates",
        "",
        _markdown_table(per_arm_bins_table),
        "",
        "## Required Analyses",
        "",
        "1. Does direct normalization increase rejection among high-difficulty instances?",
        analyses[0],
        "2. Does it improve accepted accuracy at comparable reject rates?",
        analyses[1],
        "3. Does it increase ambiguity rate, novelty rate, or both?",
        analyses[2],
        "4. Does using VA difficulty and score normalization together appear to double-count difficulty?",
        analyses[3],
        "5. Which arm is recommended for further development?",
        analyses[4],
        "",
    ]
    md_path.write_text(content + "\n" + "\n".join(extra), encoding="utf-8")


def run(config: RunConfig) -> None:
    """Run Scenario 9 difficulty-normalized reject strategy ablation."""
    rows: list[dict[str, Any]] = []
    datasets = task_specs("binary", quick=config.quick) + task_specs("multiclass", quick=config.quick)
    confidences = tuple(float(c) for c in confidence_grid(config.quick))

    for spec in datasets:
        for seed_offset in seed_grid(config):
            seed = int(config.seed + seed_offset)
            bundles_by_va: dict[bool, ClassificationBundle] = {}
            difficulty_scores_by_va: dict[bool, np.ndarray] = {}
            difficulty_estimators_by_va: dict[bool, DeterministicDifficultyEstimator] = {}

            for use_va in (False, True):
                bundle, difficulty_scores, difficulty_estimator = _build_classification_bundle(
                    spec,
                    config,
                    seed_offset=seed_offset,
                    use_va_difficulty=use_va,
                )
                bundles_by_va[use_va] = bundle
                difficulty_scores_by_va[use_va] = difficulty_scores
                difficulty_estimators_by_va[use_va] = difficulty_estimator

            for arm in _ARMS:
                bundle = bundles_by_va[arm.use_va_difficulty]
                difficulty_scores = difficulty_scores_by_va[arm.use_va_difficulty]
                difficulty_estimator = difficulty_estimators_by_va[arm.use_va_difficulty]
                policy = RejectPolicySpec.flag(ncf=arm.ncf, w=_W)
                errors = np.asarray(bundle.baseline_pred != bundle.y_test, dtype=bool)
                full_accuracy = float(np.mean(bundle.baseline_pred == bundle.y_test))

                # Red-team critical: enable reject-only difficulty for non-VA experimental
                # arms so A vs C isolates direct reject-score normalization.
                if arm.use_va_difficulty:
                    bundle.wrapper.explainer.difficulty_estimator = difficulty_estimator
                elif arm.difficulty_normalized:
                    bundle.wrapper.explainer.difficulty_estimator = difficulty_estimator
                else:
                    bundle.wrapper.explainer.difficulty_estimator = None

                for confidence in confidences:
                    result = bundle.wrapper.predict(
                        bundle.x_test,
                        reject_policy=policy,
                        confidence=confidence,
                        strategy=arm.strategy,
                    )
                    breakdown = breakdown_from_reject_output(
                        result,
                        default_confidence=float(confidence),
                    )
                    rejected = np.asarray(breakdown["rejected"], dtype=bool)
                    accepted = ~rejected
                    set_sizes = np.asarray(breakdown["prediction_set_size"], dtype=int)
                    prediction_set = breakdown.get("prediction_set")
                    metadata = getattr(result, "metadata", {}) or {}

                    accepted_acc = accepted_accuracy(bundle.y_test, bundle.baseline_pred, accepted)
                    rejected_errors = errors & rejected
                    total_errors = int(np.sum(errors))
                    rejected_error_capture_rate = (
                        float(np.sum(rejected_errors) / total_errors)
                        if total_errors > 0
                        else float("nan")
                    )
                    empirical_cov = _empirical_coverage_or_nan(prediction_set, bundle.y_test)
                    coverage_gap = (
                        empirical_cov - float(confidence)
                        if np.isfinite(empirical_cov)
                        else float("nan")
                    )
                    difficulty_auc = _difficulty_reject_auc(difficulty_scores, rejected)
                    difficulty_gap = _safe_mean(difficulty_scores[rejected]) - _safe_mean(
                        difficulty_scores[accepted]
                    )

                    rows.append(
                        {
                            "task_type": spec.task_type,
                            "dataset": spec.name,
                            "seed": seed,
                            "confidence": float(confidence),
                            "epsilon": float(breakdown["epsilon"]),
                            "n_train": int(len(bundle.x_fit)),
                            "n_cal": int(len(bundle.x_cal)),
                            "n_test": int(len(bundle.x_test)),
                            "arm_code": arm.code,
                            "arm_label": (
                                f"{arm.code}|va={int(arm.use_va_difficulty)}|"
                                f"strategy={arm.strategy}|ncf={arm.ncf}"
                            ),
                            "ncf": arm.ncf,
                            "strategy": arm.strategy,
                            "use_va_difficulty": bool(arm.use_va_difficulty),
                            "difficulty_normalized": bool(
                                metadata.get("difficulty_normalized", arm.difficulty_normalized)
                            ),
                            "double_count_difficulty": bool(arm.double_count_difficulty),
                            "accept_rate": float(np.mean(accepted)),
                            "reject_rate": float(breakdown["reject_rate"]),
                            "ambiguity_rate": float(breakdown["ambiguity_rate"]),
                            "novelty_rate": float(breakdown["novelty_rate"]),
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
                            "mean_difficulty_all": _safe_mean(difficulty_scores),
                            "mean_difficulty_accepted": _safe_mean(difficulty_scores[accepted]),
                            "mean_difficulty_rejected": _safe_mean(difficulty_scores[rejected]),
                            "difficulty_gap_rejected_minus_accepted": difficulty_gap,
                            "difficulty_reject_auc": difficulty_auc,
                            "empty_rate": _safe_rate(set_sizes == 0),
                            "singleton_rate": _safe_rate(set_sizes == 1),
                            "multilabel_rate": _safe_rate(set_sizes >= 2),
                            "empirical_coverage": empirical_cov,
                            "coverage_gap": coverage_gap,
                            "coverage_defined": bool(np.isfinite(empirical_cov)),
                        }
                    )

    df = pd.DataFrame(rows)
    outcome_summary: dict[str, Any] = {
        "rows": int(len(df)),
        "datasets": int(df["dataset"].nunique()) if not df.empty else 0,
        "seeds": int(df["seed"].nunique()) if not df.empty else 0,
        "mean_accept_rate": float(df["accept_rate"].mean()) if not df.empty else float("nan"),
        "mean_accuracy_delta": (
            float(df["accuracy_delta"].mean()) if not df.empty else float("nan")
        ),
    }

    arm_summary = pd.DataFrame()
    confidence_arm_summary = pd.DataFrame()
    per_arm_bins = pd.DataFrame()
    matched_bins = pd.DataFrame()
    analyses: list[str] = [
        "Insufficient data.",
        "Insufficient data.",
        "Insufficient data.",
        "Insufficient data.",
        "Insufficient data.",
    ]

    if not df.empty:
        arm_summary = (
            df.groupby(
                [
                    "arm_code",
                    "strategy",
                    "ncf",
                    "use_va_difficulty",
                    "difficulty_normalized",
                    "double_count_difficulty",
                ]
            )[
                [
                    "accept_rate",
                    "reject_rate",
                    "accepted_accuracy",
                    "accuracy_delta",
                    "ambiguity_rate",
                    "novelty_rate",
                    "rejected_error_capture_rate",
                    "mean_difficulty_accepted",
                    "mean_difficulty_rejected",
                    "difficulty_gap_rejected_minus_accepted",
                    "difficulty_reject_auc",
                    "empirical_coverage",
                    "coverage_gap",
                ]
            ]
            .mean(numeric_only=True)
            .reset_index()
            .sort_values("arm_code", kind="mergesort")
        )

        confidence_arm_summary = (
            df.groupby(
                [
                    "confidence",
                    "epsilon",
                    "arm_code",
                    "strategy",
                    "ncf",
                    "use_va_difficulty",
                    "difficulty_normalized",
                    "double_count_difficulty",
                ]
            )[
                [
                    "reject_rate",
                    "accepted_accuracy",
                    "ambiguity_rate",
                    "novelty_rate",
                    "difficulty_gap_rejected_minus_accepted",
                    "difficulty_reject_auc",
                ]
            ]
            .mean(numeric_only=True)
            .reset_index()
            .sort_values(["arm_code", "confidence"], kind="mergesort")
        )

        per_arm_bins, matched_bins = _matched_reject_bin_tables(df)
        a_row = _arm_row(arm_summary, "A")
        b_row = _arm_row(arm_summary, "B")
        c_row = _arm_row(arm_summary, "C")
        d_row = _arm_row(arm_summary, "D")
        e_row = _arm_row(arm_summary, "E")
        f_row = _arm_row(arm_summary, "F")

        if a_row is not None and c_row is not None:
            reject_delta_ac = float(c_row["reject_rate"] - a_row["reject_rate"])
            diff_gap_delta_ac = float(
                c_row["difficulty_gap_rejected_minus_accepted"]
                - a_row["difficulty_gap_rejected_minus_accepted"]
            )
            diff_auc_delta_ac = float(c_row["difficulty_reject_auc"] - a_row["difficulty_reject_auc"])
            outcome_summary["A_vs_C_reject_rate_delta"] = reject_delta_ac
            outcome_summary["A_vs_C_difficulty_gap_delta"] = diff_gap_delta_ac
            outcome_summary["A_vs_C_difficulty_reject_auc_delta"] = diff_auc_delta_ac

            analyses[0] = (
                "Direct normalization (C vs A) changed reject_rate by "
                f"{reject_delta_ac:+.4f}, difficulty-gap by {diff_gap_delta_ac:+.4f}, "
                f"and difficulty_reject_auc by {diff_auc_delta_ac:+.4f}."
            )

            ambiguity_delta_ac = float(c_row["ambiguity_rate"] - a_row["ambiguity_rate"])
            novelty_delta_ac = float(c_row["novelty_rate"] - a_row["novelty_rate"])
            analyses[2] = (
                "For C vs A, ambiguity_rate changed by "
                f"{ambiguity_delta_ac:+.4f} and novelty_rate by {novelty_delta_ac:+.4f}."
            )
            outcome_summary["A_vs_C_ambiguity_rate_delta"] = ambiguity_delta_ac
            outcome_summary["A_vs_C_novelty_rate_delta"] = novelty_delta_ac

        if not matched_bins.empty:
            matched_acc_delta = float(matched_bins["accepted_accuracy_delta_C_minus_A"].mean())
            analyses[1] = (
                "At matched reject-rate bins, C minus A mean accepted_accuracy is "
                f"{matched_acc_delta:+.4f}."
            )
            outcome_summary["A_vs_C_matched_bin_accepted_accuracy_delta"] = matched_acc_delta

        if b_row is not None and d_row is not None and e_row is not None and f_row is not None:
            reject_delta_bd = float(d_row["reject_rate"] - b_row["reject_rate"])
            reject_delta_ef = float(f_row["reject_rate"] - e_row["reject_rate"])
            diff_gap_delta_bd = float(
                d_row["difficulty_gap_rejected_minus_accepted"]
                - b_row["difficulty_gap_rejected_minus_accepted"]
            )
            diff_gap_delta_ef = float(
                f_row["difficulty_gap_rejected_minus_accepted"]
                - e_row["difficulty_gap_rejected_minus_accepted"]
            )
            analyses[3] = (
                "Double-count diagnostics: D-B reject_rate delta "
                f"{reject_delta_bd:+.4f}, F-E reject_rate delta {reject_delta_ef:+.4f}; "
                f"difficulty-gap deltas are {diff_gap_delta_bd:+.4f} and {diff_gap_delta_ef:+.4f}."
            )
            outcome_summary["D_minus_B_reject_rate_delta"] = reject_delta_bd
            outcome_summary["F_minus_E_reject_rate_delta"] = reject_delta_ef
            outcome_summary["D_minus_B_difficulty_gap_delta"] = diff_gap_delta_bd
            outcome_summary["F_minus_E_difficulty_gap_delta"] = diff_gap_delta_ef

        recommendation = "C"
        reason = "primary A-vs-C contrast with direct normalization and no VA double-count risk"
        if a_row is not None and c_row is not None:
            c_acc_delta = float(c_row["accepted_accuracy"] - a_row["accepted_accuracy"])
            c_reject_delta = float(c_row["reject_rate"] - a_row["reject_rate"])
            if c_acc_delta < -0.01 and c_reject_delta > 0.05:
                recommendation = "A"
                reason = "direct normalization appears too conservative in quick run"
        analyses[4] = f"Recommended arm for next iteration: {recommendation} ({reason})."
        outcome_summary["recommended_arm"] = recommendation
        outcome_summary["recommendation_reason"] = reason

    meta = {
        "scenario": "scenario_9_difficulty_normalized_ncf",
        "display_name": "Scenario 9 - Difficulty-normalized reject NCF strategy ablation",
        "guarantee_status": "empirical",
        "quick": config.quick,
        "highlights": [
            "Compares indirect VA-difficulty support against direct experimental difficulty-normalized reject scoring.",
            "Primary scientific contrast is A vs C (default NCF, no VA difficulty in either arm).",
            "Arms D and F are diagnostic for potential difficulty double-counting when VA and score normalization are both enabled.",
            "Includes strategy metadata and difficulty_reject_auc for reject-selectivity diagnostics.",
            "Includes accepted-accuracy comparison at matched reject-rate bins for A vs C.",
            *analyses,
        ],
        "outcome": outcome_summary,
    }

    write_csv_json_md("scenario_9_difficulty_normalized_ncf", df, meta)
    _append_readable_sections(
        "scenario_9_difficulty_normalized_ncf",
        arm_summary,
        confidence_arm_summary,
        matched_bins,
        per_arm_bins,
        analyses,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    arguments = parser.parse_args()
    run(RunConfig(seed=42, quick=arguments.quick))
