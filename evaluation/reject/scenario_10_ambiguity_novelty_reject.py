"""Scenario 10: ambiguity-normalized novelty-penalized reject strategy.

This scenario evaluates the second experimental reject strategy added after
Scenario 9:

    alpha_new(x, y) = alpha_base(x, y) / d_amb(x) + lambda * d_nov(x)

The comparison is intentionally narrow:

    A. builtin.default, ncf=default
    C. experimental.difficulty_normalized, ncf=default
    G. experimental.ambiguity_normalized_novelty_penalized, ncf=default

The primary question is whether adding a separate novelty penalty can move some
rejections from ambiguous multi-label sets toward novelty empty sets without
destroying the accepted-accuracy benefit observed in Scenario 9.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from calibrated_explanations import RejectPolicySpec

from .common_reject import (
    ClassificationBundle,
    RunConfig,
    accepted_accuracy,
    breakdown_from_reject_output,
    confidence_grid,
    seed_grid,
    task_specs,
    write_csv_json_md,
)
from .scenario_9_difficulty_normalized_ncf import (
    DeterministicDifficultyEstimator,
    _build_classification_bundle,
    _difficulty_reject_auc,
    _empirical_coverage_or_nan,
    _format_scalar,
    _markdown_table,
    _safe_mean,
    _safe_rate,
    _singleton_error_rate,
)

_W = 0.5
_NOVELTY_WEIGHT = 0.1


@dataclass(frozen=True)
class ArmSpec:
    """One ablation arm for Scenario 10."""

    code: str
    strategy: str
    ncf: str = "default"
    novelty_weight: float = 0.0

    @property
    def difficulty_normalized(self) -> bool:
        return self.strategy in {
            "experimental.difficulty_normalized",
            "experimental.ambiguity_normalized_novelty_penalized",
        }

    @property
    def novelty_penalized(self) -> bool:
        return self.strategy == "experimental.ambiguity_normalized_novelty_penalized"


@dataclass
class DeterministicNoveltyEstimator:
    """Deterministic evaluation-only novelty estimator.

    The estimator measures excess standardized distance beyond a high quantile
    of the proper-training distribution. It is fitted before calibration and
    uses no calibration labels or residuals.
    """

    center_: np.ndarray
    scale_: np.ndarray
    radius_threshold_: float
    radius_scale_: float
    fitted: bool = True
    fit_source: str = "proper_train"
    uses_calibration_labels: bool = False
    uses_calibration_residuals: bool = False

    @classmethod
    def fit(cls, x: np.ndarray) -> DeterministicNoveltyEstimator:
        x_arr = np.asarray(x, dtype=float)
        center = np.mean(x_arr, axis=0)
        scale = np.std(x_arr, axis=0)
        scale = np.where(np.isfinite(scale) & (scale > 1e-9), scale, 1.0)
        normalized = (x_arr - center) / scale
        radius = np.sqrt(np.maximum(np.mean(np.square(normalized), axis=1), 0.0))
        threshold = float(np.quantile(radius, 0.75))
        spread = float(np.quantile(radius, 0.95) - threshold)
        if not np.isfinite(spread) or spread <= 1e-9:
            spread = 1.0
        return cls(
            center_=center,
            scale_=scale,
            radius_threshold_=threshold,
            radius_scale_=spread,
            fitted=True,
        )

    def apply(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        normalized = (x_arr - self.center_) / self.scale_
        radius = np.sqrt(np.maximum(np.mean(np.square(normalized), axis=1), 0.0))
        novelty = np.maximum(0.0, (radius - self.radius_threshold_) / self.radius_scale_)
        novelty = np.where(np.isfinite(novelty) & (novelty >= 0.0), novelty, 0.0)
        return novelty.astype(float)


_ARMS: tuple[ArmSpec, ...] = (
    ArmSpec("A", strategy="builtin.default"),
    ArmSpec("C", strategy="experimental.difficulty_normalized"),
    ArmSpec(
        "G",
        strategy="experimental.ambiguity_normalized_novelty_penalized",
        novelty_weight=_NOVELTY_WEIGHT,
    ),
)


def _append_readable_sections(
    prefix: str,
    arm_summary: pd.DataFrame,
    confidence_arm_summary: pd.DataFrame,
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
        "## Required Analyses",
        "",
        "1. Does novelty penalization increase novelty/empty-set rejection relative to C?",
        analyses[0],
        "2. Does it reduce ambiguity/multi-label rejection relative to C?",
        analyses[1],
        "3. Does it preserve accepted accuracy relative to C?",
        analyses[2],
        "4. Which arm is recommended for further development?",
        analyses[3],
        "",
    ]
    md_path.write_text(content + "\n" + "\n".join(extra), encoding="utf-8")


def _arm_row(table: pd.DataFrame, arm_code: str) -> pd.Series | None:
    subset = table[table["arm_code"] == arm_code]
    if subset.empty:
        return None
    return subset.iloc[0]


def _run_arm(
    *,
    arm: ArmSpec,
    bundle: ClassificationBundle,
    difficulty_estimator: DeterministicDifficultyEstimator,
    novelty_estimator: DeterministicNoveltyEstimator,
    difficulty_scores: np.ndarray,
    novelty_scores: np.ndarray,
    confidence: float,
) -> dict[str, Any]:
    """Run one arm/confidence combination and return one metrics row."""
    policy = RejectPolicySpec.flag(ncf=arm.ncf, w=_W)
    errors = np.asarray(bundle.baseline_pred != bundle.y_test, dtype=bool)
    full_accuracy = float(np.mean(bundle.baseline_pred == bundle.y_test))

    if arm.difficulty_normalized:
        bundle.wrapper.explainer.difficulty_estimator = difficulty_estimator
    else:
        bundle.wrapper.explainer.difficulty_estimator = None

    predict_kwargs: dict[str, Any] = {
        "reject_policy": policy,
        "confidence": confidence,
        "strategy": arm.strategy,
    }
    if arm.novelty_penalized:
        predict_kwargs["novelty_estimator"] = novelty_estimator
        predict_kwargs["novelty_weight"] = arm.novelty_weight

    result = bundle.wrapper.predict(bundle.x_test, **predict_kwargs)
    breakdown = breakdown_from_reject_output(result, default_confidence=float(confidence))
    rejected = np.asarray(breakdown["rejected"], dtype=bool)
    accepted = ~rejected
    set_sizes = np.asarray(breakdown["prediction_set_size"], dtype=int)
    prediction_set = breakdown.get("prediction_set")
    metadata = getattr(result, "metadata", {}) or {}

    accepted_acc = accepted_accuracy(bundle.y_test, bundle.baseline_pred, accepted)
    total_errors = int(np.sum(errors))
    rejected_error_capture_rate = (
        float(np.sum(errors & rejected) / total_errors) if total_errors > 0 else float("nan")
    )
    empirical_cov = _empirical_coverage_or_nan(prediction_set, bundle.y_test)
    coverage_gap = empirical_cov - float(confidence) if np.isfinite(empirical_cov) else float("nan")

    return {
        "task_type": "classification",
        "dataset": bundle.dataset_name,
        "confidence": float(confidence),
        "epsilon": float(breakdown["epsilon"]),
        "n_train": int(len(bundle.x_fit)),
        "n_cal": int(len(bundle.x_cal)),
        "n_test": int(len(bundle.x_test)),
        "arm_code": arm.code,
        "arm_label": (
            f"{arm.code}|strategy={arm.strategy}|ncf={arm.ncf}|"
            f"novelty_weight={arm.novelty_weight}"
        ),
        "ncf": arm.ncf,
        "strategy": arm.strategy,
        "difficulty_normalized": bool(
            metadata.get("difficulty_normalized", arm.difficulty_normalized)
        ),
        "novelty_penalized": bool(metadata.get("novelty_penalized", arm.novelty_penalized)),
        "novelty_weight": float(metadata.get("novelty_weight", arm.novelty_weight)),
        "accept_rate": float(np.mean(accepted)),
        "reject_rate": float(breakdown["reject_rate"]),
        "ambiguity_rate": float(breakdown["ambiguity_rate"]),
        "novelty_rate": float(breakdown["novelty_rate"]),
        "accepted_accuracy": accepted_acc,
        "full_accuracy": full_accuracy,
        "accuracy_delta": accepted_acc - full_accuracy if np.isfinite(accepted_acc) else float("nan"),
        "singleton_error_rate": _singleton_error_rate(breakdown),
        "error_rate_defined": bool(breakdown["error_rate_defined"]),
        "rejected_error_capture_rate": rejected_error_capture_rate,
        "mean_difficulty_all": _safe_mean(difficulty_scores),
        "mean_difficulty_accepted": _safe_mean(difficulty_scores[accepted]),
        "mean_difficulty_rejected": _safe_mean(difficulty_scores[rejected]),
        "difficulty_gap_rejected_minus_accepted": (
            _safe_mean(difficulty_scores[rejected]) - _safe_mean(difficulty_scores[accepted])
        ),
        "difficulty_reject_auc": _difficulty_reject_auc(difficulty_scores, rejected),
        "mean_novelty_all": _safe_mean(novelty_scores),
        "mean_novelty_accepted": _safe_mean(novelty_scores[accepted]),
        "mean_novelty_rejected": _safe_mean(novelty_scores[rejected]),
        "novelty_gap_rejected_minus_accepted": (
            _safe_mean(novelty_scores[rejected]) - _safe_mean(novelty_scores[accepted])
        ),
        "novelty_reject_auc": _difficulty_reject_auc(novelty_scores, rejected),
        "empty_rate": _safe_rate(set_sizes == 0),
        "singleton_rate": _safe_rate(set_sizes == 1),
        "multilabel_rate": _safe_rate(set_sizes >= 2),
        "empirical_coverage": empirical_cov,
        "coverage_gap": coverage_gap,
        "coverage_defined": bool(np.isfinite(empirical_cov)),
    }


def run(config: RunConfig) -> None:
    """Run Scenario 10 novelty-aware experimental reject comparison."""
    rows: list[dict[str, Any]] = []
    datasets = task_specs("binary", quick=config.quick) + task_specs("multiclass", quick=config.quick)
    confidences = tuple(float(c) for c in confidence_grid(config.quick))

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
                    rows.append(row)

    df = pd.DataFrame(rows)
    outcome_summary: dict[str, Any] = {
        "rows": int(len(df)),
        "datasets": int(df["dataset"].nunique()) if not df.empty else 0,
        "seeds": int(df["seed"].nunique()) if not df.empty else 0,
        "novelty_weight": _NOVELTY_WEIGHT,
        "mean_accept_rate": float(df["accept_rate"].mean()) if not df.empty else float("nan"),
        "mean_accuracy_delta": float(df["accuracy_delta"].mean()) if not df.empty else float("nan"),
    }

    arm_summary = pd.DataFrame()
    confidence_arm_summary = pd.DataFrame()
    analyses = ["Insufficient data.", "Insufficient data.", "Insufficient data.", "Insufficient data."]

    if not df.empty:
        arm_summary = (
            df.groupby(
                [
                    "arm_code",
                    "strategy",
                    "ncf",
                    "difficulty_normalized",
                    "novelty_penalized",
                    "novelty_weight",
                ]
            )[
                [
                    "accept_rate",
                    "reject_rate",
                    "accepted_accuracy",
                    "accuracy_delta",
                    "ambiguity_rate",
                    "novelty_rate",
                    "empty_rate",
                    "multilabel_rate",
                    "rejected_error_capture_rate",
                    "difficulty_gap_rejected_minus_accepted",
                    "difficulty_reject_auc",
                    "novelty_gap_rejected_minus_accepted",
                    "novelty_reject_auc",
                    "empirical_coverage",
                    "coverage_gap",
                ]
            ]
            .mean(numeric_only=True)
            .reset_index()
            .sort_values("arm_code", kind="mergesort")
        )

        confidence_arm_summary = (
            df.groupby(["confidence", "epsilon", "arm_code", "strategy"])[
                [
                    "reject_rate",
                    "accepted_accuracy",
                    "ambiguity_rate",
                    "novelty_rate",
                    "novelty_gap_rejected_minus_accepted",
                    "novelty_reject_auc",
                ]
            ]
            .mean(numeric_only=True)
            .reset_index()
            .sort_values(["arm_code", "confidence"], kind="mergesort")
        )

        a_row = _arm_row(arm_summary, "A")
        c_row = _arm_row(arm_summary, "C")
        g_row = _arm_row(arm_summary, "G")

        if c_row is not None and g_row is not None:
            novelty_delta = float(g_row["novelty_rate"] - c_row["novelty_rate"])
            empty_delta = float(g_row["empty_rate"] - c_row["empty_rate"])
            ambiguity_delta = float(g_row["ambiguity_rate"] - c_row["ambiguity_rate"])
            multilabel_delta = float(g_row["multilabel_rate"] - c_row["multilabel_rate"])
            accepted_acc_delta = float(g_row["accepted_accuracy"] - c_row["accepted_accuracy"])
            novelty_auc_delta = float(g_row["novelty_reject_auc"] - c_row["novelty_reject_auc"])

            outcome_summary["G_minus_C_novelty_rate_delta"] = novelty_delta
            outcome_summary["G_minus_C_empty_rate_delta"] = empty_delta
            outcome_summary["G_minus_C_ambiguity_rate_delta"] = ambiguity_delta
            outcome_summary["G_minus_C_multilabel_rate_delta"] = multilabel_delta
            outcome_summary["G_minus_C_accepted_accuracy_delta"] = accepted_acc_delta
            outcome_summary["G_minus_C_novelty_reject_auc_delta"] = novelty_auc_delta

            analyses[0] = (
                "G vs C changed novelty_rate by "
                f"{novelty_delta:+.4f}, empty_rate by {empty_delta:+.4f}, "
                f"and novelty_reject_auc by {novelty_auc_delta:+.4f}."
            )
            analyses[1] = (
                "G vs C changed ambiguity_rate by "
                f"{ambiguity_delta:+.4f} and multilabel_rate by {multilabel_delta:+.4f}."
            )
            analyses[2] = (
                "G vs C changed accepted_accuracy by "
                f"{accepted_acc_delta:+.4f}."
            )

        recommendation = "C"
        reason = "difficulty-normalized remains the simpler experimental baseline"
        if c_row is not None and g_row is not None:
            if (
                float(g_row["novelty_rate"] - c_row["novelty_rate"]) > 0.01
                and float(g_row["accepted_accuracy"] - c_row["accepted_accuracy"]) > -0.02
            ):
                recommendation = "G"
                reason = "novelty penalty adds novelty routing without large accepted-accuracy loss"
        elif a_row is not None:
            recommendation = "A"
            reason = "experimental arms unavailable"
        analyses[3] = f"Recommended arm for next iteration: {recommendation} ({reason})."
        outcome_summary["recommended_arm"] = recommendation
        outcome_summary["recommendation_reason"] = reason

    meta = {
        "scenario": "scenario_10_ambiguity_novelty_reject",
        "display_name": "Scenario 10 - Ambiguity-normalized novelty-penalized reject strategy",
        "guarantee_status": "empirical",
        "quick": config.quick,
        "highlights": [
            "Compares built-in default, direct difficulty-normalized, and novelty-aware experimental reject strategies.",
            "Primary contrast is C vs G: direct difficulty normalization with and without novelty penalty.",
            "Novelty estimator is deterministic, fitted on proper-training features only, and uses no calibration labels/residuals.",
            *analyses,
        ],
        "outcome": outcome_summary,
    }

    write_csv_json_md("scenario_10_ambiguity_novelty_reject", df, meta)
    _append_readable_sections(
        "scenario_10_ambiguity_novelty_reject",
        arm_summary,
        confidence_arm_summary,
        analyses,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    arguments = parser.parse_args()
    run(RunConfig(seed=42, quick=arguments.quick))
