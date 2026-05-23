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
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from crepes.extras import DifficultyEstimator as CrepesDifficultyEstimator
from scipy.stats import pearsonr
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

logger = logging.getLogger(__name__)

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

# Four crepes.extras.DifficultyEstimator setups for RT-7 ablation.
# Keyed by a short code; value is a description used in the estimator_type column.
_CREPES_ESTIMATOR_CODES = (
    "de_knn_dist",   # fit(x_fit) — distances to k nearest neighbours
    "de_knn_std",    # fit(x_fit, y=y_fit) — std of target labels of k nearest neighbours
    "de_knn_res",    # fit(x_fit, residuals=...) — mean abs residuals of k nearest neighbours
    "de_rf_var",     # fit(x_fit, learner=model) — variance from RF ensemble predictions
)


def _build_crepes_estimators(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    model: RandomForestClassifier,
) -> dict[str, CrepesDifficultyEstimator]:
    """Fit the four promoted crepes.extras.DifficultyEstimator setups."""
    # For knn_res, use |y - p_correct| as a classification residual proxy.
    proba_fit = model.predict_proba(x_fit)
    p_correct = proba_fit[np.arange(len(y_fit)), y_fit.astype(int)]
    residuals_fit = np.abs(y_fit.astype(float) - p_correct)

    de_knn_dist = CrepesDifficultyEstimator().fit(x_fit)
    de_knn_std = CrepesDifficultyEstimator().fit(x_fit, y=y_fit)
    de_knn_res = CrepesDifficultyEstimator().fit(x_fit, residuals=residuals_fit)
    de_rf_var = CrepesDifficultyEstimator().fit(x_fit, learner=model)

    return {
        "de_knn_dist": de_knn_dist,
        "de_knn_std": de_knn_std,
        "de_knn_res": de_knn_res,
        "de_rf_var": de_rf_var,
    }


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
    metric_consistency_note: dict[str, Any] | None = None,
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
    if metric_consistency_note is not None:
        extra.extend(
            [
                "",
                "## Metric Consistency Note (RT-5)",
                "",
                (
                    "Scenario 9 reports a full-grid A-vs-C difficulty_reject_auc delta "
                    f"of {metric_consistency_note['full_grid_auc_delta']:+.4f}, while "
                    "Scenario 11 reports +0.0155 at matched operating points. This "
                    "reduction is a selection effect, not a contradiction:"
                ),
                "",
                (
                    "- Scenario 9 averages over all confidence values. The positive "
                    "delta is strongest in high-confidence rows "
                    f"(conf >= 0.91), where the AUC delta is "
                    f"{metric_consistency_note['hi_conf_auc_delta']:+.4f}."
                ),
                (
                    "- At moderate confidence (conf < 0.91), the Scenario 9 AUC delta "
                    f"is {metric_consistency_note['lo_conf_auc_delta']:+.4f}: smaller "
                    "than the high-confidence tail, but still positive."
                ),
                (
                    "- Scenario 11 targets reject rates of 10-40%, uses matched "
                    "operating-point selection, and reduces the observed "
                    "difficulty-AUC effect to +0.0155."
                ),
                "",
                (
                    "Conclusion: the full-grid Scenario 9 AUC advantage is not "
                    "sufficient evidence for public promotion. At matched operating "
                    "points, accepted-accuracy gains are tiny or negative and the "
                    "difficulty-selection advantage is much smaller."
                ),
            ]
        )
    md_path.write_text(content + "\n" + "\n".join(extra), encoding="utf-8")


def _diagnose_db_paradox(
    config: RunConfig,
    *,
    dataset_name: str = "breast_cancer",
    seed_offset: int = 0,
    confidence: float = 0.90,
) -> dict[str, Any]:
    """Diagnose the D-B reject-rate paradox for a single dataset.

    Arm B uses VA difficulty + builtin strategy. Arm D adds direct score
    normalization on top of VA. D has lower reject rate than B despite adding
    more difficulty signal — this function logs the score distributions and
    VA-vs-difficulty correlation that explain why.
    """
    from .common_reject import load_dataset, split_dataset

    # Find the requested dataset spec
    all_specs = (
        task_specs("binary", quick=False) + task_specs("multiclass", quick=False)
    )
    spec = next((s for s in all_specs if s.name == dataset_name), None)
    if spec is None:
        logger.warning("diagnose-db: dataset %s not found; skipping", dataset_name)
        return {}

    _, x_all, y_all, feature_names = load_dataset(spec)
    seed = config.seed + seed_offset
    x_fit, x_cal, x_test, y_fit, y_cal, y_test = split_dataset(
        x_all, y_all, seed=seed, stratify=True
    )
    difficulty_estimator = DeterministicDifficultyEstimator.fit(x_fit)
    difficulty_test = np.asarray(difficulty_estimator.apply(x_test), dtype=float)

    # Arm B: VA difficulty + builtin strategy
    from calibrated_explanations import RejectPolicySpec, WrapCalibratedExplainer

    model = RandomForestClassifier(n_estimators=60, random_state=seed, max_depth=8, n_jobs=1)
    wrapper_b = WrapCalibratedExplainer(model)
    wrapper_b.fit(x_fit, y_fit)
    wrapper_b.calibrate(
        x_cal, y_cal, feature_names=list(feature_names),
        difficulty_estimator=difficulty_estimator,
    )
    wrapper_b.set_difficulty_estimator(difficulty_estimator, initialize=False)

    # Arm D: VA difficulty + experimental strategy
    wrapper_d = WrapCalibratedExplainer(model)
    wrapper_d.fit(x_fit, y_fit)
    wrapper_d.calibrate(
        x_cal, y_cal, feature_names=list(feature_names),
        difficulty_estimator=difficulty_estimator,
    )
    wrapper_d.set_difficulty_estimator(difficulty_estimator, initialize=False)

    policy = RejectPolicySpec.flag(ncf="default", w=_W)
    res_b = wrapper_b.predict(x_test, reject_policy=policy, confidence=confidence, strategy="builtin.default")
    res_d = wrapper_d.predict(x_test, reject_policy=policy, confidence=confidence, strategy="experimental.difficulty_normalized")

    bd_b = breakdown_from_reject_output(res_b, default_confidence=confidence)
    bd_d = breakdown_from_reject_output(res_d, default_confidence=confidence)

    # Extract NCF scores from metadata
    meta_b = getattr(res_b, "metadata", {}) or {}
    meta_d = getattr(res_d, "metadata", {}) or {}

    # Compute raw VA-scaled proba for arm B (post-calibration)
    va_proba_b = wrapper_b.explainer.interval_learner.predict_proba(x_test)
    if isinstance(va_proba_b, tuple):
        va_proba_b = va_proba_b[0]
    va_proba_b = np.asarray(va_proba_b, dtype=float)
    p_max_b = np.max(va_proba_b, axis=1)

    # Pearson correlation: VA p_max vs difficulty
    r_val, p_val = pearsonr(p_max_b, difficulty_test)

    rr_b = float(bd_b["reject_rate"])
    rr_d = float(bd_d["reject_rate"])

    diagnostics: dict[str, Any] = {
        "dataset": dataset_name,
        "seed": seed,
        "confidence": confidence,
        "arm_B_reject_rate": rr_b,
        "arm_D_reject_rate": rr_d,
        "D_minus_B_reject_rate": rr_d - rr_b,
        "difficulty_mean": float(np.mean(difficulty_test)),
        "difficulty_std": float(np.std(difficulty_test)),
        "difficulty_min": float(np.min(difficulty_test)),
        "difficulty_max": float(np.max(difficulty_test)),
        "va_p_max_mean": float(np.mean(p_max_b)),
        "va_p_max_std": float(np.std(p_max_b)),
        "pearson_r_pmax_vs_difficulty": float(r_val),
        "pearson_p_value": float(p_val),
        "arm_B_meta": {k: v for k, v in meta_b.items() if isinstance(v, (int, float, str, bool))},
        "arm_D_meta": {k: v for k, v in meta_d.items() if isinstance(v, (int, float, str, bool))},
    }

    print("\n=== D-B Paradox Diagnostics ===")
    print(f"Dataset: {dataset_name}, seed={seed}, confidence={confidence}")
    print(f"Arm B reject_rate: {rr_b:.4f}   Arm D reject_rate: {rr_d:.4f}   D-B: {rr_d-rr_b:+.4f}")
    print(f"Difficulty: mean={diagnostics['difficulty_mean']:.4f}, std={diagnostics['difficulty_std']:.4f}, "
          f"range=[{diagnostics['difficulty_min']:.4f},{diagnostics['difficulty_max']:.4f}]")
    print(f"VA p_max: mean={diagnostics['va_p_max_mean']:.4f}, std={diagnostics['va_p_max_std']:.4f}")
    print(f"Pearson r(VA_p_max, difficulty)={r_val:.4f} (p={p_val:.4f})")
    if abs(r_val) < 0.1:
        print("  -> VA probability and difficulty are nearly uncorrelated: normalization does")
        print("     not reorder instances, just rescales scores uniformly, reducing ambiguity rate.")
    print("=" * 40)

    logger.info(
        "diagnose-db %s: B_rr=%.4f D_rr=%.4f r(p_max,diff)=%.4f",
        dataset_name, rr_b, rr_d, r_val,
    )
    return diagnostics


def run(config: RunConfig, *, diagnose_db: bool = False, crepes_ablation: bool = False) -> None:
    """Run Scenario 9 difficulty-normalized reject strategy ablation.

    Parameters
    ----------
    config:
        Run configuration.
    diagnose_db:
        If True, run the D-B paradox diagnostic before the main loop.
    crepes_ablation:
        If True, additionally run arm A and arm C with each of the four
        ``crepes.extras.DifficultyEstimator`` setups (RT-7). Results are
        appended with an ``estimator_type`` column; the deterministic-estimator
        rows retain ``estimator_type="deterministic"``.
    """
    if diagnose_db:
        _diagnose_db_paradox(config)

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
                if arm.use_va_difficulty or arm.difficulty_normalized:
                    bundle.wrapper.set_difficulty_estimator(difficulty_estimator, initialize=False)
                else:
                    bundle.wrapper.set_difficulty_estimator(None, initialize=False)

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
                            "estimator_type": "deterministic",
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

            # RT-7: crepes.extras.DifficultyEstimator ablation (opt-in via --crepes-ablation)
            if crepes_ablation and spec.task_type == "binary":
                bundle_no_va = bundles_by_va[False]
                model_obj = bundle_no_va.wrapper.explainer.interval_learner
                base_model = getattr(model_obj, "learner", None)
                if base_model is None:
                    continue
                crepes_estimators = _build_crepes_estimators(
                    bundle_no_va.x_fit, bundle_no_va.y_fit, base_model
                )
                errors_nova = np.asarray(
                    bundle_no_va.baseline_pred != bundle_no_va.y_test, dtype=bool
                )
                full_acc_nova = float(np.mean(bundle_no_va.baseline_pred == bundle_no_va.y_test))

                for est_code, crepes_est in crepes_estimators.items():
                    diff_scores_crepes = np.asarray(
                        crepes_est.apply(bundle_no_va.x_test), dtype=float
                    )
                    for arm_code_crepes, strategy_crepes in [
                        ("A_crepes", "builtin.default"),
                        ("C_crepes", "experimental.difficulty_normalized"),
                    ]:
                        bundle_no_va.wrapper.set_difficulty_estimator(
                            crepes_est if arm_code_crepes == "C_crepes" else None,
                            initialize=False,
                        )
                        policy_crepes = RejectPolicySpec.flag(ncf="default", w=_W)
                        for confidence in confidences:
                            result_c = bundle_no_va.wrapper.predict(
                                bundle_no_va.x_test,
                                reject_policy=policy_crepes,
                                confidence=confidence,
                                strategy=strategy_crepes,
                            )
                            bd_c = breakdown_from_reject_output(
                                result_c, default_confidence=float(confidence)
                            )
                            rejected_c = np.asarray(bd_c["rejected"], dtype=bool)
                            accepted_c = ~rejected_c
                            set_sizes_c = np.asarray(bd_c["prediction_set_size"], dtype=int)
                            pred_set_c = bd_c.get("prediction_set")
                            acc_c = accepted_accuracy(
                                bundle_no_va.y_test, bundle_no_va.baseline_pred, accepted_c
                            )
                            rej_err_c = (
                                float(np.sum(errors_nova & rejected_c) / np.sum(errors_nova))
                                if np.sum(errors_nova) > 0
                                else float("nan")
                            )
                            emp_cov_c = _empirical_coverage_or_nan(pred_set_c, bundle_no_va.y_test)
                            rows.append(
                                {
                                    "task_type": spec.task_type,
                                    "dataset": spec.name,
                                    "seed": seed,
                                    "confidence": float(confidence),
                                    "epsilon": float(bd_c["epsilon"]),
                                    "n_train": int(len(bundle_no_va.x_fit)),
                                    "n_cal": int(len(bundle_no_va.x_cal)),
                                    "n_test": int(len(bundle_no_va.x_test)),
                                    "arm_code": arm_code_crepes,
                                    "arm_label": f"{arm_code_crepes}|{est_code}|strategy={strategy_crepes}",
                                    "estimator_type": est_code,
                                    "ncf": "default",
                                    "strategy": strategy_crepes,
                                    "use_va_difficulty": False,
                                    "difficulty_normalized": arm_code_crepes == "C_crepes",
                                    "double_count_difficulty": False,
                                    "accept_rate": float(np.mean(accepted_c)),
                                    "reject_rate": float(bd_c["reject_rate"]),
                                    "ambiguity_rate": float(bd_c["ambiguity_rate"]),
                                    "novelty_rate": float(bd_c["novelty_rate"]),
                                    "accepted_accuracy": acc_c,
                                    "full_accuracy": full_acc_nova,
                                    "accuracy_delta": (
                                        acc_c - full_acc_nova if np.isfinite(acc_c) else float("nan")
                                    ),
                                    "singleton_error_rate": _singleton_error_rate(bd_c),
                                    "error_rate_defined": bool(bd_c["error_rate_defined"]),
                                    "rejected_error_capture_rate": rej_err_c,
                                    "mean_difficulty_all": _safe_mean(diff_scores_crepes),
                                    "mean_difficulty_accepted": _safe_mean(diff_scores_crepes[accepted_c]),
                                    "mean_difficulty_rejected": _safe_mean(diff_scores_crepes[rejected_c]),
                                    "difficulty_gap_rejected_minus_accepted": (
                                        _safe_mean(diff_scores_crepes[rejected_c])
                                        - _safe_mean(diff_scores_crepes[accepted_c])
                                    ),
                                    "difficulty_reject_auc": _difficulty_reject_auc(
                                        diff_scores_crepes, rejected_c
                                    ),
                                    "empty_rate": _safe_rate(set_sizes_c == 0),
                                    "singleton_rate": _safe_rate(set_sizes_c == 1),
                                    "multilabel_rate": _safe_rate(set_sizes_c >= 2),
                                    "empirical_coverage": emp_cov_c,
                                    "coverage_gap": (
                                        emp_cov_c - float(confidence)
                                        if np.isfinite(emp_cov_c)
                                        else float("nan")
                                    ),
                                    "coverage_defined": bool(np.isfinite(emp_cov_c)),
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

        def _a_vs_c_auc_delta(subset: pd.DataFrame) -> float:
            means = subset[subset["arm_code"].isin(["A", "C"])].groupby("arm_code")[
                "difficulty_reject_auc"
            ].mean()
            if "A" not in means or "C" not in means:
                return float("nan")
            return float(means["C"] - means["A"])

        metric_consistency_note = {
            "full_grid_auc_delta": float(outcome_summary.get("A_vs_C_difficulty_reject_auc_delta", float("nan"))),
            "hi_conf_auc_delta": _a_vs_c_auc_delta(df[df["confidence"] >= 0.91]),
            "lo_conf_auc_delta": _a_vs_c_auc_delta(df[df["confidence"] < 0.91]),
            "scenario_11_matched_delta": 0.0155,
            "note": (
                "Full-grid positive delta is strongest in high-confidence rows. "
                "Scenario 11 matched operating-point selection reduces the observed "
                "difficulty-AUC effect and remains the promotion decision gate."
            ),
        }
        outcome_summary["metric_consistency_note"] = metric_consistency_note
    else:
        metric_consistency_note = None

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
        metric_consistency_note,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--diagnose-db",
        action="store_true",
        help="Run D-B paradox diagnostic (score distribution and VA-difficulty correlation) then proceed normally.",
    )
    parser.add_argument(
        "--crepes-ablation",
        action="store_true",
        help="RT-7: additionally run arm A and C with all four crepes.extras.DifficultyEstimator setups.",
    )
    arguments = parser.parse_args()
    run(
        RunConfig(seed=42, quick=arguments.quick),
        diagnose_db=arguments.diagnose_db,
        crepes_ablation=arguments.crepes_ablation,
    )
