"""Scenario A2: representative-level constraint evaluation for guarded factual rules.

Scientific question
-------------------
Do guard p-values improve factual explanations by removing representative
perturbations that violate a known data-generating constraint?

Option A semantics
------------------
Guarded factual explanations filter representative perturbation candidates. They
DO NOT certify whole emitted intervals. This evaluation is representative-level
only: all metrics are computed on representative perturbation values, and no
interval boundary or interior probing is used in the primary evaluation.

Known synthetic constraint
--------------------------
x1 <= 2 * x0 + 3

Required paper-facing metrics
-----------------------------
- representative_violation_rate
- candidate_violation_auroc
- rule_count
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from common_guarded import (
    ProgressTracker,
    dataframe_to_markdown,
    write_progress_snapshot,
    write_report,
)
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from calibrated_explanations.ce_agent_utils import ensure_ce_first_wrapper, fit_and_calibrate

DEFAULT_SIGNIFICANCE: Tuple[float, ...] = (0.01, 0.05, 0.1, 0.2)
DEFAULT_N_NEIGHBORS = 5
DEFAULT_MERGE_ADJACENT = False
DEFAULT_NORMALIZE_GUARD = True
DEFAULT_N_DIMS: Tuple[int, ...] = (2, 10)
CONSTRAINED_FEATURES = frozenset({0, 1})
MODELS = ("logreg", "rf")


@dataclass(frozen=True)
class MethodBundle:
    """Container for one-instance rule and candidate records by method."""

    standard_rows: List[Dict[str, Any]]
    multibin_rows: List[Dict[str, Any]]
    guarded_rows_by_significance: Dict[float, List[Dict[str, Any]]]
    multibin_rows_by_significance: Dict[float, List[Dict[str, Any]]]
    candidate_rows_by_method_and_significance: Dict[Tuple[str, float], List[Dict[str, Any]]]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for Scenario A2."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "artifacts" / "guarded" / "scenario_a2_representative_constraint",
    )
    parser.add_argument("--num-seeds", type=int, default=8)
    parser.add_argument("--sample-test", type=int, default=160)
    parser.add_argument("--n-train", type=int, default=3500)
    parser.add_argument("--n-cal", type=int, default=1800)
    parser.add_argument("--n-test", type=int, default=1500)
    parser.add_argument(
        "--paper-focused",
        action="store_true",
        help="Restrict to paper-facing defaults while keeping the same metric contract.",
    )
    parser.add_argument(
        "--large",
        action="store_true",
        help="Paper-focused large profile for stronger synthetic evidence.",
    )
    parser.add_argument("--quick", action="store_true", help="Small smoke-test profile.")
    return parser.parse_args()


def scenario_constraint(x: np.ndarray) -> np.ndarray:
    """Return mask for x1 <= 2*x0 + 3."""
    return x[:, 1] <= (2.0 * x[:, 0] + 3.0)


def generate_scenario_a2(n: int, seed: int, n_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data satisfying the known structural constraint.

    n_dim=2 gives the geometric sanity-check profile.
    n_dim=10 gives constrained features + correlated support + nonlinear support
    + additional predictive features + noise.
    """
    if n_dim not in DEFAULT_N_DIMS:
        raise ValueError(f"Unsupported n_dim={n_dim}; expected one of {DEFAULT_N_DIMS}")

    rng = np.random.default_rng(seed)
    accepted: List[np.ndarray] = []
    cov = np.array([[1.0, 0.9], [0.9, 1.4]])
    while sum(chunk.shape[0] for chunk in accepted) < n:
        candidate = rng.multivariate_normal(mean=np.array([0.0, 0.0]), cov=cov, size=max(2048, n))
        mask = scenario_constraint(candidate)
        if np.any(mask):
            accepted.append(candidate[mask])
    x_info = np.vstack(accepted)[:n]

    if n_dim == 2:
        x = x_info
    else:
        # 8 extra features for n_dim=10 profile.
        x2 = 0.70 * x_info[:, 0] + 0.20 * x_info[:, 1] + rng.normal(0.0, 0.30, size=n)
        x3 = -0.25 * x_info[:, 0] + 0.60 * x_info[:, 1] + rng.normal(0.0, 0.30, size=n)
        x4 = np.sin(x_info[:, 0]) + rng.normal(0.0, 0.20, size=n)
        x5 = np.tanh(x_info[:, 1]) + rng.normal(0.0, 0.20, size=n)
        x6 = 0.40 * x_info[:, 0] * x_info[:, 1] + rng.normal(0.0, 0.35, size=n)
        x7 = 0.55 * np.cos(x_info[:, 0]) + rng.normal(0.0, 0.25, size=n)
        x8 = rng.standard_normal(n)
        x9 = rng.standard_normal(n)
        x = np.column_stack([x_info[:, 0], x_info[:, 1], x2, x3, x4, x5, x6, x7, x8, x9])

    logits = (
        0.80 * x[:, 0]
        + 0.55 * x[:, 1]
        + (0.35 * x[:, 2] if n_dim >= 3 else 0.0)
        - (0.30 * x[:, 3] if n_dim >= 4 else 0.0)
        + (0.45 * x[:, 4] if n_dim >= 5 else 0.0)
        + (0.35 * x[:, 6] if n_dim >= 7 else 0.0)
        + rng.normal(0.0, 0.70, size=n)
    )
    y = (logits > np.median(logits)).astype(int)
    return x, y


def build_splits(seed: int, n_dim: int, n_train: int, n_cal: int, n_test: int) -> Tuple[np.ndarray, ...]:
    """Build train/cal/test splits for one seed and dimensionality profile."""
    x_train, y_train = generate_scenario_a2(n_train, seed * 1000 + n_dim * 11 + 1, n_dim)
    x_cal, y_cal = generate_scenario_a2(n_cal, seed * 1000 + n_dim * 11 + 3, n_dim)
    x_test, y_test = generate_scenario_a2(n_test, seed * 1000 + n_dim * 11 + 5, n_dim)
    return x_train, y_train, x_cal, y_cal, x_test, y_test


def get_models(seed: int) -> Dict[str, Any]:
    """Return model factory map for Scenario A2 runs."""
    return {
        "logreg": LogisticRegression(max_iter=1200, random_state=seed),
        "rf": RandomForestClassifier(n_estimators=280, max_depth=9, random_state=seed, n_jobs=1),
    }


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float, np.number)):
        return float(value)
    arr = np.asarray(value).reshape(-1)
    if arr.size == 0:
        return None
    return float(arr[-1])


def _violates_constraint_with_feature_replacement(
    x_instance: np.ndarray,
    feature: int,
    representative: Any,
) -> Optional[bool]:
    """Evaluate violation after replacing only one feature by representative value."""
    rep = _safe_float(representative)
    if rep is None:
        return None
    candidate = np.array(x_instance, copy=True)
    candidate[feature] = rep
    return not bool(scenario_constraint(candidate.reshape(1, -1))[0])


def _extract_standard_rows(
    explanation: Any,
    *,
    seed: int,
    model_name: str,
    n_dim: int,
    instance_id: int,
) -> List[Dict[str, Any]]:
    """Extract emitted factual rows from explain_factual output."""
    rules = explanation.get_rules()
    rows: List[Dict[str, Any]] = []
    rule_list = rules.get("rule", [])
    features = rules.get("feature", [])
    sampled_values = rules.get("sampled_values", [])
    for idx, _ in enumerate(rule_list):
        if idx >= len(features):
            continue
        feature = int(features[idx])
        representative = _safe_float(sampled_values[idx] if idx < len(sampled_values) else None)
        rows.append(
            {
                "seed": seed,
                "model": model_name,
                "n_dim": n_dim,
                "method": "standard",
                "instance_id": int(instance_id),
                "rule_id": f"r{idx}",
                "feature": feature,
                "representative": representative,
                "is_constrained_feature": bool(feature in CONSTRAINED_FEATURES),
                "p_value": np.nan,
                "emitted": True,
            }
        )
    return rows


def _extract_audit_rows(
    audit_instance: Dict[str, Any],
    *,
    seed: int,
    model_name: str,
    n_dim: int,
    instance_id: int,
    method: str,
    significance: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Extract emitted and candidate representative rows from guarded audit."""
    emitted_rows: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []
    intervals = audit_instance.get("intervals", [])

    for idx, rec in enumerate(intervals):
        feature = int(rec.get("feature", -1))
        representative = _safe_float(rec.get("representative"))
        p_value = _safe_float(rec.get("p_value"))
        emitted = bool(rec.get("emitted", False))
        is_constrained = feature in CONSTRAINED_FEATURES

        row_common = {
            "seed": seed,
            "model": model_name,
            "n_dim": n_dim,
            "method": method,
            "significance": float(significance),
            "instance_id": int(instance_id),
            "rule_id": f"r{idx}",
            "feature": feature,
            "representative": representative,
            "is_constrained_feature": bool(is_constrained),
            "p_value": p_value,
            "emitted": emitted,
        }

        candidate_rows.append(dict(row_common))
        if emitted:
            emitted_rows.append(dict(row_common))
    return emitted_rows, candidate_rows


def _auroc_from_candidates(frame: pd.DataFrame) -> float:
    """Compute candidate-level AUROC using 1-p as anomaly score."""
    if frame.empty:
        return float("nan")
    y = frame["violates_constraint"].astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        return float("nan")
    scores = 1.0 - frame["p_value"].astype(float).to_numpy()
    return float(roc_auc_score(y, scores))


def _ci95(values: Sequence[float]) -> Tuple[float, float, float]:
    """Return mean and normal-approximation 95% CI for finite values."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(arr))
    if arr.size == 1:
        return mean, mean, mean
    se = float(np.std(arr, ddof=1) / np.sqrt(arr.size))
    delta = 1.96 * se
    return mean, mean - delta, mean + delta


def _paired_wilcoxon(
    summary_df: pd.DataFrame,
    metric: str,
    left: str,
    right: str,
) -> Tuple[float, float, int]:
    """Paired Wilcoxon over seed-model-n_dim-significance aggregates."""
    metric_df = summary_df[summary_df["metric"] == metric].copy()
    pivot = metric_df.pivot_table(
        index=["seed", "model", "n_dim", "significance"],
        columns="method",
        values="value",
        aggfunc="mean",
    ).dropna(subset=[left, right])
    if pivot.empty:
        return float("nan"), float("nan"), 0
    diffs = pivot[left] - pivot[right]
    if np.allclose(diffs.to_numpy(), 0.0):
        return 1.0, 0.0, int(len(diffs))
    stat = wilcoxon(pivot[left], pivot[right], zero_method="wilcox", alternative="two-sided")
    return float(stat.pvalue), float(np.median(diffs)), int(len(diffs))


def _deterministic_group_seed(seed: int, model_name: str, n_dim: int, significance: float) -> int:
    """Stable group-level seed for random pruning."""
    model_code = 17 if model_name == "logreg" else 29
    sig_code = int(round(significance * 10_000))
    return int(seed * 1_000_003 + model_code * 10_007 + n_dim * 997 + sig_code * 31)


def _build_method_bundle_for_instance(
    wrapper: Any,
    x_instance: np.ndarray,
    *,
    seed: int,
    model_name: str,
    n_dim: int,
    instance_id: int,
    significance_grid: Sequence[float],
    multibin_noguard_sig: float,
) -> MethodBundle:
    """Run all method calls for one instance and return extracted rows."""
    std_expl = wrapper.explain_factual(x_instance)
    standard_rows = _extract_standard_rows(
        std_expl[0],
        seed=seed,
        model_name=model_name,
        n_dim=n_dim,
        instance_id=instance_id,
    )

    mb_expl = wrapper.explain_factual(
        x_instance,
        guarded=True,
        significance=multibin_noguard_sig,
        n_neighbors=DEFAULT_N_NEIGHBORS,
        merge_adjacent=DEFAULT_MERGE_ADJACENT,
        normalize_guard=DEFAULT_NORMALIZE_GUARD,
    )
    mb_audit_instance = mb_expl.get_guarded_audit()["instances"][0]

    multibin_rows_by_significance: Dict[float, List[Dict[str, Any]]] = {}
    guarded_rows_by_significance: Dict[float, List[Dict[str, Any]]] = {}
    candidate_rows_by_method_and_significance: Dict[Tuple[str, float], List[Dict[str, Any]]] = {}

    mb_emitted_template, mb_candidate_template = _extract_audit_rows(
        mb_audit_instance,
        seed=seed,
        model_name=model_name,
        n_dim=n_dim,
        instance_id=instance_id,
        method="multibin_noguard",
        significance=multibin_noguard_sig,
    )

    for significance in significance_grid:
        # No-guard baseline is held fixed and re-labeled onto each significance level
        # so paired comparisons isolate the guard filter rather than rule-format change.
        mb_emitted = [dict(row, significance=float(significance)) for row in mb_emitted_template]
        mb_candidates = [dict(row, significance=float(significance)) for row in mb_candidate_template]
        multibin_rows_by_significance[float(significance)] = mb_emitted
        candidate_rows_by_method_and_significance[("multibin_noguard", float(significance))] = mb_candidates

        guarded_expl = wrapper.explain_factual(
            x_instance,
            guarded=True,
            significance=float(significance),
            n_neighbors=DEFAULT_N_NEIGHBORS,
            merge_adjacent=DEFAULT_MERGE_ADJACENT,
            normalize_guard=DEFAULT_NORMALIZE_GUARD,
        )
        guarded_audit_instance = guarded_expl.get_guarded_audit()["instances"][0]
        guarded_emitted, guarded_candidates = _extract_audit_rows(
            guarded_audit_instance,
            seed=seed,
            model_name=model_name,
            n_dim=n_dim,
            instance_id=instance_id,
            method="guarded",
            significance=float(significance),
        )
        guarded_rows_by_significance[float(significance)] = guarded_emitted
        candidate_rows_by_method_and_significance[("guarded", float(significance))] = guarded_candidates

    return MethodBundle(
        standard_rows=standard_rows,
        multibin_rows=[dict(row) for row in mb_emitted_template],
        guarded_rows_by_significance=guarded_rows_by_significance,
        multibin_rows_by_significance=multibin_rows_by_significance,
        candidate_rows_by_method_and_significance=candidate_rows_by_method_and_significance,
    )


def _validate_summary_contract(summary_df: pd.DataFrame) -> None:
    """Fail fast if headline metric contract is violated."""
    expected_metrics = {"representative_violation_rate", "candidate_violation_auroc", "rule_count"}
    found_metrics = set(summary_df["metric"].dropna().unique().tolist())
    if found_metrics != expected_metrics:
        raise RuntimeError(
            "summary_metrics.csv metric mismatch: "
            f"expected {sorted(expected_metrics)}, found {sorted(found_metrics)}"
        )


def _validate_required_methods(summary_df: pd.DataFrame) -> None:
    """Fail fast if required comparison methods are absent from summary."""
    required_methods = {"guarded", "multibin_noguard", "random_pruned_multibin"}
    found_methods = set(summary_df["method"].dropna().unique().tolist())
    missing = sorted(required_methods - found_methods)
    if missing:
        raise RuntimeError(f"Missing required method(s) in summary_metrics.csv: {missing}")


def _validate_setup_consistency(run_cfg: Dict[str, Any], summary_df: pd.DataFrame) -> None:
    """Ensure summary dimensions/significance/models/seeds agree with run config."""
    cfg_models = set(run_cfg["models"])
    cfg_dims = set(run_cfg["n_dims"])
    cfg_sigs = set(run_cfg["significance"])
    cfg_seed_count = int(run_cfg["num_seeds"])

    sum_models = set(summary_df["model"].dropna().unique().tolist())
    sum_dims = set(int(v) for v in summary_df["n_dim"].dropna().unique().tolist())
    sum_sigs = set(float(v) for v in summary_df["significance"].dropna().unique().tolist())
    sum_seed_count = int(summary_df["seed"].nunique())

    if cfg_models != sum_models:
        raise RuntimeError(f"run_config/models mismatch: {sorted(cfg_models)} vs {sorted(sum_models)}")
    if cfg_dims != sum_dims:
        raise RuntimeError(f"run_config/n_dims mismatch: {sorted(cfg_dims)} vs {sorted(sum_dims)}")
    if cfg_sigs != sum_sigs:
        raise RuntimeError(f"run_config/significance mismatch: {sorted(cfg_sigs)} vs {sorted(sum_sigs)}")
    if cfg_seed_count != sum_seed_count:
        raise RuntimeError(f"run_config/num_seeds mismatch: {cfg_seed_count} vs {sum_seed_count}")


def _build_report_sections(
    *,
    run_cfg: Dict[str, Any],
    summary_df: pd.DataFrame,
) -> List[Tuple[str, str]]:
    """Build markdown report sections for the required paper narrative."""
    metric_summary_rows: List[Dict[str, Any]] = []
    for metric_name in ["representative_violation_rate", "candidate_violation_auroc", "rule_count"]:
        metric_slice = summary_df[summary_df["metric"] == metric_name]
        for method in sorted(metric_slice["method"].dropna().unique().tolist()):
            values = metric_slice[metric_slice["method"] == method]["value"].to_numpy(dtype=float)
            mean, lo, hi = _ci95(values)
            metric_summary_rows.append(
                {
                    "metric": metric_name,
                    "method": method,
                    "mean": mean,
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "n_pairs": int(np.isfinite(values).sum()),
                }
            )
    metric_summary_df = pd.DataFrame(metric_summary_rows)

    p_g_mb, med_g_mb, n_g_mb = _paired_wilcoxon(
        summary_df,
        metric="representative_violation_rate",
        left="guarded",
        right="multibin_noguard",
    )
    p_g_rp, med_g_rp, n_g_rp = _paired_wilcoxon(
        summary_df,
        metric="representative_violation_rate",
        left="guarded",
        right="random_pruned_multibin",
    )

    comparisons_df = pd.DataFrame(
        [
            {
                "comparison": "guarded vs multibin_noguard",
                "metric": "representative_violation_rate",
                "wilcoxon_p_value": p_g_mb,
                "median_difference_guarded_minus_baseline": med_g_mb,
                "paired_seed_model_n": n_g_mb,
            },
            {
                "comparison": "guarded vs random_pruned_multibin",
                "metric": "representative_violation_rate",
                "wilcoxon_p_value": p_g_rp,
                "median_difference_guarded_minus_baseline": med_g_rp,
                "paired_seed_model_n": n_g_rp,
            },
        ]
    )

    setup_text = "\n".join(
        [
            f"- Seeds: {run_cfg['num_seeds']}",
            f"- Models: {', '.join(run_cfg['models'])}",
            f"- Dimensionality profiles: {run_cfg['n_dims']}",
            f"- Train/Calibration/Test sizes: {run_cfg['n_train']}/{run_cfg['n_cal']}/{run_cfg['n_test']}",
            f"- Sampled test instances per seed-model-dim: {run_cfg['sample_test']}",
            (
                "- Guard settings: "
                f"normalize_guard={run_cfg['guard_settings']['normalize_guard']}, "
                f"n_neighbors={run_cfg['guard_settings']['n_neighbors']}, "
                f"merge_adjacent={run_cfg['guard_settings']['merge_adjacent']}, "
                f"significance={run_cfg['significance']}"
            ),
            "- Methods: standard, multibin_noguard, guarded, random_pruned_multibin",
        ]
    )

    sections: List[Tuple[str, str]] = [
        (
            "Scientific question",
            "Do guard p-values improve factual explanations by removing representative perturbations "
            "that violate a known structural data-generating constraint (x1 <= 2*x0 + 3)?",
        ),
        (
            "Semantics statement",
            "This evaluation is representative-level only. We score only representative perturbation "
            "values from emitted/candidate records. We do not probe interval boundaries or interiors, "
            "and we do not claim whole-interval validity.",
        ),
        (
            "Setup",
            setup_text,
        ),
        (
            "Primary result: representative violation rate",
            (
                "Summary over seed-model-level aggregates (mean and 95% CI):\n\n"
                f"{dataframe_to_markdown(metric_summary_df[metric_summary_df['metric'] == 'representative_violation_rate'], index=False)}\n\n"
                "Paired Wilcoxon tests on seed-model-n_dim-significance aggregates:\n\n"
                f"{dataframe_to_markdown(comparisons_df, index=False)}\n\n"
                "Interpretation target: guarded should be lower than both multibin_noguard and "
                "random_pruned_multibin if improvement is not merely discretizer change or rule-count reduction."
            ),
        ),
        (
            "Mechanism result: candidate violation AUROC",
            (
                "Candidate-level AUROC uses 1 - p_value as anomaly score and labels each constrained-feature "
                "candidate by representative-level constraint violation.\n\n"
                f"{dataframe_to_markdown(metric_summary_df[metric_summary_df['metric'] == 'candidate_violation_auroc'], index=False)}\n\n"
                "This tests alignment between guard score ranking and representative-level structural violations."
            ),
        ),
        (
            "Cost: rule count",
            (
                "rule_count is the mean number of emitted factual rules per instance.\n\n"
                f"{dataframe_to_markdown(metric_summary_df[metric_summary_df['metric'] == 'rule_count'], index=False)}"
            ),
        ),
        (
            "Interpretation with Scenario B",
            (
                "Scenario B shows that guard p-values carry distributional signal under controlled shift. "
                "Scenario A2 complements this by showing that the same p-values, when used inside factual "
                "explanation generation, preferentially remove representative perturbations that violate a known "
                "structural constraint.\n\n"
                "Acceptable claim: Under representative-level semantics, guarded factual explanations reduce "
                "constraint-violating representative perturbations relative to an unguarded multibin baseline, "
                "and the guard p-values rank violating representatives as less conforming."
            ),
        ),
        (
            "Limitations",
            (
                "This study evaluates representative perturbation points only and does not certify full emitted "
                "interval conditions. It focuses on one known synthetic structural constraint and two model families. "
                "Results should not be generalized to arbitrary domain constraints without task-specific validation."
            ),
        ),
    ]
    return sections


def _validate_report_content(report_text: str) -> None:
    """Reject forbidden whole-interval claims and enforce required semantics text."""
    required_phrases = [
        "representative-level",
        "We do not probe interval boundaries or interiors",
        "Scenario B",
    ]
    for phrase in required_phrases:
        if phrase not in report_text:
            raise RuntimeError(f"report.md missing required phrase: {phrase}")

    forbidden_substrings = [
        "guarded intervals are safe",
        "emitted interval conditions are certified",
        "guard enforces arbitrary domain constraints",
        "Fisher-combined p-values are conformal p-values",
        "representative-level validity implies whole-rule validity",
        "whole-interval safety",
    ]
    lowered = report_text.lower()
    for phrase in forbidden_substrings:
        if phrase.lower() in lowered:
            raise RuntimeError(f"report.md contains forbidden claim: {phrase}")


def main() -> None:
    args = parse_args()

    if args.large:
        args.paper_focused = True
        args.num_seeds = max(args.num_seeds, 12)
        args.sample_test = max(args.sample_test, 500)
        args.n_train = max(args.n_train, 5500)
        args.n_cal = max(args.n_cal, 3200)
        args.n_test = max(args.n_test, 2600)

    if args.quick:
        args.num_seeds = min(args.num_seeds, 2)
        args.sample_test = min(args.sample_test, 40)
        args.n_train = min(args.n_train, 700)
        args.n_cal = min(args.n_cal, 450)
        args.n_test = min(args.n_test, 260)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    run_cfg: Dict[str, Any] = {
        "scenario": "A2",
        "date": time.strftime("%Y-%m-%d"),
        "num_seeds": int(args.num_seeds),
        "sample_test": int(args.sample_test),
        "n_train": int(args.n_train),
        "n_cal": int(args.n_cal),
        "n_test": int(args.n_test),
        "n_dims": list(DEFAULT_N_DIMS),
        "models": list(MODELS),
        "methods": ["standard", "multibin_noguard", "guarded", "random_pruned_multibin"],
        "significance": list(DEFAULT_SIGNIFICANCE),
        "guard_settings": {
            "normalize_guard": DEFAULT_NORMALIZE_GUARD,
            "n_neighbors": DEFAULT_N_NEIGHBORS,
            "merge_adjacent": DEFAULT_MERGE_ADJACENT,
        },
        "quick": bool(args.quick),
        "paper_focused": bool(args.paper_focused),
        "large": bool(args.large),
        "headline_metrics": [
            "representative_violation_rate",
            "candidate_violation_auroc",
            "rule_count",
        ],
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")

    emitted_rows: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []

    total_units = args.num_seeds * len(DEFAULT_N_DIMS) * len(MODELS) * args.sample_test
    tracker = ProgressTracker("Scenario A2", total_units)
    tracker.start(
        (
            f"seeds={args.num_seeds}, dims={list(DEFAULT_N_DIMS)}, models={list(MODELS)}, "
            f"instances_per_group={args.sample_test}, output_dir={out_dir}"
        )
    )
    progress_path = out_dir / "progress.json"
    write_progress_snapshot(progress_path, tracker, detail="started", extra={"emitted_rows": 0, "candidate_rows": 0})

    multibin_noguard_sig = 1e-6

    for seed in range(args.num_seeds):
        for n_dim in DEFAULT_N_DIMS:
            x_train, y_train, x_cal, y_cal, x_test, _ = build_splits(
                seed,
                n_dim,
                args.n_train,
                args.n_cal,
                args.n_test,
            )
            rng = np.random.default_rng(seed * 10_000 + n_dim * 101 + 7)
            sample_size = min(args.sample_test, args.n_test)
            sampled_ids = np.sort(rng.choice(np.arange(args.n_test), size=sample_size, replace=False))

            for model_name, model in get_models(seed).items():
                wrapper = ensure_ce_first_wrapper(model)
                fit_and_calibrate(wrapper, x_train, y_train, x_cal, y_cal)
                assert wrapper.explainer is not None
                assert np.array_equal(np.asarray(wrapper.explainer.x_cal), np.asarray(x_cal))

                for local_idx, instance_id in enumerate(sampled_ids, start=1):
                    x_instance = x_test[instance_id : instance_id + 1]
                    bundle = _build_method_bundle_for_instance(
                        wrapper,
                        x_instance,
                        seed=seed,
                        model_name=model_name,
                        n_dim=n_dim,
                        instance_id=int(instance_id),
                        significance_grid=DEFAULT_SIGNIFICANCE,
                        multibin_noguard_sig=multibin_noguard_sig,
                    )

                    standard_rows = [dict(r) for r in bundle.standard_rows]
                    for significance in DEFAULT_SIGNIFICANCE:
                        # Standard is duplicated across significance to keep paired summaries aligned.
                        emitted_rows.extend(
                            [
                                dict(
                                    row,
                                    significance=float(significance),
                                    method="standard",
                                )
                                for row in standard_rows
                            ]
                        )

                        emitted_rows.extend(bundle.multibin_rows_by_significance[float(significance)])
                        emitted_rows.extend(bundle.guarded_rows_by_significance[float(significance)])
                        candidate_rows.extend(
                            bundle.candidate_rows_by_method_and_significance[("multibin_noguard", float(significance))]
                        )
                        candidate_rows.extend(
                            bundle.candidate_rows_by_method_and_significance[("guarded", float(significance))]
                        )

                    tracker.advance(
                        (
                            f"seed {seed + 1}/{args.num_seeds}, n_dim={n_dim}, model={model_name}, "
                            f"instance={local_idx}/{len(sampled_ids)}"
                        )
                    )

                write_progress_snapshot(
                    progress_path,
                    tracker,
                    detail=f"seed={seed}, n_dim={n_dim}, model={model_name} complete",
                    extra={"emitted_rows": len(emitted_rows), "candidate_rows": len(candidate_rows)},
                )

    emitted_df = pd.DataFrame(emitted_rows)
    candidate_df = pd.DataFrame(candidate_rows)

    if emitted_df.empty:
        raise RuntimeError("No emitted rows were produced.")
    if candidate_df.empty:
        raise RuntimeError("No candidate rows were produced.")

    # Compute representative-level violation labels from representative values only.
    x_lookup: Dict[Tuple[int, int, str, int], np.ndarray] = {}
    for seed in range(args.num_seeds):
        for n_dim in DEFAULT_N_DIMS:
            _, _, _, _, x_test, _ = build_splits(seed, n_dim, args.n_train, args.n_cal, args.n_test)
            for model_name in MODELS:
                for inst_id in emitted_df[
                    (emitted_df["seed"] == seed) & (emitted_df["n_dim"] == n_dim) & (emitted_df["model"] == model_name)
                ]["instance_id"].dropna().unique().tolist():
                    x_lookup[(seed, n_dim, model_name, int(inst_id))] = np.array(x_test[int(inst_id)], copy=True)

    def _attach_violation(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        violations: List[Optional[bool]] = []
        for row in out.itertuples(index=False):
            key = (int(row.seed), int(row.n_dim), str(row.model), int(row.instance_id))
            x_instance = x_lookup.get(key)
            if x_instance is None:
                violations.append(None)
                continue
            violations.append(
                _violates_constraint_with_feature_replacement(
                    x_instance,
                    int(row.feature),
                    row.representative,
                )
            )
        out["violates_constraint"] = violations
        return out

    emitted_df = _attach_violation(emitted_df)
    candidate_df = _attach_violation(candidate_df)

    # Random-pruned multibin baseline matched to guarded emitted rule count per
    # seed-model-dim-significance group.
    pruned_rows: List[Dict[str, Any]] = []
    group_cols = ["seed", "model", "n_dim", "significance"]
    grouped_keys = (
        emitted_df[group_cols]
        .drop_duplicates()
        .sort_values(group_cols)
        .itertuples(index=False)
    )
    for g in grouped_keys:
        g_seed = int(g.seed)
        g_model = str(g.model)
        g_dim = int(g.n_dim)
        g_sig = float(g.significance)

        guarded_pool = emitted_df[
            (emitted_df["seed"] == g_seed)
            & (emitted_df["model"] == g_model)
            & (emitted_df["n_dim"] == g_dim)
            & (emitted_df["significance"] == g_sig)
            & (emitted_df["method"] == "guarded")
        ]
        mb_pool = emitted_df[
            (emitted_df["seed"] == g_seed)
            & (emitted_df["model"] == g_model)
            & (emitted_df["n_dim"] == g_dim)
            & (emitted_df["significance"] == g_sig)
            & (emitted_df["method"] == "multibin_noguard")
        ]

        target_count = int(len(guarded_pool))
        source_count = int(len(mb_pool))
        if target_count <= 0 or source_count <= 0:
            continue

        keep = min(target_count, source_count)
        rng = np.random.default_rng(_deterministic_group_seed(g_seed, g_model, g_dim, g_sig))
        chosen_idx = rng.choice(mb_pool.index.to_numpy(), size=keep, replace=False)
        sampled = mb_pool.loc[chosen_idx].copy()
        sampled["method"] = "random_pruned_multibin"
        pruned_rows.extend(sampled.to_dict(orient="records"))

    if pruned_rows:
        emitted_df = pd.concat([emitted_df, pd.DataFrame(pruned_rows)], ignore_index=True)

    # Persist raw artifacts.
    candidate_cols = [
        "seed",
        "model",
        "n_dim",
        "significance",
        "method",
        "instance_id",
        "rule_id",
        "feature",
        "representative",
        "p_value",
        "is_constrained_feature",
        "violates_constraint",
        "emitted",
    ]
    emitted_cols = [
        "seed",
        "model",
        "n_dim",
        "significance",
        "method",
        "instance_id",
        "rule_id",
        "feature",
        "representative",
        "p_value",
        "is_constrained_feature",
        "violates_constraint",
        "emitted",
    ]
    candidate_df = candidate_df.reindex(columns=candidate_cols)
    emitted_df = emitted_df.reindex(columns=emitted_cols)

    candidate_path = out_dir / "candidate_records.csv"
    emitted_path = out_dir / "emitted_rule_records.csv"
    candidate_df.to_csv(candidate_path, index=False)
    emitted_df.to_csv(emitted_path, index=False)

    # Aggregate metrics at seed x model x dimension x method x significance.
    summary_rows: List[Dict[str, Any]] = []
    methods_for_summary = ["standard", "multibin_noguard", "guarded", "random_pruned_multibin"]

    for (seed, model_name, n_dim, significance), group_emitted in emitted_df.groupby(
        ["seed", "model", "n_dim", "significance"], sort=True
    ):
        all_instances = sorted(group_emitted["instance_id"].dropna().astype(int).unique().tolist())
        if not all_instances:
            all_instances = sorted(
                candidate_df[
                    (candidate_df["seed"] == seed)
                    & (candidate_df["model"] == model_name)
                    & (candidate_df["n_dim"] == n_dim)
                    & (candidate_df["significance"] == significance)
                ]["instance_id"].dropna().astype(int).unique().tolist()
            )

        for method in methods_for_summary:
            m_emit = group_emitted[group_emitted["method"] == method]
            constrained = m_emit[m_emit["is_constrained_feature"] == True]  # noqa: E712
            constrained_valid = constrained[constrained["violates_constraint"].notna()]
            if constrained_valid.empty:
                rep_violation = float("nan")
            else:
                rep_violation = float(constrained_valid["violates_constraint"].astype(float).mean())

            if all_instances:
                counts = (
                    m_emit.groupby("instance_id").size().reindex(all_instances, fill_value=0).astype(float)
                )
                rule_count = float(counts.mean())
            else:
                rule_count = float("nan")

            summary_rows.append(
                {
                    "seed": int(seed),
                    "model": str(model_name),
                    "n_dim": int(n_dim),
                    "method": str(method),
                    "significance": float(significance),
                    "metric": "representative_violation_rate",
                    "value": rep_violation,
                }
            )
            summary_rows.append(
                {
                    "seed": int(seed),
                    "model": str(model_name),
                    "n_dim": int(n_dim),
                    "method": str(method),
                    "significance": float(significance),
                    "metric": "rule_count",
                    "value": rule_count,
                }
            )

            # Mechanism metric is defined for scored candidate sets (guarded + multibin_noguard).
            if method in {"guarded", "multibin_noguard"}:
                cands = candidate_df[
                    (candidate_df["seed"] == seed)
                    & (candidate_df["model"] == model_name)
                    & (candidate_df["n_dim"] == n_dim)
                    & (candidate_df["significance"] == significance)
                    & (candidate_df["method"] == method)
                    & (candidate_df["is_constrained_feature"] == True)  # noqa: E712
                ]
                cands = cands[cands["p_value"].notna() & cands["violates_constraint"].notna()]
                auroc = _auroc_from_candidates(cands)
            else:
                auroc = float("nan")

            summary_rows.append(
                {
                    "seed": int(seed),
                    "model": str(model_name),
                    "n_dim": int(n_dim),
                    "method": str(method),
                    "significance": float(significance),
                    "metric": "candidate_violation_auroc",
                    "value": auroc,
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "summary_metrics.csv"
    summary_df.to_csv(summary_path, index=False)

    _validate_summary_contract(summary_df)
    _validate_required_methods(summary_df)
    _validate_setup_consistency(run_cfg, summary_df)

    report_sections = _build_report_sections(run_cfg=run_cfg, summary_df=summary_df)
    write_report(
        out_dir / "report.md",
        "Scenario A2: Representative Constraint Filtering in Guarded Factual Explanations",
        report_sections,
    )
    report_text = (out_dir / "report.md").read_text(encoding="utf-8")
    _validate_report_content(report_text)

    tracker.finish(f"final artifacts written to {out_dir}")
    write_progress_snapshot(
        progress_path,
        tracker,
        detail="completed",
        extra={
            "candidate_rows": int(len(candidate_df)),
            "emitted_rows": int(len(emitted_df)),
            "summary_rows": int(len(summary_df)),
            "artifacts": {
                "candidate_records": str(candidate_path),
                "emitted_rule_records": str(emitted_path),
                "summary_metrics": str(summary_path),
                "report": str(out_dir / "report.md"),
                "run_config": str(out_dir / "run_config.json"),
            },
        },
    )

    print(f"Wrote: {candidate_path}")
    print(f"Wrote: {emitted_path}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
