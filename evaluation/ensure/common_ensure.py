"""Shared helpers for ensured evaluation scripts.

All code in this module is evaluation-only (ADR-010). It intentionally uses
public APIs from `calibrated_explanations` and avoids bringing heavy optional
dependencies into core.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Literal, Optional

import numpy as np
import pandas as pd

from calibrated_explanations.utils.helper import calculate_metrics


@dataclass(frozen=True)
class EnsureRunConfig:
    """Configuration shared by all ensured evaluation runners."""

    test_size: int = 100
    calibration_sizes: tuple[int, ...] = (100, 300, 500)
    random_state: int = 42
    # Conjunction controls
    n_top_features: int = 5
    max_rule_size: int = 2
    # Ranking analysis weights (paper)
    ranking_weights: tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0)
    # Plain regression normalization for ensured scoring
    normalize_plain_regression: bool = True
    # Regression interval percentiles for plain regression
    low_high_percentiles: tuple[int, int] = (5, 95)


def _safe_test_size(n_samples: int, requested: int) -> int:
    if n_samples <= 3:
        return 1
    # Keep at least 2 samples for training+calibration pool.
    max_allowed = max(1, n_samples - 2)
    if requested <= 0:
        return 1
    if requested <= max_allowed:
        return requested
    # Fall back to 20% when dataset is small.
    return max(1, n_samples // 5)


def _safe_calibration_pool_size(n_train: int, requested_max: int, *, min_remaining: int = 2) -> int:
    """Choose a calibration-pool size that leaves enough samples for training.

    Parameters
    ----------
    n_train:
        Number of samples available before splitting into proper-train and
        calibration-pool.
    requested_max:
        Upper bound (e.g., max calibration size).
    min_remaining:
        Minimum number of samples that must remain in the proper-training
        split.
    """

    min_remaining = max(1, int(min_remaining))
    max_allowed = max(1, n_train - min_remaining)
    return min(int(requested_max), int(max_allowed))


def subsample_calibration(
    x_cal_pool: np.ndarray,
    y_cal_pool: np.ndarray,
    cal_size: int,
    *,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    if cal_size >= len(x_cal_pool):
        return x_cal_pool, y_cal_pool

    # Attempt a stratified calibration subsample to preserve class coverage
    try:
        from sklearn.model_selection import train_test_split

        # Only stratify when y_cal_pool is 1D and has more than one unique value
        if y_cal_pool.ndim == 1 and len(np.unique(y_cal_pool)) > 1:
            # train_test_split here returns the 'calibration' subset of the
            # requested size using `test_size=cal_size` semantics on the pool.
            _, x_cal, _, y_cal = train_test_split(
                x_cal_pool,
                y_cal_pool,
                test_size=cal_size,
                random_state=random_state,
                stratify=y_cal_pool,
            )
            return x_cal, y_cal
    except Exception:
        # Fallback to randomized sampling below
        pass

    rng = np.random.RandomState(random_state)
    idx = rng.choice(len(x_cal_pool), size=cal_size, replace=False)
    return x_cal_pool[idx], y_cal_pool[idx]


def compute_thresholds_from_non_test_targets(
    y_non_test: np.ndarray, percentiles: Iterable[int]
) -> dict[str, float]:
    """Compute threshold values from non-test targets (avoids test leakage)."""

    thresholds: dict[str, float] = {}
    for p in percentiles:
        thresholds[f"p{p}"] = float(np.percentile(y_non_test, p))
    return thresholds


def _rules_frame(explanation: Any) -> pd.DataFrame:
    """Return a normalized DataFrame of alternative rules.

    Uses public `get_rules()` representation.
    """

    rules = explanation.get_rules()
    if not rules or len(rules.get("rule", [])) == 0:
        return pd.DataFrame(
            columns=[
                "rule",
                "predict",
                "predict_low",
                "predict_high",
            ]
        )
    return pd.DataFrame(
        {
            "rule": list(rules["rule"]),
            "predict": np.asarray(rules["predict"], dtype=float),
            "predict_low": np.asarray(rules["predict_low"], dtype=float),
            "predict_high": np.asarray(rules["predict_high"], dtype=float),
        }
    )


def ablation_counts(explanation: Any) -> dict[str, int]:
    """Compute per-instance counts used in the paper tables.

    Delegates entirely to the public filter API on the explanation object so
    counts are always consistent with what the API surfaces to callers.
    """
    total = len(explanation)
    if total == 0:
        return {
            "total": 0,
            "ensured": 0,
            "counterfactual": 0,
            "counterpotential": 0,
            "semifactual": 0,
            "semipotential": 0,
            "superfactual": 0,
            "superpotential": 0,
            "pareto": 0,
        }

    super_with = len(explanation.super(include_potential=True))
    super_without = len(explanation.super(include_potential=False))

    semi_with = len(explanation.semi(include_potential=True))
    semi_without = len(explanation.semi(include_potential=False))

    counter_with = len(explanation.counter(include_potential=True))
    counter_without = len(explanation.counter(include_potential=False))

    ensured_with = len(explanation.ensured(include_potential=True))

    pareto_with = len(explanation.pareto(include_potential=True))

    return {
        "total": total,
        "ensured": ensured_with,
        "counterfactual": counter_without,
        "counterpotential": counter_with - counter_without,
        "semifactual": semi_without,
        "semipotential": semi_with - semi_without,
        "superfactual": super_without,
        "superpotential": super_with - super_without,
        "pareto": pareto_with,
    }


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    # pandas handles ties/ranking correctly.
    return float(pd.Series(a).corr(pd.Series(b), method="spearman"))


def ranking_validation_for_instance(
    explanation: Any,
    *,
    weights: Iterable[float],
    normalize: bool,
) -> dict[str, Any]:
    """Compute ranking-metric validation stats for one instance explanation."""

    df = _rules_frame(explanation)
    if df.empty:
        return {
            "n_rules": 0,
            "per_w": {},
        }

    uncertainty = (df["predict_high"] - df["predict_low"]).to_numpy(dtype=float)
    prediction = df["predict"].to_numpy(dtype=float)

    per_w: dict[str, Any] = {}
    for w in weights:
        score = calculate_metrics(
            uncertainty=uncertainty,
            prediction=prediction,
            w=float(w),
            metric="ensured",
            normalize=normalize,
        )
        score = np.asarray(score, dtype=float)
        if score.size == 0 or np.all(np.isnan(score)):
            continue

        per_w[str(w)] = {
            "spearman_score_uncertainty": _spearman(score, uncertainty),
            "spearman_score_prediction": _spearman(score, prediction),
        }

    return {
        "n_rules": int(len(df)),
        "per_w": per_w,
    }


def summarize_ranking_validation(
    per_instance: list[dict[str, Any]],
    *,
    weights: Iterable[float],
) -> dict[str, Any]:
    """Aggregate per-instance validation into dataset-level summary."""

    per_w_summary: dict[str, Any] = {}
    for w in weights:
        w_key = str(float(w))
        rows = [d.get("per_w", {}).get(w_key) for d in per_instance]
        rows = [r for r in rows if r is not None]
        if not rows:
            continue

        def _mean(field: str) -> float:
            vals = [r.get(field) for r in rows]
            vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
            return float(np.mean(vals)) if vals else float("nan")

        per_w_summary[w_key] = {
            "spearman_score_uncertainty": _mean("spearman_score_uncertainty"),
            "spearman_score_prediction": _mean("spearman_score_prediction"),
        }

    return {
        "per_w": per_w_summary,
    }


def _pareto_indexes(
    pred: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
) -> list[int]:
    """Return indexes on the output-envelope Pareto frontier.

    This is a direct port of `Explanation.__pareto_rule_indexes`.

    The frontier is *weight-independent*: it keeps the minimum-uncertainty
    rules across the entire output axis (a V-shaped envelope).
    """

    outputs = np.asarray(pred, dtype=float)
    predict_low = np.asarray(low, dtype=float)
    predict_high = np.asarray(high, dtype=float)

    rule_count = int(len(outputs))
    if rule_count <= 1:
        return list(range(rule_count))

    tolerance = 1e-12
    best_per_output: dict[float, dict[str, float | int]] = {}
    for index in range(rule_count):
        output_value = float(outputs[index])
        uncertainty_value = float(predict_high[index]) - float(predict_low[index])
        output_key = round(output_value, 12)

        current_best = best_per_output.get(output_key)
        if current_best is None:
            best_per_output[output_key] = {
                "index": int(index),
                "output": output_value,
                "uncertainty": uncertainty_value,
            }
            continue

        if (
            uncertainty_value < float(current_best["uncertainty"]) - tolerance
            or math.isclose(
                uncertainty_value,
                float(current_best["uncertainty"]),
                rel_tol=tolerance,
                abs_tol=tolerance,
            )
            and int(index) < int(current_best["index"])
        ):
            best_per_output[output_key] = {
                "index": int(index),
                "output": output_value,
                "uncertainty": uncertainty_value,
            }

    candidates = sorted(best_per_output.values(), key=lambda candidate: float(candidate["output"]))
    if len(candidates) <= 2:
        return sorted(int(candidate["index"]) for candidate in candidates)

    left_mins: list[float] = []
    running_left_min = float("inf")
    for candidate in candidates:
        running_left_min = min(running_left_min, float(candidate["uncertainty"]))
        left_mins.append(running_left_min)

    right_mins: list[float] = [0.0] * len(candidates)
    running_right_min = float("inf")
    for reverse_index in range(len(candidates) - 1, -1, -1):
        running_right_min = min(
            running_right_min, float(candidates[reverse_index]["uncertainty"])
        )
        right_mins[reverse_index] = running_right_min

    kept_indexes = {
        int(candidates[0]["index"]),
        int(candidates[-1]["index"]),
    }
    for position, candidate in enumerate(candidates):
        uncertainty_value = float(candidate["uncertainty"])
        if (
            uncertainty_value <= left_mins[position] + tolerance
            or uncertainty_value <= right_mins[position] + tolerance
        ):
            kept_indexes.add(int(candidate["index"]))

    return sorted(kept_indexes)


def timed_call(fn, *args, **kwargs):
    """Return (result, elapsed_seconds)."""

    tic = time.time()
    out = fn(*args, **kwargs)
    return out, float(time.time() - tic)


def can_run_dataset(
    *,
    n_samples: int,
    task: Literal["binary", "multiclass", "regression"],
    config: EnsureRunConfig,
    n_classes: int | None = None,
) -> tuple[bool, str]:
    """Decide whether the dataset is suitable for running the ensured evaluation.

    Returns (ok, reason). If ok is False, reason describes why it's skipped.
    """

    if n_samples <= 3:
        return False, f"too few samples ({n_samples})"

    test_size = _safe_test_size(n_samples, config.test_size)
    if test_size >= n_samples:
        return False, f"test size {test_size} >= n_samples {n_samples}"

    # After removing test, need calibration + minimum training samples
    remaining = n_samples - test_size
    min_cal = max(config.calibration_sizes) if config.calibration_sizes else 100
    min_train = max(100, int(n_classes)) if task in ("binary", "multiclass") and n_classes else 100
    min_needed = min_cal + min_train

    if remaining < min_needed:
        return False, (
            f"not enough remaining samples after test ({remaining} < "
            f"{min_cal} cal + {min_train} train)"
        )

    # For classification, ensure at least one sample per class remains for stratify
    if task in ("binary", "multiclass"):
        if n_classes is None:
            return False, "n_classes required for classification checks"
        if remaining <= int(n_classes):
            return False, (
                f"not enough samples for {n_classes} classes "
                f"(remaining={remaining})"
            )

    return True, "ok"
