"""Scenario 3: thresholded regression binary-event reject validity.

Paper mapping: RQ3 (empirical binary-event validity diagnostic).

Thresholded regression reject is evaluated as conformal binary classification
over a user-defined event, not as interval-width selection:

* scalar threshold: event 1 iff ``y <= threshold``;
* interval threshold: event 1 iff ``low < y <= high``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from calibrated_explanations.core.reject.orchestrator import (
    regression_threshold_event_labels,
)

from .common_reject import (
    RunConfig,
    _markdown_table_from_df,
    build_regression_bundle,
    clopper_pearson_interval,
    empirical_coverage,
    quantile_grid,
    reject_breakdown,
    singleton_precision_recall,
    task_specs,
    write_csv_json_md,
)


@dataclass(frozen=True)
class ThresholdSpec:
    """Concrete threshold contract used by Scenario 3."""

    threshold_id: str
    threshold_type: str
    threshold: float | tuple[float, float]
    threshold_quantile: float | None
    threshold_lower_quantile: float | None
    threshold_upper_quantile: float | None


def _threshold_specs(y_cal: np.ndarray, quick: bool) -> tuple[ThresholdSpec, ...]:
    """Return scalar quantile thresholds plus one shared interval threshold."""
    specs: list[ThresholdSpec] = []
    for quantile in tuple(float(q) for q in quantile_grid(quick)):
        threshold = float(np.quantile(y_cal, quantile))
        specs.append(
            ThresholdSpec(
                threshold_id=f"q{quantile:.2f}",
                threshold_type="scalar",
                threshold=threshold,
                threshold_quantile=quantile,
                threshold_lower_quantile=None,
                threshold_upper_quantile=None,
            )
        )

    low_q = 0.25
    high_q = 0.75
    low = float(np.quantile(y_cal, low_q))
    high = float(np.quantile(y_cal, high_q))
    specs.append(
        ThresholdSpec(
            threshold_id="q0.25_q0.75",
            threshold_type="interval",
            threshold=(low, high),
            threshold_quantile=None,
            threshold_lower_quantile=low_q,
            threshold_upper_quantile=high_q,
        )
    )
    return tuple(specs)


def _prediction_set_counts(prediction_set: Any) -> dict[str, float | int]:
    """Return event prediction-set structural counts and rates."""
    prediction_set_arr = np.asarray(prediction_set, dtype=bool)
    if prediction_set_arr.ndim != 2 or prediction_set_arr.shape[0] == 0:
        return {
            "n_total": 0,
            "n_empty": 0,
            "n_singleton": 0,
            "n_ambiguity": 0,
            "novelty_rate": float("nan"),
            "singleton_rate": float("nan"),
            "ambiguity_rate": float("nan"),
            "reject_rate_from_sets": float("nan"),
        }
    set_sizes = np.sum(prediction_set_arr, axis=1)
    n_total = int(prediction_set_arr.shape[0])
    n_empty = int(np.sum(set_sizes == 0))
    n_singleton = int(np.sum(set_sizes == 1))
    n_ambiguity = int(np.sum(set_sizes >= 2))
    return {
        "n_total": n_total,
        "n_empty": n_empty,
        "n_singleton": n_singleton,
        "n_ambiguity": n_ambiguity,
        "novelty_rate": float(n_empty / n_total),
        "singleton_rate": float(n_singleton / n_total),
        "ambiguity_rate": float(n_ambiguity / n_total),
        "reject_rate_from_sets": float((n_empty + n_ambiguity) / n_total),
    }


def _singleton_error(prediction_set: Any, event_labels: np.ndarray) -> float:
    """Return empirical event error among singleton rows only."""
    prediction_set_arr = np.asarray(prediction_set, dtype=bool)
    labels = np.asarray(event_labels, dtype=int).reshape(-1)
    if prediction_set_arr.ndim != 2 or prediction_set_arr.shape[0] != len(labels):
        return float("nan")
    if prediction_set_arr.shape[1] <= int(np.max(labels, initial=0)):
        return float("nan")
    set_sizes = np.sum(prediction_set_arr, axis=1)
    singleton_mask = set_sizes == 1
    if not np.any(singleton_mask):
        return float("nan")
    covered = prediction_set_arr[np.arange(len(labels)), labels]
    return float(np.mean(~covered[singleton_mask]))


def run(config: RunConfig) -> None:
    """Measure binary-event coverage for thresholded regression reject."""
    rows: list[dict[str, float | str | int | bool | None]] = []
    confidences = (0.90, 0.95)

    for spec in task_specs("regression", quick=config.quick):
        bundle = build_regression_bundle(spec, config)
        for threshold_spec in _threshold_specs(bundle.y_cal, config.quick):
            event_labels = regression_threshold_event_labels(
                bundle.y_test,
                threshold_spec.threshold,
            )
            event_prevalence = float(np.mean(event_labels)) if len(event_labels) else float("nan")
            for confidence in confidences:
                epsilon = 1.0 - float(confidence)
                breakdown = reject_breakdown(
                    bundle.wrapper,
                    bundle.x_test,
                    confidence=float(confidence),
                    threshold=threshold_spec.threshold,
                )
                prediction_set = breakdown.get("prediction_set")
                coverage_defined = (
                    prediction_set is not None
                    and np.asarray(prediction_set).ndim == 2
                    and np.asarray(prediction_set).shape[0] == len(event_labels)
                    and np.asarray(prediction_set).shape[1] >= 2
                )
                if coverage_defined:
                    prediction_set_arr = np.asarray(prediction_set, dtype=bool)
                    coverage = empirical_coverage(prediction_set_arr, event_labels)
                    successes = int(
                        np.sum(prediction_set_arr[np.arange(len(event_labels)), event_labels])
                    )
                    lower_ci, upper_ci = clopper_pearson_interval(
                        successes,
                        len(event_labels),
                    )
                    singleton_metrics = singleton_precision_recall(
                        prediction_set_arr,
                        event_labels,
                    )
                    set_counts = _prediction_set_counts(prediction_set_arr)
                    singleton_error = _singleton_error(prediction_set_arr, event_labels)
                else:
                    coverage = float("nan")
                    lower_ci = float("nan")
                    upper_ci = float("nan")
                    singleton_metrics = singleton_precision_recall(None, event_labels)
                    set_counts = _prediction_set_counts(None)
                    singleton_error = float("nan")

                threshold = threshold_spec.threshold
                rows.append(
                    {
                        "dataset": spec.name,
                        "confidence": float(confidence),
                        "epsilon": float(epsilon),
                        "effective_confidence": float(breakdown["effective_confidence"]),
                        "threshold_type": threshold_spec.threshold_type,
                        "threshold_id": threshold_spec.threshold_id,
                        "threshold_quantile": threshold_spec.threshold_quantile,
                        "threshold_lower_quantile": threshold_spec.threshold_lower_quantile,
                        "threshold_upper_quantile": threshold_spec.threshold_upper_quantile,
                        "effective_threshold": str(breakdown.get("effective_threshold")),
                        "threshold_value": (
                            float(threshold)
                            if threshold_spec.threshold_type == "scalar"
                            else float("nan")
                        ),
                        "threshold_low": (
                            float(threshold[0])
                            if threshold_spec.threshold_type == "interval"
                            else float("nan")
                        ),
                        "threshold_high": (
                            float(threshold[1])
                            if threshold_spec.threshold_type == "interval"
                            else float("nan")
                        ),
                        "threshold_source": breakdown.get("threshold_source"),
                        "n_cal": int(len(bundle.x_cal)),
                        "n_test": int(len(bundle.x_test)),
                        "event_prevalence": event_prevalence,
                        "empirical_event_coverage": coverage,
                        "coverage_defined": bool(coverage_defined),
                        "lower_ci": lower_ci,
                        "upper_ci": upper_ci,
                        "violation": bool(coverage_defined and coverage < confidence),
                        "structural_violation": bool(
                            coverage_defined and upper_ci < confidence
                        ),
                        "empirical_singleton_error": singleton_error,
                        "reject_rate": float(breakdown["reject_rate"]),
                        **set_counts,
                        **singleton_metrics,
                    }
                )

    df = pd.DataFrame(rows)
    violation_count = int(df["violation"].sum()) if not df.empty else 0
    structural_count = int(df["structural_violation"].sum()) if not df.empty else 0
    mean_coverage = (
        float(df["empirical_event_coverage"].mean()) if not df.empty else float("nan")
    )
    meta = {
        "scenario": "scenario_3_regression_threshold_baseline",
        "display_name": "Scenario 3 - Thresholded regression binary-event reject validity",
        "paper_rq": "RQ3",
        "guarantee_status": "empirical_binary_event_validity_diagnostic",
        "quick": config.quick,
        "highlights": [
            "Thresholded regression reject is evaluated as binary conformal classification over event labels.",
            "Scalar event: y <= threshold. Interval event: low < y <= high.",
            "Coverage is empirical event-label coverage from conformal prediction sets over {0, 1}.",
            "Singleton precision, recall, and empirical singleton error are derived from those same event labels.",
            "No interval-width selection, interval coverage, or accepted-interval-width diagnostic is part of Scenario 3.",
            f"Observed event-coverage violations: {violation_count}/{len(df)}.",
            f"Structural violations (CI upper bound < confidence): {structural_count}/{len(df)}.",
        ],
        "outcome": {
            "datasets": int(df["dataset"].nunique()) if not df.empty else 0,
            "rows": int(len(df)),
            "mean_event_coverage": mean_coverage,
            "mean_reject_rate": float(df["reject_rate"].mean()) if not df.empty else float("nan"),
            "mean_empirical_singleton_error": (
                float(df["empirical_singleton_error"].mean())
                if not df.empty
                else float("nan")
            ),
            "coverage_violations": violation_count,
            "structural_violations": structural_count,
        },
    }

    extra_sections: list[str] = []
    if not df.empty:
        by_type = (
            df.groupby("threshold_type")
            .agg(
                n_rows=("empirical_event_coverage", "size"),
                mean_event_prevalence=("event_prevalence", "mean"),
                mean_event_coverage=("empirical_event_coverage", "mean"),
                violations=("violation", "sum"),
                structural_violations=("structural_violation", "sum"),
                mean_reject_rate=("reject_rate", "mean"),
                mean_singleton_precision=("singleton_precision", "mean"),
                mean_singleton_recall=("singleton_recall", "mean"),
            )
            .reset_index()
        )
        extra_sections += [
            "## By threshold type",
            "",
            _markdown_table_from_df(by_type),
            "",
        ]

        by_dataset = (
            df.groupby("dataset")
            .agg(
                violations=("violation", "sum"),
                structural_violations=("structural_violation", "sum"),
                mean_event_coverage=("empirical_event_coverage", "mean"),
                mean_reject_rate=("reject_rate", "mean"),
                mean_singleton_precision=("singleton_precision", "mean"),
                mean_singleton_recall=("singleton_recall", "mean"),
            )
            .reset_index()
            .sort_values("structural_violations", ascending=False)
        )
        extra_sections += [
            "## Per-dataset event validity",
            "",
            _markdown_table_from_df(by_dataset),
            "",
        ]

    write_csv_json_md(
        "scenario_3_regression_threshold_baseline",
        df,
        meta,
        extra_sections=extra_sections,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    arguments = parser.parse_args()
    run(RunConfig(seed=42, quick=arguments.quick))
