"""Tests for Scenario 3 thresholded-regression binary-event diagnostics."""

from __future__ import annotations

import numpy as np

from calibrated_explanations.core.reject.orchestrator import (
    regression_threshold_event_labels,
)
from evaluation.reject.common_reject import binary_accuracy_from_threshold


def singleton_error(prediction_set: np.ndarray, event_labels: np.ndarray) -> float:
    """Compute singleton-row event error for a binary prediction set."""
    set_sizes = np.sum(prediction_set, axis=1)
    singleton_mask = set_sizes == 1
    if not np.any(singleton_mask):
        return float("nan")

    covered = prediction_set[np.arange(len(event_labels)), event_labels]
    return float(np.mean(~covered[singleton_mask]))


def prediction_set_counts(prediction_set: np.ndarray) -> dict[str, float | int]:
    """Compute structural counts and rates for a binary prediction set."""
    set_sizes = np.sum(prediction_set, axis=1)
    n_total = int(prediction_set.shape[0])
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
    }


def test_scenario_3_coverage_labels_use_threshold_event_contract():
    labels = regression_threshold_event_labels(np.array([0.2, 0.5, 0.8]), 0.5)
    prediction_set = np.array(
        [
            [False, True],
            [False, True],
            [True, False],
        ],
        dtype=bool,
    )

    assert labels.tolist() == [1, 1, 0]
    assert singleton_error(prediction_set, labels) == 0.0


def test_scenario_3_singleton_error_is_from_binary_event_labels():
    prediction_set = np.array(
        [
            [False, True],
            [True, False],
            [True, True],
            [False, False],
        ],
        dtype=bool,
    )
    labels = np.array([1, 1, 0, 0])

    assert singleton_error(prediction_set, labels) == 0.5


def test_scenario_3_prediction_set_counts_are_event_set_counts():
    prediction_set = np.array(
        [
            [False, True],
            [True, False],
            [True, True],
            [False, False],
        ],
        dtype=bool,
    )
    counts = prediction_set_counts(prediction_set)

    assert counts["n_total"] == 4
    assert counts["n_empty"] == 1
    assert counts["n_singleton"] == 2
    assert counts["n_ambiguity"] == 1
    assert counts["novelty_rate"] == 0.25
    assert counts["singleton_rate"] == 0.5
    assert counts["ambiguity_rate"] == 0.25


def test_binary_accuracy_from_threshold_uses_same_event_contract():
    accuracy = binary_accuracy_from_threshold(
        np.array([0.5, 0.7, 0.8]),
        np.array([0.5, 0.4, 0.9]),
        threshold=0.5,
    )

    assert accuracy == 2 / 3
