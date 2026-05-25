"""Tests for Scenario 3 thresholded-regression binary-event diagnostics."""

from __future__ import annotations

import numpy as np

from calibrated_explanations.core.reject.orchestrator import (
    regression_threshold_event_labels,
)
from evaluation.reject.common_reject import binary_accuracy_from_threshold
from evaluation.reject.scenario_3_regression_threshold_baseline import (
    _prediction_set_counts,
    _singleton_error,
)


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
    assert _singleton_error(prediction_set, labels) == 0.0


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

    assert _singleton_error(prediction_set, labels) == 0.5


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
    counts = _prediction_set_counts(prediction_set)

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
