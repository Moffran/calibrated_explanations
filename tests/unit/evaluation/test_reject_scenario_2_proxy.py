"""Tests for Scenario 2 multiclass correctness-proxy diagnostics."""

from __future__ import annotations

import numpy as np

from evaluation.reject.scenario_2_multiclass_correctness import (
    proxy_correctness_diagnostics,
)
from evaluation.reject.common_reject import singleton_precision_recall


def test_proxy_correctness_accuracy_uses_binary_top1_correctness_labels():
    """Proxy accuracy must score {0}/{1} against top-1 correctness labels."""
    y_true = np.array([2, 1, 0, 1])
    top1_pred = np.array([2, 2, 0, 0])
    positive_singleton = np.array([True, True, False, False])
    proxy_negative_singleton = np.array([False, False, True, True])

    diagnostics = proxy_correctness_diagnostics(
        y_true,
        top1_pred,
        positive_singleton,
        proxy_negative_singleton,
    )

    assert diagnostics["proxy_singleton_count"] == 4
    assert diagnostics["proxy_singleton_accuracy_defined"] is True
    assert diagnostics["proxy_singleton_accuracy"] == 0.5
    assert diagnostics["accepted_top1_accuracy"] == 0.5
    assert diagnostics["proxy_negative_singleton_accuracy"] == 0.5


def test_proxy_correctness_accuracy_is_undefined_without_singletons():
    """Proxy accuracy is undefined when no singleton proxy prediction exists."""
    diagnostics = proxy_correctness_diagnostics(
        np.array([0, 1]),
        np.array([0, 0]),
        np.array([False, False]),
        np.array([False, False]),
    )

    assert diagnostics["proxy_singleton_count"] == 0
    assert diagnostics["proxy_singleton_accuracy_defined"] is False
    assert np.isnan(diagnostics["proxy_singleton_accuracy"])


def test_singleton_precision_recall_scores_correct_singletons_only():
    """Singleton precision/recall should ignore ambiguous and empty sets."""
    prediction_set = np.array(
        [
            [False, True],
            [True, False],
            [True, True],
            [False, False],
        ],
        dtype=bool,
    )
    metrics = singleton_precision_recall(prediction_set, np.array([1, 1, 0, 0]))

    assert metrics["singleton_count"] == 2
    assert metrics["singleton_correct_count"] == 1
    assert metrics["singleton_precision"] == 0.5
    assert metrics["singleton_recall"] == 0.25
    assert metrics["singleton_precision_recall_defined"] is True
