"""Unit tests for predict_conjunction_tuple hardening (Phase 4B)."""

import numpy as np
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

from calibrated_explanations.explanations.explanation import FactualExplanation


def make_explanation_with_mock_predict(predict_fn_return):
    """Create a minimal CalibratedExplanation with mocked predict_internal."""
    exp = FactualExplanation.__new__(FactualExplanation)

    # Mock the chain: get_explainer().prediction_orchestrator.predict_internal
    mock_predict = MagicMock(return_value=predict_fn_return)
    mock_orchestrator = SimpleNamespace(predict_internal=mock_predict)
    mock_explainer = SimpleNamespace(prediction_orchestrator=mock_orchestrator)
    exp.get_explainer = MagicMock(return_value=mock_explainer)

    # Mock calibrated_explanations.low_high_percentiles
    mock_ce = SimpleNamespace(low_high_percentiles=(5, 95))
    exp.calibrated_explanations = mock_ce

    return exp, mock_predict






def test_predict_conjunction_tuple_empty_values():
    """Empty value iterables should return zeros."""
    exp, _ = make_explanation_with_mock_predict(
        (np.array([]), np.array([]), np.array([]), np.array([]))
    )

    perturbed = np.array([[1.0, 2.0]])
    rule_value_set = []
    original_features = []

    p, lo, hi = exp.predict_conjunction_tuple(
        rule_value_set, original_features, perturbed, threshold=0.5, predicted_class=1
    )

    assert p == 0.0
    assert lo == 0.0
    assert hi == 0.0


def test_predict_conjunction_tuple_bins_scalar():
    """Scalar bins should be tiled to match batch size."""
    p_vals = np.array([0.5, 0.6])
    low_vals = np.array([0.4, 0.5])
    high_vals = np.array([0.6, 0.7])
    dummy = np.array([0.0, 0.0])

    exp, mock_predict = make_explanation_with_mock_predict((p_vals, low_vals, high_vals, dummy))

    perturbed = np.array([[1.0, 2.0]])
    rule_value_set = [np.array([10.0, 20.0])]
    original_features = [0]

    exp.predict_conjunction_tuple(
        rule_value_set, original_features, perturbed, threshold=0.5, predicted_class=1, bins=3
    )

    call_args = mock_predict.call_args
    batch_bins = call_args[1]["bins"]
    assert len(batch_bins) == 2
    assert all(b == 3 for b in batch_bins)


def test_predict_conjunction_tuple_bins_array():
    """Array bins should be tiled to match batch size."""
    p_vals = np.array([0.5, 0.6])
    low_vals = np.array([0.4, 0.5])
    high_vals = np.array([0.6, 0.7])
    dummy = np.array([0.0, 0.0])

    exp, mock_predict = make_explanation_with_mock_predict((p_vals, low_vals, high_vals, dummy))

    perturbed = np.array([[1.0, 2.0]])
    rule_value_set = [np.array([10.0, 20.0])]
    original_features = [0]

    exp.predict_conjunction_tuple(
        rule_value_set,
        original_features,
        perturbed,
        threshold=0.5,
        predicted_class=1,
        bins=np.array([5]),
    )

    call_args = mock_predict.call_args
    batch_bins = call_args[1]["bins"]
    assert len(batch_bins) == 2




