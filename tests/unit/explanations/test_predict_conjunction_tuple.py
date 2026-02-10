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


def test_predict_conjunction_tuple_basic():
    """Basic test: known inputs produce expected output."""
    # predict_fn returns known values for any batch
    p_vals = np.array([0.6, 0.8])
    low_vals = np.array([0.5, 0.7])
    high_vals = np.array([0.7, 0.9])
    dummy = np.array([0.0, 0.0])

    exp, mock_predict = make_explanation_with_mock_predict((p_vals, low_vals, high_vals, dummy))

    perturbed = np.array([[1.0, 2.0, 3.0]])
    rule_value_set = [np.array([10.0, 20.0])]  # 2 values for feature 0
    original_features = [0]

    p, lo, hi = exp.predict_conjunction_tuple(
        rule_value_set, original_features, perturbed, threshold=0.5, predicted_class=1
    )

    assert p == pytest.approx(0.7)  # mean of [0.6, 0.8]
    assert lo == pytest.approx(0.6)  # mean of [0.5, 0.7]
    assert hi == pytest.approx(0.8)  # mean of [0.7, 0.9]

    # Verify predict was called with correct batch shape
    call_args = mock_predict.call_args
    batch_arg = call_args[0][0]
    assert batch_arg.shape == (2, 3)  # 2 combinations x 3 features


def test_predict_conjunction_tuple_1d_perturbed():
    """1D perturbed input should be handled without error."""
    p_vals = np.array([0.5])
    low_vals = np.array([0.4])
    high_vals = np.array([0.6])
    dummy = np.array([0.0])

    exp, mock_predict = make_explanation_with_mock_predict((p_vals, low_vals, high_vals, dummy))

    # 1D perturbed - this used to fail before hardening
    perturbed = np.array([1.0, 2.0, 3.0])
    rule_value_set = [np.array([10.0])]
    original_features = [0]

    p, lo, hi = exp.predict_conjunction_tuple(
        rule_value_set, original_features, perturbed, threshold=0.5, predicted_class=1
    )

    assert isinstance(p, float)
    assert isinstance(lo, float)
    assert isinstance(hi, float)


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


def test_predict_conjunction_tuple_numpy_int_features():
    """numpy integer features should be coerced to Python ints."""
    p_vals = np.array([0.5])
    low_vals = np.array([0.4])
    high_vals = np.array([0.6])
    dummy = np.array([0.0])

    exp, mock_predict = make_explanation_with_mock_predict((p_vals, low_vals, high_vals, dummy))

    perturbed = np.array([[1.0, 2.0, 3.0]])
    rule_value_set = [np.array([10.0])]
    # Use numpy int64 instead of Python int
    original_features = [np.int64(1)]

    p, lo, hi = exp.predict_conjunction_tuple(
        rule_value_set, original_features, perturbed, threshold=0.5, predicted_class=1
    )

    assert isinstance(p, float)


def test_predict_conjunction_tuple_multi_feature():
    """Test conjunctions with multiple features."""
    # 2 features x 2 values each = 4 combinations
    p_vals = np.array([0.5, 0.6, 0.7, 0.8])
    low_vals = np.array([0.4, 0.5, 0.6, 0.7])
    high_vals = np.array([0.6, 0.7, 0.8, 0.9])
    dummy = np.zeros(4)

    exp, mock_predict = make_explanation_with_mock_predict((p_vals, low_vals, high_vals, dummy))

    perturbed = np.array([[1.0, 2.0, 3.0]])
    rule_value_set = [np.array([10.0, 20.0]), np.array([30.0, 40.0])]
    original_features = [0, 2]

    p, lo, hi = exp.predict_conjunction_tuple(
        rule_value_set, original_features, perturbed, threshold=0.5, predicted_class=1
    )

    assert p == pytest.approx(0.65)  # mean of [0.5, 0.6, 0.7, 0.8]

    # Verify batch shape
    call_args = mock_predict.call_args
    batch_arg = call_args[0][0]
    assert batch_arg.shape == (4, 3)
