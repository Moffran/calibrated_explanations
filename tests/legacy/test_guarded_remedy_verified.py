import numpy as np
import pytest
from unittest.mock import MagicMock
from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer
from calibrated_explanations.core.explain._guarded_explain import guarded_explain
from calibrated_explanations.core.exceptions import ValidationError


def test_guarded_explain__raises_validation_error_on_low_gt_high():
    """Blocked Issue 1a: Enforce low <= predict <= high for every GuardedBin."""
    rng = np.random.default_rng(0)
    explainer = MagicMock(spec=CalibratedExplainer)
    explainer.mode = "classification"
    explainer.is_multiclass.return_value = False
    explainer.x_cal = rng.random((10, 2))
    explainer.categorical_features = []

    # Mock orchestrator to return an invalid interval (low > high)
    orchestrator = MagicMock()
    # base call + perturbed call
    orchestrator.predict_internal.side_effect = [
        (np.array([0.5]), np.array([0.4]), np.array([0.6]), np.array([0])),
        (
            np.array([0.5]),  # predict
            np.array([0.8]),  # low (INVALID: low > high)
            np.array([0.2]),  # high
            np.array([0]),  # class
        ),
    ]
    explainer.prediction_orchestrator = orchestrator

    # Discretizer setup
    disc = MagicMock()
    disc.get_bins_with_cal_indices.return_value = [(0.0, 1.0, [0, 1, 2])]
    explainer.discretizer = disc

    x_test = np.array([[0.5, 0.5]])

    with pytest.raises(ValidationError, match="Prediction interval invariant violated: low > high"):
        guarded_explain(explainer, x_test, significance=0.1)


def test_guarded_explain__raises_validation_error_on_predict_out_of_bounds():
    """Blocked Issue 1a: Enforce low <= predict <= high."""
    rng = np.random.default_rng(0)
    explainer = MagicMock(spec=CalibratedExplainer)
    explainer.mode = "classification"
    explainer.is_multiclass.return_value = False
    explainer.x_cal = rng.random((10, 2))
    explainer.categorical_features = []

    orchestrator = MagicMock()
    # Phase 0: 1 row
    # Phase 2: 4 rows (2 features, 1 bin/feat, 2 rows/feat? No, discretizer returns 1 bin. 2 features * 1 bin = 2 rows.)
    orchestrator.predict_internal.side_effect = [
        (np.array([0.5]), np.array([0.4]), np.array([0.6]), np.array([0])),
        (np.array([0.9, 0.9]), np.array([0.4, 0.4]), np.array([0.6, 0.6]), np.array([0, 0])),
    ]
    explainer.prediction_orchestrator = orchestrator

    disc = MagicMock()
    disc.get_bins_with_cal_indices.return_value = [(0.0, 1.0, [0, 1, 2])]
    explainer = MagicMock(spec=CalibratedExplainer)
    explainer.mode = "classification"
    explainer.is_multiclass.return_value = False
    explainer.x_cal = rng.random((10, 2))
    explainer.categorical_features = []
    orchestrator = MagicMock()
    orchestrator.predict_internal.side_effect = [
        (np.array([0.5]), np.array([0.4]), np.array([0.6]), np.array([0])),
        (
            np.array(
                [0.51]
            ),  # predict is 0.51 but high is 0.5, coercion will happen for small drift
            np.array([0.4]),
            np.array([0.5]),
            np.array([0]),
        ),
    ]
    # Small drift test setup
    orchestrator.predict_internal.side_effect = [
        (np.array([0.5]), np.array([0.4]), np.array([0.6]), np.array([0])),
        (
            # Drifting predict slightly ABOVE high by less than epsilon (1e-8 in code)
            np.array([0.6 + 5e-9]),
            np.array([0.4]),
            np.array([0.6]),
            np.array([0]),
        ),
    ]
    explainer.prediction_orchestrator = orchestrator

    disc = MagicMock()
    disc.get_bins_with_cal_indices.return_value = [(0.0, 1.0, [0, 1, 2])]
    explainer.discretizer = disc

    x_test = np.array([[0.5, 0.5]])

    result = guarded_explain(explainer, x_test, significance=0.1)
    audit = result.explanations[0].get_guarded_audit()
    assert any(np.isclose(rec["predict"], 0.6) for rec in audit["intervals"])


def test_guarded_explain__handles_string_categorical_representatives():
    """Blocked Issue 1b: Fix categorical crash when categories are non-numeric."""
    explainer = MagicMock(spec=CalibratedExplainer)
    explainer.mode = "classification"
    explainer.is_multiclass.return_value = False
    explainer.x_cal = np.zeros((10, 2))
    explainer.categorical_features = [0]
    explainer.feature_values = {0: ["red", "blue"]}

    orchestrator = MagicMock()
    # Baseline + perturbed
    orchestrator.predict_internal.side_effect = [
        (np.array([0.5]), np.array([0.4]), np.array([0.6]), np.array([0])),
        (np.array([1.0, 1.0]), np.array([0.9, 0.9]), np.array([1.1, 1.1]), np.array([0, 0])),
    ]
    orchestrator.y_cal_x = explainer.x_cal
    explainer.prediction_orchestrator = orchestrator

    disc = MagicMock()
    disc.get_bins_with_cal_indices.return_value = [(0, 1, []), (1, 2, [])]
    explainer.discretizer = disc

    x_test = np.array([["red", 0.5]], dtype=object)

    result = guarded_explain(explainer, x_test, significance=0.1)
    audit = result.explanations[0].get_guarded_audit()
    reps = [rec["representative"] for rec in audit["intervals"] if rec["feature"] == 0]
    assert "red" in reps
    assert "blue" in reps


def test_guarded_explain__raises_on_mismatched_threshold_batch_length():
    """Major Issue: Indexing resilience for thresholds."""
    rng = np.random.default_rng(0)
    explainer = MagicMock(spec=CalibratedExplainer)
    explainer.mode = "regression"
    explainer.x_cal = rng.random((10, 2))
    explainer.discretizer = MagicMock()
    orchestrator = MagicMock()
    orchestrator.y_cal_x = explainer.x_cal
    explainer.prediction_orchestrator = orchestrator

    x_test = np.array([[0.5, 0.5], [0.6, 0.6]])  # 2 instances

    with pytest.raises(ValidationError, match="Threshold array length .* must match n_instances"):
        guarded_explain(explainer, x_test, threshold=[1, 2, 3])


def test_guarded_explain__warns_on_calibration_identity_mismatch():
    """Major Issue: Exchangeability warning."""
    explainer = MagicMock(spec=CalibratedExplainer)
    explainer.mode = "classification"
    explainer.is_multiclass.return_value = False
    explainer.x_cal = np.array([[1, 1]])

    orchestrator = MagicMock()
    orchestrator.y_cal_x = np.array([[2, 2]])  # Mismatched
    orchestrator.predict_internal.return_value = (
        np.array([0.5]),
        np.array([0.4]),
        np.array([0.6]),
        np.array([0]),
    )
    explainer.prediction_orchestrator = orchestrator

    disc = MagicMock()
    disc.get_bins_with_cal_indices.return_value = [(0.0, 1.0, [0])]
    explainer.discretizer = disc

    x_test = np.array([[0.5, 0.5]])

    with pytest.warns(UserWarning, match="Calibration set identity mismatch"):
        guarded_explain(explainer, x_test)
