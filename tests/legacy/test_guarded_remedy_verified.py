"""Legacy guarded remedy tests for fail-fast alignment and invariant handling."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer
from calibrated_explanations.core.exceptions import ValidationError
from calibrated_explanations.core.explain._guarded_explain import (
    _finalise_group,
    _merge_adjacent_bins,
    guarded_explain,
)
from calibrated_explanations.explanations.guarded_explanation import GuardedBin
from calibrated_explanations.utils.exceptions import ConfigurationError, NotFittedError
from evaluation.guarded.scenario_a_guarded_vs_standard import _rule_rows_from_guarded_audit


def make_guarded_mock_explainer(
    *,
    mode: str = "classification",
    x_cal: np.ndarray | None = None,
    categorical_features: list[int] | None = None,
) -> tuple[MagicMock, MagicMock, MagicMock]:
    rng = np.random.default_rng(0)
    explainer = MagicMock(spec=CalibratedExplainer)
    explainer.mode = mode
    explainer.is_multiclass.return_value = False
    explainer.x_cal = np.asarray(x_cal if x_cal is not None else rng.random((10, 2)))
    explainer.categorical_features = list(categorical_features or [])
    explainer.feature_values = {}

    discretizer = MagicMock()
    discretizer.get_bins_with_cal_indices.return_value = [(0.0, 1.0, [0, 1, 2])]
    explainer.discretizer = discretizer

    # Defaults: initialized and non-fast so existing tests are not broken.
    explainer.initialized = True
    explainer.is_fast.return_value = False

    orchestrator = MagicMock()
    orchestrator.get_interval_calibration_features.return_value = explainer.x_cal
    explainer.prediction_orchestrator = orchestrator
    return explainer, orchestrator, discretizer


def test_should_raise_validation_error_when_guarded_bin_has_low_greater_than_high():
    """Guarded bins must respect ADR-021 interval ordering."""
    explainer, orchestrator, _ = make_guarded_mock_explainer()
    orchestrator.predict_internal.side_effect = [
        (np.array([0.5]), np.array([0.4]), np.array([0.6]), np.array([0])),
        (
            np.array([0.5, 0.5]),
            np.array([0.8, 0.8]),
            np.array([0.2, 0.2]),
            np.array([0, 0]),
        ),
    ]

    with pytest.raises(ValidationError, match="Prediction interval invariant violated: low > high"):
        guarded_explain(explainer, np.array([[0.5, 0.5]]), significance=0.1)


def test_should_coerce_small_predict_drift_back_inside_guarded_interval():
    """Small floating-point drift should be coerced back into the structural interval."""
    explainer, orchestrator, _ = make_guarded_mock_explainer()
    orchestrator.predict_internal.side_effect = [
        (np.array([0.5]), np.array([0.4]), np.array([0.6]), np.array([0])),
        (
            np.array([0.6 + 5e-9, 0.6 + 5e-9]),
            np.array([0.4, 0.4]),
            np.array([0.6, 0.6]),
            np.array([0, 0]),
        ),
    ]

    result = guarded_explain(explainer, np.array([[0.5, 0.5]]), significance=0.1)
    audit = result.explanations[0].get_guarded_audit()
    assert any(np.isclose(rec["predict"], 0.6) for rec in audit["intervals"])


def test_should_handle_string_categorical_representatives_when_guarded_candidates_are_built():
    """Representative-point selection must preserve non-numeric categorical values."""
    explainer, orchestrator, discretizer = make_guarded_mock_explainer(
        categorical_features=[0],
        x_cal=np.zeros((10, 2), dtype=object),
    )
    explainer.feature_values = {0: ["red", "blue"]}
    discretizer.get_bins_with_cal_indices.return_value = [(0, 1, []), (1, 2, [])]
    orchestrator.predict_internal.side_effect = [
        (np.array([0.5]), np.array([0.4]), np.array([0.6]), np.array([0])),
        (
            np.array([1.0, 1.0, 1.0, 1.0]),
            np.array([0.9, 0.9, 0.9, 0.9]),
            np.array([1.1, 1.1, 1.1, 1.1]),
            np.array([0, 0, 0, 0]),
        ),
    ]

    result = guarded_explain(explainer, np.array([["red", 0.5]], dtype=object), significance=0.1)
    audit = result.explanations[0].get_guarded_audit()
    representatives = [rec["representative"] for rec in audit["intervals"] if rec["feature"] == 0]
    assert "red" in representatives
    assert "blue" in representatives


def test_should_raise_validation_error_when_threshold_batch_length_is_mismatched():
    """Threshold arrays must align with the explained batch size."""
    explainer, _, _ = make_guarded_mock_explainer(mode="regression")

    with pytest.raises(ValidationError, match="Threshold array length .* must match n_instances"):
        guarded_explain(explainer, np.array([[0.5, 0.5], [0.6, 0.6]]), threshold=[1, 2, 3])


def test_should_raise_validation_error_when_backend_calibration_features_differ():
    """Guarded filtering must hard-fail when backend and explainer calibration features diverge."""
    explainer, orchestrator, _ = make_guarded_mock_explainer(x_cal=np.array([[1, 1]]))
    orchestrator.get_interval_calibration_features.return_value = np.array([[2, 2]])

    with pytest.raises(
        ValidationError,
        match="prediction backend to use the same calibration features as explainer.x_cal",
    ):
        guarded_explain(explainer, np.array([[0.5, 0.5]]))


def test_should_raise_validation_error_when_backend_calibration_features_are_unavailable():
    """Guarded entrypoints require backend calibration-feature provenance."""
    explainer, orchestrator, _ = make_guarded_mock_explainer()
    orchestrator.get_interval_calibration_features.return_value = None

    with pytest.raises(ValidationError, match="require interval calibration features"):
        guarded_explain(explainer, np.array([[0.5, 0.5]]))


def test_should_raise_validation_error_when_perturbed_prediction_batch_is_short():
    """Guarded batching must fail instead of repairing short predictor output."""
    explainer, orchestrator, _ = make_guarded_mock_explainer()
    orchestrator.predict_internal.side_effect = [
        (np.array([0.5]), np.array([0.4]), np.array([0.6]), np.array([0])),
        (np.array([0.5]), np.array([0.4]), np.array([0.6]), np.array([0])),
    ]

    with pytest.raises(ValidationError, match="Guarded perturbed prediction batch length mismatch"):
        guarded_explain(explainer, np.array([[0.5, 0.5]]), significance=0.1)


def test_should_raise_configuration_error_when_guarded_called_on_fast_explainer():
    """Fast explainers must be rejected at guarded entrypoints with ConfigurationError."""
    explainer, _, _ = make_guarded_mock_explainer()
    # Override to simulate a fast explainer.
    explainer.is_fast.return_value = True

    with pytest.raises(ConfigurationError, match="not supported for fast explainers"):
        guarded_explain(explainer, np.array([[0.5, 0.5]]))


def test_should_raise_not_fitted_error_when_guarded_called_on_uninitialized_explainer():
    """Uninitialized explainers must raise NotFittedError, not a calibration alignment error."""
    explainer, _, _ = make_guarded_mock_explainer()
    # Override to simulate an uninitialized explainer.
    explainer.initialized = False

    with pytest.raises(NotFittedError, match="initialized before calling guarded entrypoints"):
        guarded_explain(explainer, np.array([[0.5, 0.5]]))


def test_should_use_single_median_probe_for_guard_conformity():
    """All bins should be probed via the single median representative."""

    class FakeGuard:
        def p_values(self, x: np.ndarray) -> np.ndarray:
            value = float(np.asarray(x)[0, 0])
            if np.isclose(value, 5.5):
                return np.array([0.7])
            raise AssertionError(f"Unexpected probe value: {value}")

    explainer, orchestrator, discretizer = make_guarded_mock_explainer(
        x_cal=np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0]])
    )
    discretizer.get_bins_with_cal_indices.return_value = [(0.0, 10.0, list(range(10)))]
    orchestrator.predict_internal.side_effect = [
        (np.array([0.5]), np.array([0.4]), np.array([0.6]), np.array([0])),
        (np.array([0.55]), np.array([0.45]), np.array([0.65]), np.array([0])),
    ]

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "calibrated_explanations.utils.distribution_guard.InDistributionGuard",
            lambda *args, **kwargs: FakeGuard(),
        )
        result = guarded_explain(explainer, np.array([[5.5]]), significance=0.6)

    rec = result.explanations[0].get_guarded_audit()["intervals"][0]
    assert rec["conforming"] is True
    assert rec["representative"] == pytest.approx(5.5)
    assert rec["p_value"] == pytest.approx(0.7)
    assert rec["lower"] == pytest.approx(0.0)
    assert rec["upper"] == pytest.approx(10.0)


def test_should_keep_sparse_bins_on_single_median_probe():
    """Sparse bins should still use the median representative as the guard probe."""

    class FakeGuard:
        def p_values(self, x: np.ndarray) -> np.ndarray:
            value = float(np.asarray(x)[0, 0])
            if np.isclose(value, 3.0):
                return np.array([0.2])
            raise AssertionError(f"Unexpected sparse probe value: {value}")

    explainer, orchestrator, discretizer = make_guarded_mock_explainer(
        x_cal=np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    )
    discretizer.get_bins_with_cal_indices.return_value = [(0.0, 5.0, list(range(5)))]
    orchestrator.predict_internal.side_effect = [
        (np.array([0.5]), np.array([0.4]), np.array([0.6]), np.array([0])),
        (np.array([0.55]), np.array([0.45]), np.array([0.65]), np.array([0])),
    ]

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "calibrated_explanations.utils.distribution_guard.InDistributionGuard",
            lambda *args, **kwargs: FakeGuard(),
        )
        result = guarded_explain(explainer, np.array([[3.0]]), significance=0.3)

    rec = result.explanations[0].get_guarded_audit()["intervals"][0]
    assert rec["conforming"] is False
    assert rec["representative"] == pytest.approx(3.0)
    assert rec["lower"] == pytest.approx(0.0)
    assert rec["upper"] == pytest.approx(5.0)


def test_should_keep_original_bins_when_merge_recheck_fails():
    """Merged intervals must revert to the original bins when the median re-check fails."""

    class FakeGuard:
        def p_values(self, x: np.ndarray) -> np.ndarray:
            value = float(np.asarray(x)[0, 0])
            if np.isclose(value, 5.5):
                return np.array([0.2])
            raise AssertionError(f"Unexpected merge probe value: {value}")

    explainer, orchestrator, _ = make_guarded_mock_explainer(
        x_cal=np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0]])
    )
    group = [
        GuardedBin(0.0, 5.0, 3.0, 0.5, 0.4, 0.6, True, 0.9, False),
        GuardedBin(5.0, 10.0, 8.0, 0.6, 0.5, 0.7, True, 0.9, False),
    ]

    result = _finalise_group(
        group,
        current_value=4.0,
        explainer=explainer,
        x_instance=np.array([4.0]),
        feature_idx=0,
        threshold=None,
        low_high_percentiles=(5, 95),
        mondrian_bins=None,
        guard=FakeGuard(),
        adjusted_sig=0.5,
    )

    assert result == group
    orchestrator.predict_internal.assert_not_called()


def test_should_iteratively_expand_factual_rule_until_no_more_adjacent_bins_can_merge():
    """Factual merge should grow from the factual bin and stop at the first failing expansion."""

    class FakeGuard:
        def p_values(self, x: np.ndarray) -> np.ndarray:
            value = float(np.asarray(x)[0, 0])
            if np.isclose(value, 1.5):
                return np.array([0.9])
            if np.isclose(value, 2.0):
                return np.array([0.2])
            if np.isclose(value, 3.5):
                return np.array([0.9])
            raise AssertionError(f"Unexpected merge probe value: {value}")

    explainer, orchestrator, _ = make_guarded_mock_explainer(
        x_cal=np.array([[1.0], [2.0], [3.0], [4.0]])
    )
    orchestrator.predict_internal.side_effect = [
        (np.array([0.55]), np.array([0.45]), np.array([0.65]), np.array([0])),
        (np.array([0.57]), np.array([0.47]), np.array([0.67]), np.array([0])),
    ]
    bins = [
        GuardedBin(0.0, 1.0, 0.5, 0.50, 0.40, 0.60, True, 0.9, False),
        GuardedBin(1.0, 2.0, 1.5, 0.52, 0.42, 0.62, True, 0.9, True),
        GuardedBin(2.0, 3.0, 2.5, 0.54, 0.44, 0.64, True, 0.9, False),
        GuardedBin(3.0, 4.0, 3.5, 0.56, 0.46, 0.66, True, 0.9, False),
    ]

    result = _merge_adjacent_bins(
        bins,
        current_value=1.5,
        explainer=explainer,
        x_instance=np.array([1.5]),
        feature_idx=0,
        threshold=None,
        low_high_percentiles=(5, 95),
        mondrian_bins=None,
        mode="factual",
        guard=FakeGuard(),
        adjusted_sig=0.5,
    )

    assert len(result) == 2
    assert result[0].lower == pytest.approx(0.0)
    assert result[0].upper == pytest.approx(2.0)
    assert result[0].is_factual is True
    assert result[0].is_merged is True
    assert result[1].lower == pytest.approx(2.0)
    assert result[1].upper == pytest.approx(4.0)
    assert result[1].is_merged is True
    assert orchestrator.predict_internal.call_count == 2


def test_should_leave_nonrejected_but_nonadjacent_to_factual_rule_unmerged_when_barrier_fails():
    """Factual merge should not skip across a failing adjacent expansion to absorb farther bins."""

    class FakeGuard:
        def p_values(self, x: np.ndarray) -> np.ndarray:
            value = float(np.asarray(x)[0, 0])
            if np.isclose(value, 1.0):
                return np.array([0.2])
            if np.isclose(value, 2.5):
                return np.array([0.9])
            raise AssertionError(f"Unexpected merge probe value: {value}")

    explainer, orchestrator, _ = make_guarded_mock_explainer(
        x_cal=np.array([[0.5], [1.0], [2.0], [3.0]])
    )
    orchestrator.predict_internal.return_value = (
        np.array([0.58]),
        np.array([0.48]),
        np.array([0.68]),
        np.array([0]),
    )
    bins = [
        GuardedBin(0.0, 1.0, 0.5, 0.50, 0.40, 0.60, True, 0.9, True),
        GuardedBin(1.0, 2.0, 1.5, 0.52, 0.42, 0.62, True, 0.9, False),
        GuardedBin(2.0, 3.0, 2.5, 0.54, 0.44, 0.64, True, 0.9, False),
    ]

    result = _merge_adjacent_bins(
        bins,
        current_value=0.5,
        explainer=explainer,
        x_instance=np.array([0.5]),
        feature_idx=0,
        threshold=None,
        low_high_percentiles=(5, 95),
        mondrian_bins=None,
        mode="factual",
        guard=FakeGuard(),
        adjusted_sig=0.5,
    )

    assert [(gbin.lower, gbin.upper) for gbin in result] == [(0.0, 1.0), (1.0, 3.0)]
    assert result[0].is_factual is True
    assert result[0].is_merged is False
    assert result[1].is_merged is True
    orchestrator.predict_internal.assert_called_once()


def test_should_evaluate_scenario_a_guarded_rows_with_interval_semantics():
    """Scenario A should score the guarded rule using lower/upper."""
    rows = _rule_rows_from_guarded_audit(
        {
            "intervals": [
                {
                    "feature": 0,
                    "representative": 2.0,
                    "lower": 1.0,
                    "upper": 2.0,
                    "p_value": 0.7,
                    "predict": 0.6,
                    "low": 0.5,
                    "high": 0.7,
                    "condition": "1 < x0 <= 2",
                    "emitted": True,
                }
            ]
        },
        dataset="scenario_a",
        seed=0,
        model_name="rf",
        instance_id=0,
        mode="factual",
        method="guarded",
        cfg=type("Cfg", (), {"significance": 0.1, "n_neighbors": 5, "merge_adjacent": False})(),
        x_instance=np.array([0.0, 5.5]),
        runtime_ms=1.0,
    )

    assert rows[0]["representative_value"] == pytest.approx(2.0)
    assert rows[0]["plausibility_flag"] is False
