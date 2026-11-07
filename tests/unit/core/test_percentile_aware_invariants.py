"""Percentile-aware ``low <= predict <= high`` checks for CPS regression (#247).

Open interval endpoints are unbounded, and invariant checks are scoped to the
sides the requested percentiles imply (ADR-021).
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from calibrated_explanations import WrapCalibratedExplainer
from calibrated_explanations.explanations.explanation import CalibratedExplanation
from calibrated_explanations.explanations.explanations import CalibratedExplanations
from calibrated_explanations.plugins.explanations import (
    ExplanationBatch,
    validate_explanation_batch,
)
from calibrated_explanations.utils.exceptions import ValidationError
from calibrated_explanations.utils.helper import median_containment_sides

CONTAINMENT = "predict not in [low, high]"
INVERTED = "low > high"


class DummyContainer(CalibratedExplanations):
    """Concrete container type for batch validation tests."""


class DummyExplanation(CalibratedExplanation):
    """Concrete explanation type for batch validation tests."""


@pytest.fixture(scope="module")
def fitted():
    """Return a calibrated linear wrapper and its calibration targets."""
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 10, size=(300, 1))
    y = x[:, 0] + rng.normal(scale=1.0, size=300)
    wrapper = WrapCalibratedExplainer(LinearRegression())
    wrapper.fit(x[:150], y[:150])
    wrapper.calibrate(x[150:], y[150:])
    learner = wrapper.learner
    y_cal = y[150:]

    def x_for(target: float) -> np.ndarray:
        """Return the input whose point prediction equals ``target``."""
        return np.array([[(target - learner.intercept_) / learner.coef_[0]]])

    return wrapper, y_cal, x_for


def _invariant_messages(fn) -> list[str]:
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        fn()
    return [str(w.message) for w in record if "invariant violated" in str(w.message)]


@pytest.mark.parametrize(
    ("percentiles", "expected"),
    [
        ((5, 95), (True, True)),
        ((40, 60), (True, True)),
        ((-np.inf, 90), (False, True)),
        ((10, np.inf), (True, False)),
        ((-np.inf, 40), (False, False)),
        ((60, np.inf), (False, False)),
        ((60, 95), (False, True)),
        ((60, 80), (False, True)),
        ((10, 80), (True, True)),
        ((5, 40), (True, False)),
        (None, (True, True)),
    ],
)
def test_median_containment_sides_follow_requested_percentiles(percentiles, expected):
    assert median_containment_sides(percentiles) == expected


@pytest.mark.parametrize(
    ("percentiles", "offset"),
    [((10, np.inf), 0.5), ((-np.inf, 90), -0.5)],
)
def test_one_sided_90_interval_beyond_calibration_range_should_not_warn(
    fitted, percentiles, offset
):
    wrapper, y_cal, x_for = fitted
    # Median just beyond the calibration-target range on the open side.
    edge = np.max(y_cal) if offset > 0 else np.min(y_cal)
    x_test = x_for(edge + offset)

    messages = _invariant_messages(
        lambda: wrapper.predict(x_test, uq_interval=True, low_high_percentiles=percentiles)
    )
    pred, (low, high) = wrapper.predict(x_test, uq_interval=True, low_high_percentiles=percentiles)

    assert messages == []
    if offset > 0:
        assert np.isposinf(high[0])
        assert low[0] <= pred[0]
    else:
        assert np.isneginf(low[0])
        assert pred[0] <= high[0]


@pytest.mark.parametrize("percentiles", [(-np.inf, 40), (60, 80), (60, np.inf)])
def test_interval_excluding_median_is_valid_when_percentiles_exclude_median(fitted, percentiles):
    wrapper, y_cal, x_for = fitted
    x_test = x_for(float(np.median(y_cal)))

    messages = _invariant_messages(
        lambda: wrapper.predict(x_test, uq_interval=True, low_high_percentiles=percentiles)
    )
    pred, (low, high) = wrapper.predict(x_test, uq_interval=True, low_high_percentiles=percentiles)

    assert messages == []
    if percentiles == (60, 80):
        assert pred[0] < low[0] <= high[0]
    else:
        assert not (low[0] <= pred[0] <= high[0])


@pytest.mark.parametrize(
    "percentiles",
    [
        (5, 95),
        (40, 60),
        (10, 80),
        (60, 80),
        (-np.inf, 90),
        (10, np.inf),
        (-np.inf, 40),
        (60, np.inf),
    ],
)
@pytest.mark.parametrize("method", ["explain_factual", "explore_alternatives"])
def test_explanations_should_not_emit_false_containment_warning(fitted, percentiles, method):
    wrapper, y_cal, x_for = fitted
    x_test = np.vstack([x_for(float(np.median(y_cal))), x_for(np.max(y_cal) + 0.5)])

    messages = _invariant_messages(
        lambda: getattr(wrapper, method)(x_test, low_high_percentiles=percentiles)
    )

    assert not [m for m in messages if CONTAINMENT in m]


def test_off_centre_open_interval_serializes_as_json_when_median_is_excluded(fitted):
    wrapper, y_cal, x_for = fitted
    explanations = wrapper.explain_factual(
        x_for(float(np.max(y_cal) + 0.5)),
        low_high_percentiles=(-np.inf, 40),
    )

    payload = explanations.to_json()

    assert payload["explanations"][0]["prediction"]["low"] == "-inf"
    json.dumps(payload, allow_nan=False)


def test_open_endpoint_does_not_invert_interval_when_prediction_exceeds_calibration_range(fitted):
    wrapper, y_cal, x_for = fitted
    # Far beyond the calibration range; the open upper endpoint remains unbounded.
    x_test = x_for(np.max(y_cal) + 20)

    messages = _invariant_messages(
        lambda: wrapper.predict(x_test, uq_interval=True, low_high_percentiles=(10, np.inf))
    )

    assert messages == []


@pytest.mark.parametrize("percentiles", [(5, 95), (40, 60), (-np.inf, 90), (10, np.inf)])
def test_orchestrator_still_flags_violated_finite_bound(fitted, percentiles):
    wrapper, _, _ = fitted
    orchestrator = wrapper.explainer.prediction_orchestrator
    # predict outside every finite bound the percentiles require
    result = (np.array([100.0, -100.0]), np.array([0.0, 0.0]), np.array([1.0, 1.0]), None)

    messages = _invariant_messages(
        lambda: orchestrator.validate_prediction_result(
            result, threshold=None, low_high_percentiles=percentiles
        )
    )

    assert any(CONTAINMENT in m for m in messages)


def test_orchestrator_thresholded_regression_keeps_full_check(fitted):
    wrapper, _, _ = fitted
    orchestrator = wrapper.explainer.prediction_orchestrator
    result = (np.array([0.9]), np.array([0.1]), np.array([0.5]), None)

    messages = _invariant_messages(
        lambda: orchestrator.validate_prediction_result(
            result, threshold=5.0, low_high_percentiles=(-np.inf, 40)
        )
    )

    assert any(CONTAINMENT in m for m in messages)


def _regression_batch(payload: dict) -> ExplanationBatch:
    return ExplanationBatch(
        container_cls=DummyContainer,
        explanation_cls=DummyExplanation,
        instances=[{"prediction": payload}],
        collection_metadata={"task": "regression", "mode": "test"},
    )


def test_batch_validator_accepts_median_outside_unbounded_endpoint():
    # (-inf, 90): the open low endpoint does not bind the median.
    batch = _regression_batch({"predict": -5.0, "low": -3.0, "high": 0.0})

    validate_explanation_batch(
        batch,
        expected_task="regression",
        expected_mode="test",
        low_high_percentiles=(-np.inf, 90),
    )

    with pytest.raises(ValidationError, match="Prediction invariant violated"):
        validate_explanation_batch(batch, expected_task="regression", expected_mode="test")


def test_batch_validator_still_rejects_violated_finite_bound():
    batch = _regression_batch({"predict": 5.0, "low": -3.0, "high": 0.0})

    with pytest.raises(ValidationError, match="Prediction invariant violated"):
        validate_explanation_batch(
            batch,
            expected_task="regression",
            expected_mode="test",
            low_high_percentiles=(-np.inf, 90),
        )
