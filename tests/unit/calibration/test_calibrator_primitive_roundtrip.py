"""ADR-031 tests for JSON-safe calibrator primitive round-trips."""

from __future__ import annotations

import base64
import hashlib
import json
import pickle
from typing import Any

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from calibrated_explanations import WrapCalibratedExplainer
from calibrated_explanations.calibration.interval_regressor import IntervalRegressor
from calibrated_explanations.calibration.venn_abers import VennAbers
from calibrated_explanations.utils.exceptions import ConfigurationError


def _classification_wrapper() -> tuple[WrapCalibratedExplainer, np.ndarray]:
    rng = np.random.default_rng(42)
    x = rng.random((48, 4))
    y = (x[:, 0] + x[:, 1] > 1.0).astype(int)
    wrapper = WrapCalibratedExplainer(RandomForestClassifier(n_estimators=12, random_state=42))
    wrapper.fit(x[:24], y[:24])
    wrapper.calibrate(x[24:40], y[24:40], seed=42)
    return wrapper, x[40:]


def _regression_wrapper() -> tuple[WrapCalibratedExplainer, np.ndarray, float]:
    rng = np.random.default_rng(43)
    x = rng.random((72, 4))
    y = x[:, 0] * 3.0 - x[:, 1] + rng.normal(0.0, 0.01, size=x.shape[0])
    wrapper = WrapCalibratedExplainer(RandomForestRegressor(n_estimators=16, random_state=43))
    wrapper.fit(x[:36], y[:36])
    wrapper.calibrate(x[36:60], y[36:60], mode="regression", seed=43)
    return wrapper, x[60:], float(np.median(y[36:60]))


def _v1_primitive(calibrator: Any, calibrator_type: str) -> dict[str, Any]:
    payload_bytes = pickle.dumps(calibrator, protocol=pickle.HIGHEST_PROTOCOL)
    return {
        "schema_version": 1,
        "calibrator_type": calibrator_type,
        "checksums": {"sha256": hashlib.sha256(payload_bytes).hexdigest()},
        "payload": {"pickle_b64": base64.b64encode(payload_bytes).decode("ascii")},
    }


def test_venn_abers_to_primitive_v2_is_json_serializable() -> None:
    """VennAbers v2 primitives must be JSON-safe and omit pickle payloads."""
    wrapper, _ = _classification_wrapper()
    calibrator = wrapper.explainer.interval_learner

    primitive = calibrator.to_primitive()

    assert isinstance(calibrator, VennAbers)
    assert primitive["schema_version"] == 2
    assert "pickle_b64" not in json.dumps(primitive)
    assert json.loads(json.dumps(primitive))["schema_version"] == 2


def test_venn_abers_roundtrip_predictions_match() -> None:
    """VennAbers v2 round-trip must preserve calibrated probabilities."""
    wrapper, x_test = _classification_wrapper()
    calibrator = wrapper.explainer.interval_learner

    restored = VennAbers.from_primitive(calibrator.to_primitive())
    restored.reattach_learner(wrapper.learner)

    np.testing.assert_allclose(
        calibrator.predict_proba(x_test[:4]),
        restored.predict_proba(x_test[:4]),
        rtol=1e-5,
        atol=1e-7,
    )


def test_venn_abers_from_primitive_v1_emits_deprecation_warning() -> None:
    """VennAbers schema v1 pickle primitives remain loadable but warn."""
    wrapper, _ = _classification_wrapper()
    calibrator = wrapper.explainer.interval_learner

    with pytest.warns(DeprecationWarning, match="schema_version 1"):
        restored = VennAbers.from_primitive(_v1_primitive(calibrator, "venn_abers"))

    assert isinstance(restored, VennAbers)


def test_venn_abers_from_primitive_unsupported_version_raises() -> None:
    """Unsupported VennAbers primitive versions fail fast."""
    with pytest.raises(ConfigurationError, match="schema_version"):
        VennAbers.from_primitive({"schema_version": 99, "calibrator_type": "venn_abers"})


def test_venn_abers_from_primitive_v2_rejects_invalid_payloads() -> None:
    """VennAbers v2 primitives must validate type and required fields."""
    with pytest.raises(ConfigurationError, match="calibrator_type"):
        VennAbers.from_primitive({"schema_version": 2, "calibrator_type": "wrong", "fields": {}})

    with pytest.raises(ConfigurationError, match="fields"):
        VennAbers.from_primitive({"schema_version": 2, "calibrator_type": "venn_abers"})

    with pytest.raises(ConfigurationError, match="cprobs"):
        VennAbers.from_primitive(
            {
                "schema_version": 2,
                "calibrator_type": "venn_abers",
                "fields": {"y_cal_numeric": [0, 1], "ctargets": [0, 1]},
            }
        )
    with pytest.raises(ConfigurationError, match="y_cal_numeric"):
        VennAbers.from_primitive(
            {
                "schema_version": 2,
                "calibrator_type": "venn_abers",
                "fields": {"cprobs": [0.2, 0.8]},
            }
        )


def test_interval_regressor_to_primitive_v2_is_json_serializable() -> None:
    """IntervalRegressor v2 primitives must be JSON-safe and omit pickle payloads."""
    wrapper, _, _ = _regression_wrapper()
    calibrator = wrapper.explainer.interval_learner

    primitive = calibrator.to_primitive()

    assert isinstance(calibrator, IntervalRegressor)
    assert primitive["schema_version"] == 2
    assert "pickle_b64" not in json.dumps(primitive)
    assert json.loads(json.dumps(primitive))["schema_version"] == 2


def test_interval_regressor_roundtrip_predictions_match() -> None:
    """IntervalRegressor v2 round-trip must preserve probabilistic predictions."""
    wrapper, x_test, threshold = _regression_wrapper()
    calibrator = wrapper.explainer.interval_learner

    restored = IntervalRegressor.from_primitive(calibrator.to_primitive())
    restored.reattach_learner(wrapper.learner, calibrated_explainer=wrapper.explainer)

    np.testing.assert_allclose(
        calibrator.predict_probability(x_test[:4], threshold)[:3],
        restored.predict_probability(x_test[:4], threshold)[:3],
        rtol=1e-5,
        atol=1e-7,
    )


def test_interval_regressor_from_primitive_v1_emits_deprecation_warning() -> None:
    """IntervalRegressor schema v1 pickle primitives remain loadable but warn."""
    wrapper, _, _ = _regression_wrapper()
    calibrator = wrapper.explainer.interval_learner

    with pytest.warns(DeprecationWarning, match="schema_version 1"):
        restored = IntervalRegressor.from_primitive(_v1_primitive(calibrator, "interval_regressor"))

    assert isinstance(restored, IntervalRegressor)


def test_interval_regressor_from_primitive_unsupported_version_raises() -> None:
    """Unsupported IntervalRegressor primitive versions fail fast."""
    with pytest.raises(ConfigurationError, match="schema_version"):
        IntervalRegressor.from_primitive(
            {"schema_version": 99, "calibrator_type": "interval_regressor"}
        )


def test_interval_regressor_from_primitive_v2_rejects_invalid_payloads() -> None:
    """IntervalRegressor v2 primitives must validate type and fields."""
    with pytest.raises(ConfigurationError, match="calibrator_type"):
        IntervalRegressor.from_primitive(
            {"schema_version": 2, "calibrator_type": "wrong", "fields": {}}
        )

    with pytest.raises(ConfigurationError, match="fields"):
        IntervalRegressor.from_primitive(
            {"schema_version": 2, "calibrator_type": "interval_regressor"}
        )
    with pytest.raises(ConfigurationError, match="y_cal_hat"):
        IntervalRegressor.from_primitive(
            {
                "schema_version": 2,
                "calibrator_type": "interval_regressor",
                "fields": {"residual_cal": [0.1], "sigma_cal": [1.0]},
            }
        )
    with pytest.raises(ConfigurationError, match="residual_cal"):
        IntervalRegressor.from_primitive(
            {
                "schema_version": 2,
                "calibrator_type": "interval_regressor",
                "fields": {"y_cal_hat": [1.0], "sigma_cal": [1.0]},
            }
        )
    with pytest.raises(ConfigurationError, match="sigma_cal"):
        IntervalRegressor.from_primitive(
            {
                "schema_version": 2,
                "calibrator_type": "interval_regressor",
                "fields": {"y_cal_hat": [1.0], "residual_cal": [0.1]},
            }
        )
