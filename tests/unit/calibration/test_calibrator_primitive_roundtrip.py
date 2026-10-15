"""ADR-031 tests for JSON-safe calibrator primitive round-trips."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import logging
import pickle
import subprocess
import sys
import textwrap
from typing import Any, Callable

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from calibrated_explanations import WrapCalibratedExplainer
from calibrated_explanations.calibration.interval_regressor import IntervalRegressor
from calibrated_explanations.calibration.venn_abers import VennAbers
from calibrated_explanations.utils.exceptions import ConfigurationError, IncompatibleStateError


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


def test_venn_abers_from_primitive_v1_raises_configuration_error() -> None:
    """VennAbers schema v1 pickle primitives raise ConfigurationError in v1.0.0."""
    wrapper, _ = _classification_wrapper()
    calibrator = wrapper.explainer.interval_learner

    with pytest.raises(ConfigurationError, match="schema_version"):
        VennAbers.from_primitive(_v1_primitive(calibrator, "venn_abers"))


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


def test_interval_regressor_from_primitive_v1_raises_configuration_error() -> None:
    """IntervalRegressor schema v1 pickle primitives raise ConfigurationError in v1.0.0."""
    wrapper, _, _ = _regression_wrapper()
    calibrator = wrapper.explainer.interval_learner

    with pytest.raises(ConfigurationError, match="schema_version"):
        IntervalRegressor.from_primitive(_v1_primitive(calibrator, "interval_regressor"))


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


_CALIBRATOR_CASES: list[tuple[str, Callable[[], Any], type, str]] = [
    ("venn_abers", lambda: _classification_wrapper()[0], VennAbers, "cprobs"),
    ("interval_regressor", lambda: _regression_wrapper()[0], IntervalRegressor, "residual_cal"),
]


@pytest.mark.parametrize(
    ("calibrator_type", "build_wrapper", "calibrator_cls", "field"),
    _CALIBRATOR_CASES,
    ids=[case[0] for case in _CALIBRATOR_CASES],
)
def test_should_produce_identical_checksum_when_primitive_is_built_in_separate_process(
    calibrator_type: str, build_wrapper: Callable[[], Any], calibrator_cls: type, field: str
) -> None:
    """The primitive checksum covers persisted content only, so it is reproducible."""
    script = textwrap.dedent(
        f"""
        import json, warnings
        warnings.simplefilter("ignore")
        from tests.unit.calibration.test_calibrator_primitive_roundtrip import (
            _classification_wrapper, _regression_wrapper,
        )
        build = _classification_wrapper if {calibrator_type!r} == "venn_abers" else _regression_wrapper
        print(json.dumps(build()[0].explainer.interval_learner.to_primitive()["checksums"]))
        """
    )

    other_process = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    in_process = build_wrapper().explainer.interval_learner.to_primitive()["checksums"]

    assert json.loads(other_process.stdout.strip().splitlines()[-1]) == in_process
    assert in_process["algorithm"] == "sha256-canonical-json-v1"


@pytest.mark.parametrize(
    ("calibrator_type", "build_wrapper", "calibrator_cls", "field"),
    _CALIBRATOR_CASES,
    ids=[case[0] for case in _CALIBRATOR_CASES],
)
def test_should_reject_primitive_when_fields_are_modified_after_writing(
    calibrator_type: str, build_wrapper: Callable[[], Any], calibrator_cls: type, field: str
) -> None:
    """A primitive whose persisted fields changed must not load."""
    primitive = json.loads(json.dumps(build_wrapper().explainer.interval_learner.to_primitive()))
    calibrator_cls.from_primitive(copy.deepcopy(primitive))  # untampered primitive loads
    values = primitive["fields"][field]
    while isinstance(values[0], list):  # cprobs is nested
        values = values[0]
    values[0] += 0.01

    with pytest.raises(IncompatibleStateError, match="checksum validation failed"):
        calibrator_cls.from_primitive(primitive)


@pytest.mark.parametrize(
    ("calibrator_type", "build_wrapper", "calibrator_cls", "field"),
    _CALIBRATOR_CASES,
    ids=[case[0] for case in _CALIBRATOR_CASES],
)
def test_should_load_with_warning_when_primitive_has_legacy_untagged_checksum(
    calibrator_type: str,
    build_wrapper: Callable[[], Any],
    calibrator_cls: type,
    field: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Primitives written by <= 1.0.1 (pickle digest, no algorithm tag) still load, visibly."""
    primitive = build_wrapper().explainer.interval_learner.to_primitive()
    primitive["checksums"] = {"sha256": "0" * 64}

    with caplog.at_level(logging.INFO), pytest.warns(UserWarning, match="no verifiable checksum"):
        restored = calibrator_cls.from_primitive(primitive)

    assert isinstance(restored, calibrator_cls)
    assert any("no verifiable checksum" in record.getMessage() for record in caplog.records)


def test_should_reject_primitive_when_checksum_algorithm_is_unknown() -> None:
    """An unknown checksum algorithm fails closed instead of skipping verification."""
    primitive = _classification_wrapper()[0].explainer.interval_learner.to_primitive()
    primitive["checksums"]["algorithm"] = "md5"

    with pytest.raises(IncompatibleStateError, match="unsupported checksum algorithm"):
        VennAbers.from_primitive(primitive)
