from __future__ import annotations

import numpy as np
import pytest

from calibrated_explanations.explanations import explanation as explanation_module


class ExplainerStub:
    def __init__(self):
        self.y_cal = np.array([0.1, 0.9])
        self.mode = "regression"
        self.class_labels = None

    def is_multiclass(self):  # pragma: no cover - not exercised here
        return False


class ContainerStub:
    def __init__(self):
        self.low_high_percentiles = (5, 95)
        self._explainer = ExplainerStub()

    def _get_explainer(self):
        return self._explainer

    def get_low_percentile(self):
        low = self.low_high_percentiles[0]
        if low == -np.inf:
            return -np.inf
        return low / 100.0

    def get_high_percentile(self):
        high = self.low_high_percentiles[1]
        if high == np.inf:
            return np.inf
        return high / 100.0


class SimpleExplanation(explanation_module.CalibratedExplanation):
    def __repr__(self):  # pragma: no cover - trivial
        return "simple"

    def plot(self, filter_top=None, **kwargs):  # pragma: no cover - not needed in tests
        return None

    def add_conjunctions(self, n_top_features=5, max_rule_size=2):  # pragma: no cover - stub
        return self

    def _check_preconditions(self):  # pragma: no cover - stub
        return None

    def _get_rules(self):
        if not hasattr(self, "_rules"):
            self._rules = {"rule": ["r1", "r2"]}
        return self._rules

    def _is_lesser(self, other):  # pragma: no cover - comparison unused in tests
        return False


@pytest.fixture
def simple_explanation():
    container = ContainerStub()
    x = np.array([[0.1, 0.2], [0.3, 0.4]])
    binned = {"weights": np.array([[1, 2], [3, 4]])}
    feature_weights = {"weights": np.array([[0.1, 0.2], [0.3, 0.4]])}
    feature_predict = {"weights": np.array([[0.5, 0.6], [0.7, 0.8]])}
    prediction = {
        "predict": np.array([0.2, 0.6]),
        "low": np.array([0.1, 0.5]),
        "high": np.array([0.4, 0.9]),
    }
    return SimpleExplanation(
        container,
        index=1,
        x=x,
        binned=binned,
        feature_weights=feature_weights,
        feature_predict=feature_predict,
        prediction=prediction,
        y_threshold=None,
        instance_bin=0,
    )


def test_prediction_interval(simple_explanation):
    assert simple_explanation.prediction_interval == (0.5, 0.9)
    assert simple_explanation.predict == pytest.approx(0.6)


def test_rank_features_requires_input(simple_explanation):
    with pytest.raises(ValueError):
        simple_explanation._rank_features()


def test_percentile_helpers():
    assert explanation_module.CalibratedExplanation._normalize_percentile_value(
        95
    ) == pytest.approx(0.95)
    assert explanation_module.CalibratedExplanation._normalize_percentile_value(-np.inf) == -np.inf
    assert explanation_module.CalibratedExplanation._normalize_percentile_value("invalid") is None

    percentiles = (0.05, 0.95)
    assert explanation_module.CalibratedExplanation._compute_confidence_level(
        percentiles
    ) == pytest.approx(0.9)
    assert explanation_module.CalibratedExplanation._compute_confidence_level(
        (-np.inf, 0.9)
    ) == pytest.approx(0.9)
    assert explanation_module.CalibratedExplanation._compute_confidence_level(
        (0.1, np.inf)
    ) == pytest.approx(0.9)


def test_get_percentiles_and_one_sided(simple_explanation):
    assert simple_explanation._get_percentiles() == (0.05, 0.95)
    assert not simple_explanation.is_one_sided()

    simple_explanation.calibrated_explanations.low_high_percentiles = (-np.inf, 95)
    assert simple_explanation.is_one_sided()


def test_to_python_number_handles_nested_arrays():
    arr = np.array([np.array([1.0]), np.array([np.nan])])
    result = explanation_module.CalibratedExplanation._to_python_number(arr)
    assert result == [[1.0], [None]]


def test_to_python_number_scalar_types():
    assert explanation_module.CalibratedExplanation._to_python_number(np.bool_(True)) is True
    assert explanation_module.CalibratedExplanation._to_python_number(np.int64(7)) == 7
    assert (
        explanation_module.CalibratedExplanation._to_python_number(np.float64(np.nan))
        is None
    )
    assert (
        explanation_module.CalibratedExplanation._to_python_number("unchanged")
        == "unchanged"
    )


def test_reset_and_remove_conjunctions(simple_explanation):
    calls = {"count": 0}

    def fake_get_rules():
        calls["count"] += 1
        return {"rule": []}

    simple_explanation._get_rules = fake_get_rules  # type: ignore[assignment]
    simple_explanation._has_rules = True  # type: ignore[attr-defined]
    simple_explanation._has_conjunctive_rules = True  # type: ignore[attr-defined]

    assert simple_explanation.reset() is simple_explanation
    assert calls["count"] == 1

    result = simple_explanation.remove_conjunctions()
    assert result is simple_explanation
    assert not simple_explanation._has_conjunctive_rules  # type: ignore[attr-defined]


class _SimpleDiscretizer:
    def __init__(self):
        self.names = [["feat0 <= 0.5", "feat0 > 0.5"]]

    def discretize(self, x):
        array = np.asarray(x)
        if array.ndim == 1:
            return np.zeros(array.shape[0], dtype=int)
        return np.zeros(array.shape[1], dtype=int)


class TelemetryExplainerStub:
    def __init__(self, mode="classification"):
        self.y_cal = np.array([0.1, 0.9])
        self.mode = mode
        self.class_labels = ["neg", "pos"]
        self.feature_names = ["feat0"]
        self.categorical_features: list[int] = []
        self.categorical_labels = None
        self.num_features = len(self.feature_names)
        self.discretizer = _SimpleDiscretizer()
        self.sample_percentiles = np.array([25, 75])
        self.x_cal = np.array([[0.2], [0.8]])

    def is_multiclass(self):  # pragma: no cover - behaviour not required in tests
        return False

    def assign_threshold(self, threshold):  # pragma: no cover - unused in tests
        return threshold


class TelemetryContainerStub:
    def __init__(self, mode="classification", percentiles=(10, 90)):
        self.low_high_percentiles = percentiles
        self._explainer = TelemetryExplainerStub(mode=mode)
        self.features_to_ignore: set[int] = set()

    def _get_explainer(self):
        return self._explainer

    def get_low_percentile(self):
        low = self.low_high_percentiles[0]
        if low == -np.inf:
            return -np.inf
        return low / 100.0

    def get_high_percentile(self):
        high = self.low_high_percentiles[1]
        if high == np.inf:
            return np.inf
        return high / 100.0


class TelemetryExplanation(explanation_module.CalibratedExplanation):
    def __init__(self, *, mode="classification", y_threshold=None, percentiles=(10, 90)):
        container = TelemetryContainerStub(mode=mode, percentiles=percentiles)
        x = np.array([0.2])
        binned = {"rule_values": np.array([[[(0.0, 1.0)]]], dtype=object)}
        feature_weights = {
            "predict": np.array([[0.3]]),
            "low": np.array([[0.1]]),
            "high": np.array([[0.5]]),
        }
        feature_predict = {
            "predict": np.array([[0.7]]),
            "low": np.array([[0.6]]),
            "high": np.array([[0.8]]),
        }
        prediction = {
            "predict": np.array([0.65]),
            "low": np.array([0.55]),
            "high": np.array([0.75]),
            "classes": np.array([1]),
        }
        super().__init__(
            container,
            index=0,
            x=x,
            binned=binned,
            feature_weights=feature_weights,
            feature_predict=feature_predict,
            prediction=prediction,
            y_threshold=y_threshold,
            instance_bin=None,
        )
        self._rules = {
            "base_predict": [self.prediction["predict"]],
            "base_predict_low": [self.prediction["low"]],
            "base_predict_high": [self.prediction["high"]],
            "predict": [self.feature_predict["predict"][0]],
            "predict_low": [self.feature_predict["low"][0]],
            "predict_high": [self.feature_predict["high"][0]],
            "weight": [self.feature_weights["predict"][0]],
            "weight_low": [self.feature_weights["low"][0]],
            "weight_high": [self.feature_weights["high"][0]],
            "value": ["0.12"],
            "rule": ["feat0 >= 0.5"],
            "feature": [0],
            "feature_value": [np.array([0.0, 1.0])],
            "is_conjunctive": [False],
            "classes": self.prediction["classes"],
        }

    def __repr__(self):  # pragma: no cover - representation unused here
        return "telemetry"

    def plot(self, filter_top=None, **kwargs):  # pragma: no cover - plotting not exercised
        return None

    def add_conjunctions(self, n_top_features=5, max_rule_size=2):  # pragma: no cover - stub
        return self

    def _check_preconditions(self):  # pragma: no cover - no-op for tests
        return None

    def _get_rules(self):
        return self._rules

    def _is_lesser(self, rule_boundary, instance_value):  # pragma: no cover - unused
        return instance_value < rule_boundary


@pytest.fixture
def telemetry_explanation():
    return TelemetryExplanation()


def test_normalize_threshold_value_handles_structures(telemetry_explanation):
    telemetry_explanation.y_threshold = np.array([0.4, 0.6])
    assert telemetry_explanation._normalize_threshold_value() == [0.4, 0.6]

    telemetry_explanation.y_threshold = (0.3,)
    assert telemetry_explanation._normalize_threshold_value() == [0.3]

    telemetry_explanation.y_threshold = []
    assert telemetry_explanation._normalize_threshold_value() is None

    telemetry_explanation.y_threshold = 0.5
    assert telemetry_explanation._normalize_threshold_value() == 0.5

    telemetry_explanation.y_threshold = None
    assert telemetry_explanation._normalize_threshold_value() is None


def test_build_uncertainty_payload_with_percentiles(telemetry_explanation):
    payload = telemetry_explanation._build_uncertainty_payload(
        value=0.6,
        low=0.4,
        high=0.8,
        representation="percentile",
        percentiles=(0.1, 0.9),
    )

    assert payload["calibrated_value"] == 0.6
    assert payload["legacy_interval"] == [0.4, 0.8]
    assert payload["raw_percentiles"] == [0.1, 0.9]
    assert payload["confidence_level"] == pytest.approx(0.8)


def test_build_instance_uncertainty_probabilistic(telemetry_explanation):
    payload = telemetry_explanation._build_instance_uncertainty()
    assert payload["representation"] == "venn_abers"
    assert payload["raw_percentiles"] is None
    assert payload["legacy_interval"] == [0.55, 0.75]


def test_build_instance_uncertainty_thresholded():
    explanation = TelemetryExplanation(mode="regression", y_threshold=(0.2, 0.8))
    payload = explanation._build_instance_uncertainty()
    assert payload["representation"] == "threshold"
    assert payload["threshold"] == [0.2, 0.8]
    assert payload["raw_percentiles"] is None


def test_build_instance_uncertainty_percentiles():
    explanation = TelemetryExplanation(mode="regression")
    payload = explanation._build_instance_uncertainty()
    assert payload["representation"] == "percentile"
    assert payload["raw_percentiles"] == [0.1, 0.9]
    assert payload["confidence_level"] == pytest.approx(0.8)


def test_safe_feature_name_and_condition_helpers(telemetry_explanation):
    assert telemetry_explanation._safe_feature_name(0) == "feat0"
    assert telemetry_explanation._safe_feature_name(5) == "5"
    assert telemetry_explanation._safe_feature_name("custom") == "custom"

    assert telemetry_explanation._convert_condition_value("-inf", 0) == -np.inf
    assert telemetry_explanation._convert_condition_value(" +inf ", 0) == np.inf
    assert telemetry_explanation._convert_condition_value(" 1.5 ", 0) == pytest.approx(1.5)
    assert telemetry_explanation._convert_condition_value(None, 0.7) == pytest.approx(0.7)
    assert (
        telemetry_explanation._convert_condition_value("not-a-number", 0)
        == "not-a-number"
    )

    operator, value = telemetry_explanation._parse_condition("feat0", "feat0 = 1.0")
    assert operator == "=="
    assert value == "1.0"

    operator, value = telemetry_explanation._parse_condition("feat0", "something else")
    assert operator == "raw"
    assert value == "something else"

    condition = telemetry_explanation._build_condition_payload(
        0,
        "feat0 >= 0.5",
        telemetry_explanation._rules["feature_value"][0],
        telemetry_explanation._rules["value"][0],
    )
    assert condition == {
        "feature": "feat0",
        "operator": ">=",
        "value": pytest.approx(0.5),
        "text": "feat0 >= 0.5",
    }

    fallback = telemetry_explanation._build_condition_payload(
        0,
        "",
        telemetry_explanation._rules["feature_value"][0],
        "display",
    )
    assert fallback["operator"] == "raw"
    assert fallback["value"] == "display"


def test_build_factual_rules_payload(telemetry_explanation):
    payloads = telemetry_explanation._build_factual_rules_payload()
    assert len(payloads) == 1
    payload = payloads[0]
    assert payload["kind"] == "factual"
    assert payload["feature"] == "feat0"
    assert payload["baseline_prediction"] == pytest.approx(0.65)
    assert payload["uncertainty"]["representation"] == "venn_abers"
    assert payload["condition"]["operator"] == ">="


def test_build_rules_payload_and_telemetry_export(telemetry_explanation):
    rules = telemetry_explanation.build_rules_payload()
    assert len(rules) == 1
    telemetry = telemetry_explanation.to_telemetry()
    assert telemetry["uncertainty"]["representation"] == "venn_abers"
    assert telemetry["rules"] == rules
