from __future__ import annotations

import numpy as np
import pytest
from pandas import Categorical

from calibrated_explanations.explanations import explanation as explanation_module


class DiscretizerStub:
    def __init__(self):
        self.names = [
            ["f0 <= 0.2", "f0 > 0.2"],
            ["f1 <= 0.2", "f1 > 0.2"],
            ["f2 <= 0.2", "f2 > 0.2"],
        ]

    def discretize(self, x):  # pragma: no cover - simple passthrough
        # mimic the discretizer contract where each column maps to a bin index
        return np.arange(x.shape[-1], dtype=int)


class ExplainerStub:
    def __init__(self, mode="regression", y_cal=None):
        self.y_cal = np.array([0.1, 0.9]) if y_cal is None else y_cal
        self.mode = mode
        self.class_labels = None
        self.feature_names = ["f0", "f1", "f2"]
        self.categorical_features: list[int] = []
        self.categorical_labels = None
        self.feature_values = [np.array([0, 1]), np.array([0, 1]), np.array([0, 1])]
        self.sample_percentiles = [25, 75]
        self.num_features = 2
        self.discretizer = DiscretizerStub()
        self.x_cal = np.array([[0.1, 0.3], [0.2, 0.4]])
        self.rule_boundaries = lambda _instance: []

    def is_multiclass(self):  # pragma: no cover - not exercised here
        return False

    def _predict(self, data, **_kwargs):
        rows = data.shape[0]
        predict = np.ones((rows, 1))
        low = np.full((rows, 1), 0.5)
        high = np.full((rows, 1), 1.5)
        return predict, low, high, np.zeros(rows)


class ContainerStub:
    def __init__(self, *, mode="regression", y_cal=None):
        self.low_high_percentiles = (5, 95)
        self._explainer = ExplainerStub(mode=mode, y_cal=y_cal)
        self.features_to_ignore = []
        self.explanations = []

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

    def build_rules_payload(self):
        rules = getattr(self, "rules", None) or getattr(self, "_rules", None) or {}
        if isinstance(rules, dict) and "feature" in rules:
            return explanation_module.FactualExplanation.build_rules_payload(self)
        prediction_interval = self._build_interval(
            self.prediction.get("low"),
            self.prediction.get("high"),
        )
        return {
            "core": {
                "kind": "simple",
                "prediction": {
                    "value": explanation_module.CalibratedExplanation._to_python_number(
                        self.prediction.get("predict")
                    ),
                    "uncertainty_interval": prediction_interval,
                },
                "feature_rules": [],
            },
            "metadata": {"feature_rules": []},
        }


def _make_explanation(
    *,
    cls=SimpleExplanation,
    mode: str = "regression",
    threshold=None,
    y_cal=None,
):
    container = ContainerStub(mode=mode, y_cal=y_cal)
    x = np.array([[0.1, 0.2], [0.3, 0.4]])
    rule_values = np.empty((2, 2), dtype=object)
    rule_values[0, 0] = np.array([[0.1]])
    rule_values[0, 1] = np.array([[0.2]])
    rule_values[1, 0] = np.array([[0.3]])
    rule_values[1, 1] = np.array([[0.4]])
    binned = {
        "predict": np.array([[0.2, 0.3], [0.4, 0.5]]),
        "low": np.array([[0.1, 0.2], [0.3, 0.4]]),
        "high": np.array([[0.3, 0.4], [0.5, 0.6]]),
        "rule_values": rule_values,
    }
    feature_weights = {
        "predict": np.array([[0.1, 0.2], [0.3, 0.4]]),
        "low": np.array([[0.05, 0.1], [0.15, 0.2]]),
        "high": np.array([[0.15, 0.25], [0.35, 0.45]]),
    }
    feature_predict = {
        "predict": np.array([[0.5, 0.6], [0.7, 0.8]]),
        "low": np.array([[0.4, 0.5], [0.6, 0.7]]),
        "high": np.array([[0.6, 0.7], [0.8, 0.9]]),
    }
    prediction = {
        "predict": np.array([0.2, 0.6]),
        "low": np.array([0.1, 0.5]),
        "high": np.array([0.4, 0.9]),
        "classes": np.array([1, 1]),
    }
    return cls(
        container,
        index=1,
        x=x,
        binned=binned,
        feature_weights=feature_weights,
        feature_predict=feature_predict,
        prediction=prediction,
        y_threshold=threshold,
        instance_bin=0,
    )


@pytest.fixture
def simple_explanation():
    return _make_explanation()


def _build_rules_fixture(explanation):
    return {
        "rule": ["f0 <= 0.2", "f1 > 0.3"],
        "feature": [0, 1],
        "sampled_values": [np.array([0.1]), np.array([0.4])],
        "feature_value": [0.1, 0.4],
        "value": ["0.10", "0.30"],
        "weight": [0.2, -0.1],
        "weight_low": [0.1, -0.2],
        "weight_high": [0.3, 0.0],
        "predict": [0.62, 0.48],
        "predict_low": [0.5, 0.4],
        "predict_high": [0.7, 0.55],
        "base_predict": [explanation.prediction["predict"]],
        "base_predict_low": [explanation.prediction["low"]],
        "base_predict_high": [explanation.prediction["high"]],
        "is_conjunctive": [False, False],
        "classes": explanation.prediction["classes"],
    }


@pytest.fixture
def telemetry_explanation():
    explanation = _make_explanation()
    rules = _build_rules_fixture(explanation)
    explanation._rules = rules
    explanation.rules = rules
    explanation._has_rules = True
    return explanation


class StubAlternative(explanation_module.AlternativeExplanation):
    def _check_preconditions(self):  # pragma: no cover - simple stub
        return None

    def _get_rules(self):
        if not hasattr(self, "_rules"):
            self._rules = {
                "rule": ["f0 <= 0.2"],
                "feature": [0],
                "sampled_values": [np.array([0.1, 0.2])],
                "feature_value": [0.1],
                "value": ["0.10"],
                "weight": [0.2],
                "weight_low": [0.1],
                "weight_high": [0.3],
                "predict": [0.75],
                "predict_low": [0.7],
                "predict_high": [0.8],
                "base_predict": [self.prediction["predict"]],
                "base_predict_low": [self.prediction["low"]],
                "base_predict_high": [self.prediction["high"]],
                "is_conjunctive": [False],
                "classes": self.prediction["classes"],
            }
        return self._rules


@pytest.fixture
def alternative_explanation():
    return _make_explanation(cls=StubAlternative)


def test_prediction_interval(simple_explanation):
    assert simple_explanation.prediction_interval == (0.5, 0.9)
    assert simple_explanation.predict == pytest.approx(0.6)


def test_categorical_target_uses_default_minmax():
    categorical = Categorical(["a", "b"])
    explanation = SimpleExplanation(
        ContainerStub(y_cal=categorical),
        index=1,
        x=np.array([[0.1, 0.2], [0.3, 0.4]]),
        binned={"weights": np.array([[1, 2], [3, 4]])},
        feature_weights={"weights": np.array([[0.1, 0.2], [0.3, 0.4]])},
        feature_predict={"weights": np.array([[0.5, 0.6], [0.7, 0.8]])},
        prediction={
            "predict": np.array([0.2, 0.6]),
            "low": np.array([0.1, 0.5]),
            "high": np.array([0.4, 0.9]),
        },
        y_threshold=None,
        instance_bin=0,
    )
    assert explanation.y_minmax == [0, 0]


def test_length_and_metadata_helpers(simple_explanation):
    explainer = simple_explanation._get_explainer()
    explainer.class_labels = ["negative", "positive"]
    assert len(simple_explanation) == 2
    assert simple_explanation.get_mode() == "regression"
    assert simple_explanation.get_class_labels() == ["negative", "positive"]
    assert simple_explanation.is_thresholded() is False
    assert simple_explanation.is_probabilistic() is False


def test_rank_features_requires_input(simple_explanation):
    from calibrated_explanations.core.exceptions import ValidationError
    with pytest.raises(ValidationError):
        simple_explanation._rank_features()


def test_rank_features_orders_by_width_and_weight(simple_explanation):
    weights = np.array([0.4, -0.1, 0.2])
    widths = np.array([0.3, 0.5, 0.1])
    ranked = simple_explanation._rank_features(feature_weights=weights, width=widths)
    assert set(ranked) == {0, 1, 2}
    assert ranked[-1] == 0  # highest absolute weight retained last
    limited = simple_explanation._rank_features(feature_weights=weights, num_to_show=2)
    assert len(limited) == 2
    width_only = simple_explanation._rank_features(width=widths)
    assert set(width_only) == {0, 1, 2}


def test_percentile_helpers():
    assert explanation_module.CalibratedExplanation._normalize_percentile_value(
        95
    ) == pytest.approx(0.95)
    assert explanation_module.CalibratedExplanation._normalize_percentile_value(-np.inf) == -np.inf
    assert explanation_module.CalibratedExplanation._normalize_percentile_value("invalid") is None
    assert explanation_module.CalibratedExplanation._normalize_percentile_value(None) is None

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


def test_normalize_threshold_value_handles_sequences():
    explanation = _make_explanation()
    explanation.y_threshold = None
    assert explanation._normalize_threshold_value() is None

    explanation.y_threshold = np.array([0.4, 0.6])
    assert explanation._normalize_threshold_value() == [0.4, 0.6]

    explanation.y_threshold = (0.2, 0.8)
    assert explanation._normalize_threshold_value() == [0.2, 0.8]

    explanation.y_threshold = []
    assert explanation._normalize_threshold_value() is None

    explanation.y_threshold = 0.75
    assert explanation._normalize_threshold_value() == pytest.approx(0.75)


def test_build_uncertainty_payload_controls_percentiles():
    explanation = _make_explanation()
    payload = explanation._build_uncertainty_payload(
        value=np.array([0.6]),
        low=0.4,
        high=0.8,
        representation="percentile",
        percentiles=(0.05, 0.95),
    )
    assert payload["legacy_interval"] == [0.4, 0.8]
    assert payload["raw_percentiles"] == [0.05, 0.95]
    assert payload["confidence_level"] == pytest.approx(0.9)

    no_percentiles = explanation._build_uncertainty_payload(
        value=0.6,
        low=0.4,
        high=0.8,
        representation="percentile",
        percentiles=(0.05, 0.95),
        include_percentiles=False,
    )
    assert no_percentiles["raw_percentiles"] is None
    assert no_percentiles["confidence_level"] is None


def test_build_instance_uncertainty_for_modes():
    base = _make_explanation()
    payload = base._build_instance_uncertainty()
    assert payload["representation"] == "percentile"
    assert payload["raw_percentiles"] == [0.05, 0.95]

    thresholded = _make_explanation(threshold=(0.2, 0.8))
    rules = _build_rules_fixture(thresholded)
    thresholded._rules = rules
    thresholded.rules = rules
    thresholded._has_rules = True
    threshold_payload = thresholded._build_instance_uncertainty()
    assert threshold_payload["representation"] == "threshold"
    assert threshold_payload["threshold"] == [0.2, 0.8]
    assert threshold_payload["raw_percentiles"] is None

    probabilistic = _make_explanation(mode="classification")
    probabilistic_payload = probabilistic._build_instance_uncertainty()
    assert probabilistic_payload["representation"] == "venn_abers"
    assert probabilistic_payload["raw_percentiles"] is None


def test_safe_feature_name_conversion():
    explanation = _make_explanation()
    explainer = explanation._get_explainer()
    explainer.feature_names = ["age", "height", "weight"]
    assert explanation._safe_feature_name(1) == "height"
    assert explanation._safe_feature_name(5) == "5"
    assert explanation._safe_feature_name("custom") == "custom"


def test_convert_condition_value_special_tokens():
    assert explanation_module.CalibratedExplanation._convert_condition_value(
        None, 1.2
    ) == pytest.approx(1.2)
    assert explanation_module.CalibratedExplanation._convert_condition_value("-inf", 0.0) == float(
        "-inf"
    )
    assert explanation_module.CalibratedExplanation._convert_condition_value(
        "infinity", 0.0
    ) == float("inf")
    assert explanation_module.CalibratedExplanation._convert_condition_value(
        "7.5", 0.0
    ) == pytest.approx(7.5)
    assert explanation_module.CalibratedExplanation._convert_condition_value("foo", 0.0) == "foo"


def test_build_condition_payload_parses_rules(telemetry_explanation):
    payload = telemetry_explanation._build_condition_payload(
        feature_index=0,
        rule_text="f0 >= 0.4",
        feature_value=np.array([0.1]),
        display_value=0.1,
    )
    assert payload["feature"] == "f0"
    assert payload["operator"] == ">="
    assert payload["value"] == pytest.approx(0.4)

    raw_payload = telemetry_explanation._build_condition_payload(
        feature_index="custom",
        rule_text="",
        feature_value=None,
        display_value=0.7,
    )
    assert raw_payload["operator"] == "raw"
    assert raw_payload["value"] == pytest.approx(0.7)


def test_build_factual_rules_payload_serializes_rules(telemetry_explanation):
    payload = telemetry_explanation.build_rules_payload()
    core = payload["core"]
    metadata = payload["metadata"]
    assert core["kind"] == "factual"
    assert len(core["feature_rules"]) == 2
    first = core["feature_rules"][0]
    assert set(first.keys()) == {"weight", "condition"}
    assert "uncertainty_interval" in first["weight"]
    assert metadata["feature_rules"][0]["weight_uncertainty"]["representation"] == "percentile"
    assert metadata["feature_rules"][0]["prediction_uncertainty"]["raw_percentiles"] == [
        0.05,
        0.95,
    ]


def test_build_factual_rules_payload_threshold_representation():
    explanation = _make_explanation(threshold=(0.2, 0.8))
    rules = _build_rules_fixture(explanation)
    explanation._rules = rules
    explanation.rules = rules
    explanation._has_rules = True
    payload = explanation.build_rules_payload()
    metadata_rule = payload["metadata"]["feature_rules"][0]
    prediction_metadata = metadata_rule["prediction_uncertainty"]
    assert prediction_metadata["representation"] == "threshold"
    assert prediction_metadata["threshold"] == [0.2, 0.8]
    assert metadata_rule["weight_uncertainty"]["representation"] == "venn_abers"


def test_build_rules_payload_for_alternative(alternative_explanation):
    payload = alternative_explanation.build_rules_payload()
    core = payload["core"]
    metadata = payload["metadata"]
    assert core["kind"] == "alternative"
    assert len(core["feature_rules"]) == 1
    feature_rule = metadata["feature_rules"][0]
    assert feature_rule["prediction_uncertainty"]["representation"] == "percentile"
    assert feature_rule["weight_uncertainty"]["representation"] == "percentile"


def test_to_telemetry_includes_serialized_rules(telemetry_explanation):
    telemetry = telemetry_explanation.to_telemetry()
    assert set(telemetry.keys()) == {"uncertainty", "rules", "metadata"}
    assert telemetry["rules"]["core"]
    assert telemetry["uncertainty"]["representation"] == "percentile"


def test_predict_conjunctive_average():
    explanation = _make_explanation()
    perturbed = np.array(explanation.x_test[1], copy=True)
    predict, low, high = explanation._predict_conjunctive(
        [np.array([0.1, 0.2]), np.array([0.3, 0.4])],
        [0, 1],
        perturbed,
        threshold=None,
        predicted_class=explanation.prediction["classes"],
    )
    assert predict == pytest.approx(1.0)
    assert low == pytest.approx(0.5)
    assert high == pytest.approx(1.5)


def test_predict_conjunctive_requires_multiple_features():
    from calibrated_explanations.core.exceptions import ValidationError
    explanation = _make_explanation()
    with pytest.raises(ValidationError):
        explanation._predict_conjunctive(
            [np.array([0.1])],
            [0],
            np.array(explanation.x_test[1], copy=True),
            threshold=None,
            predicted_class=explanation.prediction["classes"],
        )


def test_define_conditions_handles_categorical_labels():
    explanation = _make_explanation()
    explanation.calibrated_explanations.features_to_ignore = [1]
    explainer = explanation._get_explainer()
    explainer.categorical_features = [0]
    explainer.categorical_labels = [[], []]
    conditions = explanation._define_conditions()
    assert conditions[0] == "f0 = 0"
    assert conditions[1] == ""
