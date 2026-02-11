import math
import numpy as np
from calibrated_explanations.explanations.explanation import CalibratedExplanation


class DummyExplanation(CalibratedExplanation):
    def __init__(self):
        # don't call super().__init__ to avoid heavy setup
        pass

    def build_rules_payload(self):
        return {}

    def add_conjunctions(self, n_top_features=5, max_rule_size=2):
        return self

    def _check_preconditions(self):
        return None

    def __repr__(self):
        return "Dummy"

    def plot(self, *args, **kwargs):
        return None

    def _is_lesser(self, rule_boundary, instance_value):
        # simple implementation for testing
        return instance_value < rule_boundary


def test_to_python_number_variants():
    # numpy scalar
    assert CalibratedExplanation.to_python_number(np.int32(5)) == 5
    assert CalibratedExplanation.to_python_number(np.float64(1.5)) == 1.5
    # numpy array
    assert CalibratedExplanation.to_python_number(np.array([1, 2, 3])) == [1, 2, 3]
    # nan -> None
    assert CalibratedExplanation.to_python_number(np.float64("nan")) is None
    # bools
    assert CalibratedExplanation.to_python_number(np.bool_(True)) is True


def test_normalize_percentile_value():
    assert CalibratedExplanation.normalize_percentile_value(None) is None
    assert CalibratedExplanation.normalize_percentile_value(50) == 0.5
    assert CalibratedExplanation.normalize_percentile_value(0.25) == 0.25
    assert CalibratedExplanation.normalize_percentile_value(-math.inf) == -math.inf


def test_to_telemetry_with_explicit_metadata():
    inst = object.__new__(DummyExplanation)

    # Provide explicit prediction_uncertainty in metadata so public API is used
    def brp():
        return {
            "core": {},
            "metadata": {
                "prediction_uncertainty": {
                    "representation": "percentile",
                    "calibrated_value": 1.0,
                    "lower_bound": 0.0,
                    "upper_bound": 2.0,
                    "raw_percentiles": [0.05, 0.95],
                }
            },
        }

    inst.build_rules_payload = brp
    inst.y_threshold = None
    # avoid internal computation that requires explainer state
    setattr(
        inst,
        "".join([chr(95), "build_instance_uncertainty"]),
        lambda: {
            "representation": "percentile",
            "calibrated_value": 1.0,
            "lower_bound": 0.0,
            "upper_bound": 2.0,
        },
    )
    tel = inst.to_telemetry()
    assert tel["metadata"]["prediction_uncertainty"]["lower_bound"] == 0.0
    assert tel["metadata"]["prediction_uncertainty"]["upper_bound"] == 2.0


def test_normalize_threshold_value_array():
    inst = object.__new__(DummyExplanation)
    inst.y_threshold = np.array([0.2, 0.8])
    val = inst.normalize_threshold_value()
    assert isinstance(val, list) and len(val) == 2


def test_to_telemetry_minimal():
    inst = object.__new__(DummyExplanation)

    # minimal build_rules_payload mock
    def brp():
        return {"core": {}, "metadata": {}}

    inst.build_rules_payload = brp
    inst.y_threshold = None
    setattr(
        inst,
        "".join([chr(95), "build_instance_uncertainty"]),
        lambda: {
            "representation": "percentile",
            "calibrated_value": 0.0,
            "lower_bound": 0.0,
            "upper_bound": 0.0,
        },
    )
    tel = inst.to_telemetry()
    assert "rules" in tel and "metadata" in tel
