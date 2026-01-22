import math

import numpy as np


from calibrated_explanations.explanations.explanation import (
    CalibratedExplanation,
)


def test_to_python_number_and_normalize():
    assert CalibratedExplanation.to_python_number(None) is None
    assert CalibratedExplanation.to_python_number(np.int32(3)) == 3
    assert CalibratedExplanation.to_python_number(np.float64(2.5)) == 2.5
    assert CalibratedExplanation.to_python_number(np.array([1, 2])) == [1, 2]


def test_normalize_percentile_value_and_confidence():
    assert CalibratedExplanation.normalize_percentile_value(None) is None
    assert CalibratedExplanation.normalize_percentile_value(50) == 0.5
    assert CalibratedExplanation.normalize_percentile_value(0.2) == 0.2
    assert math.isinf(CalibratedExplanation.normalize_percentile_value(float("inf")))

    # compute_confidence_level
    assert CalibratedExplanation.compute_confidence_level(None) is None
    assert CalibratedExplanation.compute_confidence_level((0.1, 0.9)) == 0.8
    assert CalibratedExplanation.compute_confidence_level((float("-inf"), 0.9)) == 0.9
    assert CalibratedExplanation.compute_confidence_level((0.1, float("inf"))) == 0.9


def test_build_interval_and_uncertainty_payload():
    interval = CalibratedExplanation.build_interval(1.0, 2.0)
    assert interval["lower"] == 1.0
    assert interval["upper"] == 2.0


def test_convert_and_parse_condition():
    assert CalibratedExplanation.convert_condition_value(None, 5) == 5
    assert CalibratedExplanation.convert_condition_value("-inf", 0) == float("-inf")
    assert CalibratedExplanation.convert_condition_value("inf", 0) == float("inf")
    assert CalibratedExplanation.convert_condition_value("3.5", 0) == 3.5

    # parse simple conditions using public API via a minimal concrete subclass
    class _Dummy(CalibratedExplanation):
        def __init__(self):
            # bypass base initializer
            pass

        def build_rules_payload(self):
            return {}

        def __repr__(self):
            return "_Dummy"

        def plot(self, filter_top=None, **kwargs):
            return None

        def add_conjunctions(self, n_top_features=5, max_rule_size=2):
            return self

        def _check_preconditions(self):
            return None

        def _is_lesser(self, rule_boundary, instance_value):
            return instance_value < float(rule_boundary)

    inst = object.__new__(_Dummy)
    op, val = _Dummy.parse_condition(inst, "feat", "feat <= 3.5")
    assert op == "<="
    assert val == "3.5"
