import pytest

from calibrated_explanations.explanations.explanation import CalibratedExplanation
from calibrated_explanations.utils.exceptions import ValidationError


def test_safe_feature_name_and_non_numeric():
    class DummyExplanation(CalibratedExplanation):
        def build_rules_payload(self):
            return {"core": {}, "metadata": {}}

        def __repr__(self):
            return "Dummy"

        def plot(self, filter_top=None, **kwargs):
            return None

        def add_conjunctions(self, n_top_features=5, max_rule_size=2):
            return self

        def _check_preconditions(self):
            return True

        def _is_lesser(self):
            return False

    inst = object.__new__(DummyExplanation)

    class DummyExplainer:
        feature_names = ["f0"]

    class DummyContainer:
        def __init__(self):
            self.expl = DummyExplainer()

        def get_explainer(self):
            return self.expl

    inst.calibrated_explanations = DummyContainer()
    # non-numeric feature index -> string
    assert inst.safe_feature_name("x") == "x"
    # numeric but out-of-range -> index string
    assert inst.safe_feature_name(99) == "99"


def test_ignored_features_for_instance_combination():
    class DummyExplanation(CalibratedExplanation):
        def build_rules_payload(self):
            return {"core": {}, "metadata": {}}

        def __repr__(self):
            return "Dummy"

        def plot(self, filter_top=None, **kwargs):
            return None

        def add_conjunctions(self, n_top_features=5, max_rule_size=2):
            return self

        def _check_preconditions(self):
            return True

        def _is_lesser(self):
            return False

    inst = object.__new__(DummyExplanation)

    class DummyExplainer:
        feature_names = ["a", "b"]
        num_features = 2

    class DummyContainer:
        features_to_ignore = (0,)

        def __init__(self):
            self.expl = DummyExplainer()
            self.feature_filter_per_instance_ignore = [[1], []]

        def get_explainer(self):
            return self.expl

    inst.calibrated_explanations = DummyContainer()
    inst.index = 0
    ignored = inst.ignored_features_for_instance()
    assert 0 in ignored and 1 in ignored


def test_rank_features_requires_input():
    class DummyExplanation(CalibratedExplanation):
        def build_rules_payload(self):
            return {"core": {}, "metadata": {}}

        def __repr__(self):
            return "Dummy"

        def plot(self, filter_top=None, **kwargs):
            return None

        def add_conjunctions(self, n_top_features=5, max_rule_size=2):
            return self

        def _check_preconditions(self):
            return True

        def _is_lesser(self):
            return False

    inst = object.__new__(DummyExplanation)
    with pytest.raises(ValidationError):
        inst.rank_features()


def test_is_one_sided_detection():
    class DummyExplanation(CalibratedExplanation):
        def build_rules_payload(self):
            return {"core": {}, "metadata": {}}

        def __repr__(self):
            return "Dummy"

        def plot(self, filter_top=None, **kwargs):
            return None

        def add_conjunctions(self, n_top_features=5, max_rule_size=2):
            return self

        def _check_preconditions(self):
            return True

        def _is_lesser(self):
            return False

    inst = object.__new__(DummyExplanation)

    class DummyContainer:
        def __init__(self):
            self.low_high_percentiles = (None, None)

        def get_low_percentile(self):
            return float("-inf")

        def get_high_percentile(self):
            return 95

    inst.calibrated_explanations = DummyContainer()
    assert inst.is_one_sided()
