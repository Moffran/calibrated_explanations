import math

from calibrated_explanations.explanations.explanation import CalibratedExplanation


def test_percentile_normalization_and_confidence():
    # normalize_percentile_value: percent values >1 are treated as percent
    assert CalibratedExplanation.normalize_percentile_value(5) == 0.05
    assert CalibratedExplanation.normalize_percentile_value(0.5) == 0.5
    assert math.isinf(CalibratedExplanation.normalize_percentile_value(float("inf")))

    # compute_confidence_level: normal case
    conf = CalibratedExplanation.compute_confidence_level((0.05, 0.95))
    assert math.isclose(conf, 0.9, rel_tol=1e-9)

    # -inf / finite -> returns high
    assert math.isclose(
        CalibratedExplanation.compute_confidence_level((float("-inf"), 0.2)), 0.2, rel_tol=1e-9
    )

    # finite / inf -> returns 1 - low
    assert math.isclose(
        CalibratedExplanation.compute_confidence_level((0.8, float("inf"))), 0.2, rel_tol=1e-9
    )


def test_build_uncertainty_payload_and_instance_branches():
    # Create a minimal concrete subclass so we can instantiate without calling __init__
    class DummyExplanation(CalibratedExplanation):
        def build_rules_payload(self):
            return {"core": {}, "metadata": {}}

        def __repr__(self):
            return "DummyExplanation"

        def plot(self, filter_top=None, **kwargs):
            return None

        def add_conjunctions(self, n_top_features=5, max_rule_size=2):
            return self

        def _check_preconditions(self):
            return True

        def _is_lesser(self):
            return False

    inst = object.__new__(DummyExplanation)

    # build_uncertainty_payload (alias) does not depend on instance state
    payload = inst.build_uncertainty_payload(
        value=0.6,
        low=0.55,
        high=0.65,
        representation="percentile",
        percentiles=(0.05, 0.95),
        threshold=None,
        include_percentiles=True,
    )
    assert payload["representation"] == "percentile"
    assert payload["raw_percentiles"] == [0.05, 0.95]
    assert math.isclose(payload["confidence_level"], 0.9, rel_tol=1e-9)

    # Prepare a minimal explainer/container for branch testing
    class DummyExplainer:
        def __init__(self, mode="regression"):
            self.mode = mode
            self.class_labels = None
            self.feature_names = ["f0"]
            self.num_features = 1
            self.categorical_features = []
            self.y_cal = [0, 1]

    class DummyContainer:
        def __init__(self, explainer, low_high_percentiles=None):
            self.expl = explainer
            self.low_high_percentiles = low_high_percentiles

        def get_explainer(self):
            return self.expl

    # Thresholded regression -> representation 'threshold' and threshold present
    inst.calibrated_explanations = DummyContainer(DummyExplainer(mode="regression"))
    inst.get_explainer = lambda: inst.calibrated_explanations.get_explainer()
    inst.prediction = {"predict": 0.6, "low": 0.55, "high": 0.65}
    inst.y_threshold = 0.5
    u = inst.build_instance_uncertainty()
    assert u["representation"] == "threshold"
    assert u["threshold"] == 0.5

    # Probabilistic (classification) -> representation 'venn_abers'
    inst.calibrated_explanations = DummyContainer(DummyExplainer(mode="classification"))
    inst.y_threshold = None
    u = inst.build_instance_uncertainty()
    assert u["representation"] == "venn_abers"

    # Percentile path (regression, no threshold) -> includes percentiles
    inst.calibrated_explanations = DummyContainer(
        DummyExplainer(mode="regression"), low_high_percentiles=(5, 95)
    )
    inst.y_threshold = None
    u = inst.build_instance_uncertainty()
    assert u["representation"] == "percentile"
    assert u["raw_percentiles"] == [0.05, 0.95]
