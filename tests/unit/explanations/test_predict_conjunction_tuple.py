import numpy as np
from types import SimpleNamespace

from calibrated_explanations.explanations.explanation import FactualExplanation


class DummyPredictionOrchestrator:
    def __init__(self):
        self.last_batch = None
        self.last_bins = None

    def predict_internal(self, batch, threshold, low_high_percentiles, classes, bins):
        self.last_batch = np.asarray(batch)
        self.last_bins = None if bins is None else np.asarray(bins)
        p_value = self.last_batch.sum(axis=1)
        low = p_value - 0.1
        high = p_value + 0.1
        return p_value, low, high, None


def _make_dummy_explanation(orchestrator):
    explainer = SimpleNamespace(prediction_orchestrator=orchestrator)
    container = SimpleNamespace(get_explainer=lambda: explainer, low_high_percentiles=(5, 95))
    explanation = FactualExplanation.__new__(FactualExplanation)
    explanation.calibrated_explanations = container
    return explanation


def test_predict_conjunction_tuple_basic():
    orchestrator = DummyPredictionOrchestrator()
    explanation = _make_dummy_explanation(orchestrator)

    rule_value_set = [np.array([1.0, 2.0]), np.array([10.0, 20.0])]
    original_features = [0, 1]
    perturbed = np.array([0.0, 0.0])

    predict, low, high = explanation.predict_conjunction_tuple(
        rule_value_set,
        original_features,
        perturbed,
        threshold=None,
        predicted_class=1,
        bins=None,
    )

    assert orchestrator.last_batch.shape == (4, 2)
    assert np.allclose(
        orchestrator.last_batch,
        np.array([[1.0, 10.0], [1.0, 20.0], [2.0, 10.0], [2.0, 20.0]]),
    )
    assert predict == np.mean([11.0, 21.0, 12.0, 22.0])
    assert low == np.mean([10.9, 20.9, 11.9, 21.9])
    assert high == np.mean([11.1, 21.1, 12.1, 22.1])


def test_predict_conjunction_tuple_1d_perturbed():
    orchestrator = DummyPredictionOrchestrator()
    explanation = _make_dummy_explanation(orchestrator)

    rule_value_set = [np.array([1.0]), np.array([2.0])]
    original_features = [0, 1]
    perturbed = np.array([0.0, 0.0])

    explanation.predict_conjunction_tuple(
        rule_value_set,
        original_features,
        perturbed,
        threshold=None,
        predicted_class=1,
        bins=None,
    )

    assert orchestrator.last_batch.shape == (1, 2)


def test_predict_conjunction_tuple_empty_values():
    orchestrator = DummyPredictionOrchestrator()
    explanation = _make_dummy_explanation(orchestrator)

    rule_value_set = [np.array([]), np.array([1.0])]
    original_features = [0, 1]
    perturbed = np.array([0.0, 0.0])

    predict, low, high = explanation.predict_conjunction_tuple(
        rule_value_set,
        original_features,
        perturbed,
        threshold=None,
        predicted_class=1,
        bins=None,
    )

    assert predict == 0.0
    assert low == 0.0
    assert high == 0.0


def test_predict_conjunction_tuple_bins_handling():
    orchestrator = DummyPredictionOrchestrator()
    explanation = _make_dummy_explanation(orchestrator)

    rule_value_set = [np.array([1.0, 2.0]), np.array([10.0])]
    original_features = [0, 1]
    perturbed = np.array([0.0, 0.0])

    explanation.predict_conjunction_tuple(
        rule_value_set,
        original_features,
        perturbed,
        threshold=None,
        predicted_class=1,
        bins=3,
    )
    assert orchestrator.last_bins.shape == (2,)
    assert np.all(orchestrator.last_bins == 3)

    explanation.predict_conjunction_tuple(
        rule_value_set,
        original_features,
        perturbed,
        threshold=None,
        predicted_class=1,
        bins=np.array([7]),
    )
    assert orchestrator.last_bins.shape == (2,)
    assert np.all(orchestrator.last_bins == 7)
