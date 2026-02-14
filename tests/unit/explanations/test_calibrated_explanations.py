"""Tests for :mod:`calibrated_explanations.explanations.explanations`."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Sequence

import numpy as np
import pytest

from calibrated_explanations.explanations import (
    AlternativeExplanations,
    CalibratedExplanations,
)
from tests.helpers.deprecation import warns_or_raises, deprecations_error_enabled


@dataclass
class DummyOriginalExplainer:
    """Lightweight stand-in for :class:`CalibratedExplainer`."""

    feature_names: Sequence[str]
    class_labels: dict[int, str]
    mode: str = "classification"

    def __post_init__(self) -> None:
        self.x_cal = np.array([[0.1, 0.2]])
        self.y_cal = np.array([0.25, 0.75])
        self.num_features = len(self.feature_names)
        self.categorical_features: List[int] = []
        self.categorical_labels: dict[int, Sequence[str]] = {}
        self.feature_values: List[Sequence[str]] = [() for _ in self.feature_names]
        self.sample_percentiles = (5.0, 95.0)
        self.is_multiclass = True
        self.discretizer = object()
        self.rule_boundaries: List[float] = []
        self.learner = "dummy-learner"
        self.difficulty_estimator = "dummy-difficulty"
        from calibrated_explanations.plugins.manager import PluginManager

        self.plugin_manager = PluginManager(self)

        class PredOrch:
            def predict_internal(self, _x, **_kw):
                return (np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1))

        self.prediction_orchestrator = PredOrch()

    def predict(self, data):  # pragma: no cover - not used directly
        return np.asarray(data)

    def discretize(self, x):
        return x

    def preload_lime(self):
        """Return a minimal structure compatible with :meth:`CalibratedExplanations.as_lime`."""

        lime_exp = SimpleNamespace(
            intercept={1: 0.0},
            local_pred=None,
            predict_proba=[0.0, 1.0],
            local_exp={1: [None] * self.num_features},
            domain_mapper=SimpleNamespace(
                discretized_feature_names=None,
                feature_values=None,
            ),
        )
        return None, lime_exp

    def preload_shap(self):
        """Return a minimal structure compatible with :meth:`CalibratedExplanations.as_shap`."""

        shap_exp = SimpleNamespace(
            base_values=np.array([0.0]),
            values=np.zeros((1, self.num_features)),
            data=None,
        )
        return None, shap_exp


class DummyExplanation:
    """Minimal explanation object exercising list-handling logic."""

    def __init__(
        self,
        index: int,
        predict: float,
        *,
        probabilities=None,
        interval=(None, None),
    ) -> None:
        self.index = index
        self.predict = predict
        self.prediction_interval = interval
        self.prediction = {
            "predict": float(predict),
            "low": float(predict) - 0.1,
            "high": float(predict) + 0.1,
        }
        weight = np.array([predict, predict / 2])
        self.feature_weights = {"predict": weight}
        self.feature_predict = {"predict": weight}
        self.prediction_probabilities = probabilities
        self.x_test = np.array([predict, predict + 1])
        self.plot_calls: list[tuple] = []
        self.conjunction_calls: list[str] = []

    # -- Helpers used by :mod:`CalibratedExplanations` --
    def remove_conjunctions(self) -> None:
        self.conjunction_calls.append("remove")

    def add_conjunctions(self, *_args, **_kwargs) -> None:
        self.conjunction_calls.append("add")

    def reset(self) -> None:
        self.conjunction_calls.append("reset")

    def plot(self, **kwargs) -> None:
        self.plot_calls.append(tuple(sorted(kwargs.items())))

    def rank_features(self, _weights, num_to_show=None):  # pylint: disable=unused-argument
        size = len(self.feature_weights["predict"])
        return list(range(size if num_to_show is None else num_to_show))

    def define_conditions(self):
        return [f"feature_{i}" for i in range(len(self.feature_weights["predict"]))]

    def get_rules(self):
        return {"rule": [0, 1]}

    def super_explanations(
        self, only_ensured=False, include_potential=True, copy=True
    ) -> DummyExplanation:
        self.conjunction_calls.append("super")
        return self

    def semi_explanations(
        self, only_ensured=False, include_potential=True, copy=True
    ) -> DummyExplanation:
        self.conjunction_calls.append("semi")
        return self

    def counter_explanations(
        self, only_ensured=False, include_potential=True, copy=True
    ) -> DummyExplanation:
        self.conjunction_calls.append("counter")
        return self

    def ensured_explanations(self, include_potential=True, copy=True) -> DummyExplanation:
        self.conjunction_calls.append("ensured")
        return self

    def filter_rule_sizes(self, *, rule_sizes=None, size_range=None, copy=True) -> DummyExplanation:
        self.conjunction_calls.append(("filter_rule_sizes", rule_sizes, size_range, copy))
        return self


@pytest.fixture()
def collection() -> CalibratedExplanations:
    explainer = DummyOriginalExplainer(feature_names=("f0", "f1"), class_labels={0: "no", 1: "yes"})
    x = np.arange(6, dtype=float).reshape(3, 2)
    thresholds = [(0.1, 0.9), (0.2, 0.8), (0.3, 0.7)]
    bins = ["bin0", "bin1", "bin2"]
    coll = CalibratedExplanations(explainer, x, thresholds, bins)
    coll.explanations = [
        DummyExplanation(0, 0.1, probabilities=np.array([0.9, 0.1]), interval=(0.0, 0.2)),
        DummyExplanation(1, 0.2, probabilities=np.array([0.8, 0.2]), interval=(0.1, 0.3)),
        DummyExplanation(2, 0.3, probabilities=np.array([0.7, 0.3]), interval=(0.2, 0.4)),
    ]
    return coll


def test_collection_filter_rule_sizes_delegates(collection: CalibratedExplanations) -> None:
    filtered = collection.filter_rule_sizes(rule_sizes=1, copy=True)
    for exp in filtered.explanations:
        assert ("filter_rule_sizes", 1, None, True) in exp.conjunction_calls


def test_one_sided_confidence_logic(collection: CalibratedExplanations) -> None:
    collection.low_high_percentiles = (5.0, np.inf)
    assert collection.is_one_sided
    assert collection.get_confidence() == 95.0
    collection.low_high_percentiles = (np.inf, 90.0)
    assert collection.get_confidence() == 90.0
    collection.low_high_percentiles = (10.0, 90.0)
    assert collection.get_confidence() == 80.0


def test_deprecated_get_explanation_checks(collection: CalibratedExplanations) -> None:
    from calibrated_explanations.utils.exceptions import ValidationError

    if deprecations_error_enabled():
        # In raise-mode deprecations trigger before the validation checks;
        # assert that the deprecation is raised instead of the ValidationError.
        with pytest.raises(DeprecationWarning):
            collection.get_explanation(1)
        with pytest.raises(DeprecationWarning):
            collection.get_explanation("1")  # type: ignore[arg-type]
        with pytest.raises(DeprecationWarning):
            collection.get_explanation(-1)
        with pytest.raises(DeprecationWarning):
            collection.get_explanation(len(collection.x_test))
    else:
        with warns_or_raises():
            assert collection.get_explanation(1) is collection.explanations[1]
        with pytest.raises(ValidationError), warns_or_raises():
            collection.get_explanation("1")  # type: ignore[arg-type]
        with pytest.raises(ValidationError), warns_or_raises():
            collection.get_explanation(-1)
        with pytest.raises(ValidationError), warns_or_raises():
            collection.get_explanation(len(collection.x_test))


def test_alternative_explanation_proxies(collection: CalibratedExplanations) -> None:
    alt = AlternativeExplanations(
        collection.calibrated_explainer.explainer,
        collection.x_test,
        collection.y_threshold,
        collection.bins,
    )
    alt.explanations = collection.explanations
    alt.super_explanations()
    alt.semi_explanations()
    alt.counter_explanations()
    alt.ensured_explanations()
    calls = [exp.conjunction_calls for exp in collection.explanations]
    assert all({"super", "semi", "counter", "ensured"}.issubset(set(call)) for call in calls)
