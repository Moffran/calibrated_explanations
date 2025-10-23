"""Tests for :mod:`calibrated_explanations.explanations.explanations`."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Sequence

import numpy as np
import pytest

from calibrated_explanations.explanations.explanations import (
    AlternativeExplanations,
    CalibratedExplanations,
)


@dataclass
class _DummyOriginalExplainer:
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
        self.assign_threshold = 0.5
        self.sample_percentiles = (5.0, 95.0)
        self.is_multiclass = True
        self.discretizer = object()
        self.rule_boundaries: List[float] = []
        self.learner = "dummy-learner"
        self.difficulty_estimator = "dummy-difficulty"

    def _predict(self, data):  # pragma: no cover - not used directly
        return np.asarray(data)

    def _preload_lime(self):
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

    def _preload_shap(self):
        """Return a minimal structure compatible with :meth:`CalibratedExplanations.as_shap`."""

        shap_exp = SimpleNamespace(
            base_values=np.array([0.0]),
            values=np.zeros((1, self.num_features)),
            data=None,
        )
        return None, shap_exp


class _DummyExplanation:
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
        self._plot_calls: list[tuple] = []
        self._conjunction_calls: list[str] = []

    # -- Helpers used by :mod:`CalibratedExplanations` --
    def remove_conjunctions(self) -> None:
        self._conjunction_calls.append("remove")

    def add_conjunctions(self, *_args, **_kwargs) -> None:
        self._conjunction_calls.append("add")

    def reset(self) -> None:
        self._conjunction_calls.append("reset")

    def plot(self, **kwargs) -> None:
        self._plot_calls.append(tuple(sorted(kwargs.items())))

    def _rank_features(self, _weights, num_to_show=None):  # pylint: disable=unused-argument
        size = len(self.feature_weights["predict"])
        return list(range(size if num_to_show is None else num_to_show))

    def _define_conditions(self):
        return [f"feature_{i}" for i in range(len(self.feature_weights["predict"]))]

    def _get_rules(self):
        return {"rule": [0, 1]}

    def super_explanations(self, **_kwargs) -> None:
        self._conjunction_calls.append("super")

    def semi_explanations(self, **_kwargs) -> None:
        self._conjunction_calls.append("semi")

    def counter_explanations(self, **_kwargs) -> None:
        self._conjunction_calls.append("counter")

    def ensured_explanations(self) -> None:
        self._conjunction_calls.append("ensured")


@pytest.fixture()
def collection() -> CalibratedExplanations:
    explainer = _DummyOriginalExplainer(
        feature_names=("f0", "f1"), class_labels={0: "no", 1: "yes"}
    )
    x = np.arange(6, dtype=float).reshape(3, 2)
    thresholds = [(0.1, 0.9), (0.2, 0.8), (0.3, 0.7)]
    bins = ["bin0", "bin1", "bin2"]
    coll = CalibratedExplanations(explainer, x, thresholds, bins)
    coll.explanations = [
        _DummyExplanation(0, 0.1, probabilities=np.array([0.9, 0.1]), interval=(0.0, 0.2)),
        _DummyExplanation(1, 0.2, probabilities=np.array([0.8, 0.2]), interval=(0.1, 0.3)),
        _DummyExplanation(2, 0.3, probabilities=np.array([0.7, 0.3]), interval=(0.2, 0.4)),
    ]
    return coll


def test_getitem_boolean_selection_reindexes(collection: CalibratedExplanations) -> None:
    subset = collection[[True, False, True]]
    assert isinstance(subset, CalibratedExplanations)
    assert len(subset) == 2
    assert subset.bins == ["bin0", "bin2"]
    assert subset.y_threshold == [(0.1, 0.9), (0.3, 0.7)]
    np.testing.assert_array_equal(subset.x_test, np.array([[0.0, 1.0], [4.0, 5.0]]))
    assert [exp.index for exp in subset.explanations] == [0, 1]


def test_getitem_int_list_returns_collection(collection: CalibratedExplanations) -> None:
    subset = collection[[2, 0]]
    assert isinstance(subset, CalibratedExplanations)
    assert [exp.index for exp in subset.explanations] == [0, 1]
    assert subset.y_threshold == [(0.3, 0.7), (0.1, 0.9)]


def test_getitem_slice_singleton_returns_explanation(collection: CalibratedExplanations) -> None:
    single = collection[1:2]
    assert single is collection.explanations[1]


def test_getitem_invalid_type_raises(collection: CalibratedExplanations) -> None:
    with pytest.raises(TypeError):
        _ = collection[1.5]  # type: ignore[index]


def test_prediction_helpers_cache_results(collection: CalibratedExplanations) -> None:
    preds_first = collection.predictions
    preds_second = collection.predictions
    np.testing.assert_allclose(preds_first, [0.1, 0.2, 0.3])
    assert preds_first is preds_second


def test_probability_vectors_are_stacked(collection: CalibratedExplanations) -> None:
    stacked = collection.probabilities
    expected = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]])
    np.testing.assert_allclose(stacked, expected)


def test_probability_matrix_passthrough(collection: CalibratedExplanations) -> None:
    full = np.array([[0.6, 0.4], [0.5, 0.5], [0.4, 0.6]])
    for exp in collection.explanations:
        exp.prediction_probabilities = full
    assert collection.probabilities is full


def test_lower_upper_cache_arrays(collection: CalibratedExplanations) -> None:
    np.testing.assert_allclose(collection.lower, np.array([0.0, 0.1, 0.2]))
    np.testing.assert_allclose(collection.upper, np.array([0.2, 0.3, 0.4]))


def test_one_sided_confidence_logic(collection: CalibratedExplanations) -> None:
    collection.low_high_percentiles = (5.0, np.inf)
    assert collection._is_one_sided()
    assert collection.get_confidence() == 95.0
    collection.low_high_percentiles = (np.inf, 90.0)
    assert collection.get_confidence() == 90.0
    collection.low_high_percentiles = (10.0, 90.0)
    assert collection.get_confidence() == 80.0


def test_get_low_high_percentile_validation(collection: CalibratedExplanations) -> None:
    collection.low_high_percentiles = (5.0, 95.0)
    assert collection.get_low_percentile() == 5.0
    assert collection.get_high_percentile() == 95.0


def test_deprecated_get_explanation_checks(collection: CalibratedExplanations) -> None:
    with pytest.warns(DeprecationWarning):
        assert collection.get_explanation(1) is collection.explanations[1]
    with pytest.raises(TypeError), pytest.warns(DeprecationWarning):
        collection.get_explanation("1")  # type: ignore[arg-type]
    with pytest.raises(ValueError), pytest.warns(DeprecationWarning):
        collection.get_explanation(-1)
    with pytest.raises(ValueError), pytest.warns(DeprecationWarning):
        collection.get_explanation(len(collection.x_test))


def test_plot_routes_calls(monkeypatch, tmp_path, collection: CalibratedExplanations) -> None:
    from calibrated_explanations.utils import helper as helper_utils

    monkeypatch.setattr(helper_utils, "make_directory", lambda *_, **__: None)
    filename = tmp_path / "plot.png"
    collection.plot(index=0, filename=str(filename), show=False)
    assert collection.explanations[0]._plot_calls  # plot called for index
    for exp in collection.explanations:
        exp._plot_calls.clear()
    collection.plot(show=False, filename=str(filename))
    assert all(exp._plot_calls for exp in collection.explanations)


def test_as_lime_and_as_shap_shapes(collection: CalibratedExplanations) -> None:
    lime_objects = collection.as_lime()
    assert len(lime_objects) == len(collection)
    assert all(obj.domain_mapper.feature_values is not None for obj in lime_objects)

    shap_exp = collection.as_shap()
    assert shap_exp.base_values.shape == (len(collection),)
    assert shap_exp.values.shape == (len(collection), collection.x_test.shape[1])
    np.testing.assert_allclose(
        shap_exp.values[0], -collection.explanations[0].feature_weights["predict"]
    )


def test_alternative_explanation_proxies(collection: CalibratedExplanations) -> None:
    alt = AlternativeExplanations(
        collection.calibrated_explainer._explainer,
        collection.x_test,
        collection.y_threshold,
        collection.bins,
    )
    alt.explanations = collection.explanations
    alt.super_explanations()
    alt.semi_explanations()
    alt.counter_explanations()
    alt.ensured_explanations()
    calls = [exp._conjunction_calls for exp in collection.explanations]
    assert all({"super", "semi", "counter", "ensured"}.issubset(set(call)) for call in calls)


def test_from_batch_validation_errors(collection: CalibratedExplanations) -> None:
    batch_missing = SimpleNamespace(collection_metadata={})
    with pytest.raises(ValueError):
        CalibratedExplanations.from_batch(batch_missing)
    batch_wrong = SimpleNamespace(collection_metadata={"container": object()})
    with pytest.raises(TypeError):
        CalibratedExplanations.from_batch(batch_wrong)


def test_frozen_explainer_attribute_proxy(collection: CalibratedExplanations) -> None:
    frozen = collection.calibrated_explainer
    assert frozen.feature_names == ("f0", "f1")
    assert frozen.class_labels == {0: "no", 1: "yes"}
    assert frozen.learner == "dummy-learner"
    assert frozen.difficulty_estimator == "dummy-difficulty"
    with pytest.raises(AttributeError):
        frozen.some_new_attribute = 123  # type: ignore[attr-defined]
