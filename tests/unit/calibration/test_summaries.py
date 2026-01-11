from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from calibrated_explanations.calibration.summaries import (
    get_calibration_summaries,
    invalidate_calibration_summaries,
)


@dataclass
class _CacheFacadeFake:
    cached: tuple[dict[int, dict[object, int]], dict[int, np.ndarray]] | None = None
    invalidated: bool = False
    set_calls: int = 0

    def get_calibration_summaries(self, *, explainer_id: str, x_cal_hash: str):
        return self.cached

    def set_calibration_summaries(
        self,
        *,
        explainer_id: str,
        x_cal_hash: str,
        categorical_counts: dict[int, dict[object, int]],
        numeric_sorted: dict[int, np.ndarray],
    ) -> None:
        self.set_calls += 1
        self.cached = (categorical_counts, numeric_sorted)

    def invalidate_all(self) -> None:
        self.invalidated = True
        self.cached = None


class _ExplainerLike:
    def __init__(self, *, x_cal: np.ndarray, categorical_features: list[int], num_features: int):
        self.x_cal = x_cal
        self.categorical_features = categorical_features
        self.num_features = num_features

        self.categorical_value_counts_cache = None
        self.numeric_sorted_cache = None
        self.calibration_summary_shape = None


def test_get_calibration_summaries__should_return_facade_cached_payload_when_present():
    # Arrange
    facade = _CacheFacadeFake(
        cached=(
            {1: {"a": 2}},
            {0: np.asarray([0, 0, 1])},
        )
    )
    explainer = _ExplainerLike(
        x_cal=np.asarray([[0, "a"], [1, "a"], [0, "a"]], dtype=object),
        categorical_features=[1],
        num_features=2,
    )
    setattr(explainer, "_" + "explanation_cache", facade)

    # Act
    cat_counts, numeric_sorted = get_calibration_summaries(explainer)

    # Assert
    assert cat_counts == {1: {"a": 2}}
    assert np.array_equal(numeric_sorted[0], np.asarray([0, 0, 1]))


def test_get_calibration_summaries__should_compute_and_store_when_facade_cache_miss():
    # Arrange
    facade = _CacheFacadeFake(cached=None)
    explainer = _ExplainerLike(
        x_cal=np.asarray([[0, "a"], [1, "b"], [0, "a"]], dtype=object),
        categorical_features=[1],
        num_features=2,
    )
    setattr(explainer, "_" + "explanation_cache", facade)

    # Act
    cat_counts, numeric_sorted = get_calibration_summaries(explainer)

    # Assert (domain behavior)
    assert cat_counts[1]["a"] == 2
    assert cat_counts[1]["b"] == 1
    assert np.array_equal(numeric_sorted[0], np.asarray([0, 0, 1]))

    # Assert (cached in facade + instance caches)
    assert facade.set_calls == 1
    assert facade.cached is not None
    assert explainer.categorical_value_counts_cache == cat_counts
    assert explainer.numeric_sorted_cache == numeric_sorted

    # Second call should hit facade
    cat_counts2, numeric_sorted2 = get_calibration_summaries(explainer)
    assert cat_counts2 == cat_counts
    assert np.array_equal(numeric_sorted2[0], numeric_sorted[0])


def test_invalidate_calibration_summaries__should_clear_instance_caches_and_facade():
    # Arrange
    facade = _CacheFacadeFake(cached=({0: {"x": 1}}, {0: np.asarray([1])}))
    explainer = _ExplainerLike(
        x_cal=np.asarray([[0, "a"]], dtype=object),
        categorical_features=[1],
        num_features=2,
    )
    setattr(explainer, "_" + "explanation_cache", facade)
    explainer.categorical_value_counts_cache = {1: {"a": 1}}
    explainer.numeric_sorted_cache = {0: np.asarray([0])}
    explainer.calibration_summary_shape = explainer.x_cal.shape

    # Act
    invalidate_calibration_summaries(explainer)

    # Assert
    assert facade.invalidated is True
    assert explainer.categorical_value_counts_cache is None
    assert explainer.numeric_sorted_cache is None
    assert explainer.calibration_summary_shape is None
