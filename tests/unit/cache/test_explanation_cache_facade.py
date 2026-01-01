from __future__ import annotations

import numpy as np

from calibrated_explanations.cache.explanation_cache import ExplanationCacheFacade


class DummyCache:
    """Minimal fake cache that mirrors the CalibratorCache interface used by the facade."""

    def __init__(self) -> None:
        self.enabled = True
        self.storage: dict[tuple[str, tuple], object] = {}
        self.flush_count = 0
        self.reset_version_calls: list[str] = []

    def key(self, stage: str, parts: tuple) -> tuple[str, tuple]:
        return stage, parts

    def get(self, *, stage: str, parts: tuple) -> object | None:
        return self.storage.get(self.key(stage, parts))

    def set(self, *, stage: str, parts: tuple, value: object) -> None:
        self.storage[self.key(stage, parts)] = value

    def compute(self, *, stage: str, parts: tuple, fn) -> object:
        key = self.key(stage, parts)
        if key in self.storage:
            return self.storage[key]
        value = fn()
        self.storage[key] = value
        return value

    def flush(self) -> None:
        self.flush_count += 1
        self.storage.clear()

    def reset_version(self, new_version: str) -> None:
        self.reset_version_calls.append(new_version)


def test_explanation_cache_facade_routes_calibration_summary_calls() -> None:
    cache = DummyCache()
    facade = ExplanationCacheFacade(cache)
    explainer_id = "test"
    xcal_hash = "hash"
    payload = ({0: {"a": 1}}, {0: np.array([1], dtype=int)})

    facade.set_calibration_summaries(
        explainer_id=explainer_id,
        x_cal_hash=xcal_hash,
        categorical_counts=payload[0],
        numeric_sorted=payload[1],
    )
    assert (
        facade.get_calibration_summaries(explainer_id=explainer_id, x_cal_hash=xcal_hash) == payload
    )

    calls = {"count": 0}
    compute_hash = "compute-hash"

    def compute_fn():
        calls["count"] += 1
        return payload

    first = facade.compute_calibration_summaries(
        explainer_id=explainer_id,
        x_cal_hash=compute_hash,
        compute_fn=compute_fn,
    )
    second = facade.compute_calibration_summaries(
        explainer_id=explainer_id,
        x_cal_hash=compute_hash,
        compute_fn=lambda: None,
    )

    assert first == payload
    assert calls["count"] == 1
    assert second == payload


def test_feature_names_and_invalidation_use_cache() -> None:
    cache = DummyCache()
    facade = ExplanationCacheFacade(cache)
    explainer_id = "names"
    facade.set_feature_names_cache(explainer_id=explainer_id, feature_names=("a", "b"))
    assert facade.get_feature_names_cache(explainer_id=explainer_id) == ("a", "b")

    assert cache.flush_count == 0
    facade.invalidate_all()
    assert cache.flush_count == 1
    assert facade.get_feature_names_cache(explainer_id=explainer_id) is None

    facade.reset_version("new-tag")
    assert cache.reset_version_calls == ["new-tag"]


def test_facade_works_when_cache_disabled() -> None:
    facade = ExplanationCacheFacade(cache=None)
    assert not facade.enabled

    seen: list[int] = []

    def build() -> tuple[list[int], list[int]]:
        seen.append(1)
        return ({0: {"len": 1}}, {0: np.array([1], dtype=int)})

    result_one = facade.compute_calibration_summaries(
        explainer_id="none",
        x_cal_hash="0",
        compute_fn=build,
    )
    result_two = facade.compute_calibration_summaries(
        explainer_id="none",
        x_cal_hash="0",
        compute_fn=build,
    )

    assert result_one == result_two
    assert len(seen) == 2
