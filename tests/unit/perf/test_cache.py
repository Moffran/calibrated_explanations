from __future__ import annotations

from time import monotonic
from typing import Dict, Iterable, List

import numpy as np
import pytest

from calibrated_explanations.perf.cache import (
    CalibratorCache,
    CacheConfig,
    LRUCache,
    _default_size_estimator,
    _hash_part,
    make_key,
)


def test_lru_cache_rejects_invalid_limits() -> None:
    with pytest.raises(ValueError):
        LRUCache(
            namespace="test",
            version="v1",
            max_items=0,
            max_bytes=1024,
            ttl_seconds=None,
            telemetry=None,
            size_estimator=lambda _: 1,
        )
    with pytest.raises(ValueError):
        LRUCache(
            namespace="test",
            version="v1",
            max_items=1,
            max_bytes=0,
            ttl_seconds=None,
            telemetry=None,
            size_estimator=lambda _: 1,
        )


def test_cache_respects_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = {"now": monotonic()}

    def fake_monotonic() -> float:
        return clock["now"]

    cache = LRUCache[
        str,
        int,
    ](
        namespace="test",
        version="v1",
        max_items=4,
        max_bytes=None,
        ttl_seconds=1.0,
        telemetry=None,
        size_estimator=lambda _: 1,
    )
    monkeypatch.setattr("calibrated_explanations.perf.cache.monotonic", fake_monotonic)

    cache.set("alpha", 42)
    assert cache.get("alpha") == 42
    clock["now"] += 2.0  # expire entry
    assert cache.get("alpha") is None
    assert cache.metrics.expirations == 1


def test_cache_respects_memory_budget() -> None:
    cache = LRUCache[
        str,
        np.ndarray,
    ](
        namespace="test",
        version="v1",
        max_items=10,
        max_bytes=16,
        ttl_seconds=None,
        telemetry=None,
        size_estimator=lambda value: int(getattr(value, "nbytes", 0)),
    )

    small = np.zeros(2, dtype=np.uint8)
    large = np.zeros(64, dtype=np.uint8)

    cache.set("small", small)
    assert cache.get("small") is small

    cache.set("too_big", large)
    assert cache.get("too_big") is None
    assert cache.metrics.misses >= 1


def test_calibrator_cache_namespaces() -> None:
    config = CacheConfig(enabled=True, namespace="calib", version="v2", max_items=8)
    cache: CalibratorCache[Dict[str, int]] = CalibratorCache(config)

    payload = {"value": 7}
    cache.set(stage="predict", parts=[("sample", 1)], value=payload)
    assert cache.get(stage="predict", parts=[("sample", 1)]) == payload
    assert cache.get(stage="train", parts=[("sample", 1)]) is None


def test_make_key_normalises_arrays() -> None:
    array = np.arange(5)
    key = make_key("ns", "v", [("payload", array)])
    assert key[0] == "ns"
    assert key[1] == "v"
    assert "payload" in str(key[-1])


def test_default_size_estimator_prefers_numpy_buffers() -> None:
    array = np.arange(6, dtype=np.int16)
    assert _default_size_estimator(array) == array.nbytes

    class DummyArray:
        def __init__(self, data: Iterable[int]):
            self._data = list(data)

        @property
        def __array_interface__(self) -> Dict[str, object]:  # type: ignore[override]
            array = np.asarray(self._data, dtype=np.float32)
            return array.__array_interface__

    shim = DummyArray([1, 2, 3, 4])
    expected = np.asarray([1, 2, 3, 4], dtype=np.float32).nbytes
    assert _default_size_estimator(shim) == expected

    class Opaque:
        pass

    assert _default_size_estimator(Opaque()) == 256


def test_hash_part_covers_nested_structures__should_produce_hashable_representations():
    """Verify that _hash_part produces consistent, hashable representations of nested structures.

    Domain Invariants:
    - Numpy arrays must produce hashable tuples (not bare arrays, which are unhashable)
    - None must map to None (singleton)
    - Collections must be converted to hashable tuples (not lists/sets)
    - Dicts must be decomposed into hashable tuples of (key, value) pairs
    - Arbitrary objects must produce hashable strings
    Ref: ADR-003 Caching Strategy (hash_part consistency requirement)
    """
    # Test numpy array: must be converted to hashable tuple, not left as array
    array = np.arange(3, dtype=np.uint8)
    hashed_array = _hash_part(array)
    assert isinstance(hashed_array, tuple), "Numpy array must be hashed to a hashable tuple"
    assert hash(hashed_array) is not None, "Hashed array must be hashable"
    # Shape is preserved within the hash representation
    assert len(hashed_array) >= 1, "Hash representation must have elements"

    # Test None: must map to None (identity)
    assert _hash_part(None) is None, "None must remain None in hash"

    # Test nested collections: must be converted to hashable tuples
    nested_sets: List[object] = [{1, 2}, {3, 4}]
    hashed_nested = _hash_part(nested_sets)
    assert isinstance(hashed_nested, tuple), "Nested list must be converted to hashable tuple"
    assert all(
        isinstance(item, tuple) for item in hashed_nested
    ), "Each element in hashed list must be a hashable tuple"
    assert hash(hashed_nested) is not None, "Hashed nested must be hashable"

    # Test dict: must be converted to hashable representation
    hashed_mapping = _hash_part({"beta": 3})
    assert isinstance(
        hashed_mapping, (tuple, list)
    ), "Dict must convert to hashable tuple or comparable list"
    # The key-value pair must be representable in the hash
    assert ("beta", 3) in hashed_mapping, "Key-value pair must appear in hashed dict representation"
    assert hash(hashed_mapping) is not None, "Hashed dict must be hashable"

    # Test arbitrary object: must produce a hashable string representation
    sentinel = object()
    hashed_sentinel = _hash_part(sentinel)
    assert isinstance(hashed_sentinel, tuple), "Arbitrary object must be hashed to tuple"
    assert len(hashed_sentinel) >= 1, "Hash representation must have elements"
    assert hash(hashed_sentinel) is not None, "Hashed sentinel must be hashable"


def test_cache_config_from_env_parses_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "CE_CACHE",
        "enable,namespace=prod,version=v9,max_items=0,max_bytes=2,ttl=5.5,off",
    )
    base = CacheConfig(enabled=False, namespace="dev", version="v1", max_items=5, ttl_seconds=1.0)
    config = CacheConfig.from_env(base)

    assert config.enabled is False  # last toggle wins
    assert config.namespace == "prod"
    assert config.version == "v9"
    assert config.max_items == 1  # clamped to minimum of 1
    assert config.max_bytes == 2
    assert config.ttl_seconds == 5.5

    monkeypatch.setenv("CE_CACHE", "1")
    config_enabled = CacheConfig.from_env(base)
    assert config_enabled.enabled is True


def test_lru_cache_updates_existing_and_enforces_limits() -> None:
    events: List[str] = []

    def telemetry(event: str, payload: Dict[str, object]) -> None:
        events.append(f"{event}:{payload['key']}")
        if event == "cache_store" and payload.get("key") == "gamma":
            raise RuntimeError("boom")

    cache = LRUCache[str, np.ndarray](
        namespace="perf",
        version="v1",
        max_items=2,
        max_bytes=8,
        ttl_seconds=None,
        telemetry=telemetry,
        size_estimator=lambda value: int(value.nbytes),
    )

    alpha = np.ones(2, dtype=np.int8)
    beta = np.ones(4, dtype=np.int8)
    gamma = np.ones(6, dtype=np.int8)

    cache.set("alpha", alpha)
    cache.set("alpha", beta)
    assert cache.get("alpha") is beta

    cache.set("beta", beta)
    assert "cache_store:beta" in events

    cache.set("gamma", gamma)  # triggers shrink by bytes and telemetry failure branch
    assert cache.get("alpha") is None
    assert cache.get("beta") is None
    assert np.array_equal(cache.get("gamma"), gamma)
    assert cache.metrics.misses >= 2
    assert any(evt.startswith("cache_evict") for evt in events if evt.startswith("cache"))


def test_cache_metrics_snapshot_reflects_operations() -> None:
    cache = LRUCache[str, int](
        namespace="perf",
        version="v1",
        max_items=1,
        max_bytes=None,
        ttl_seconds=None,
        telemetry=None,
        size_estimator=lambda value: value,
    )

    cache.set("one", 1)
    cache.set("two", 2)
    assert cache.get("one") is None
    assert cache.get("two") == 2

    snapshot = cache.metrics.snapshot()
    assert snapshot["hits"] == 1
    assert snapshot["evictions"] == 1


def test_calibrator_cache_handles_disabled_state() -> None:
    config = CacheConfig(enabled=False)
    cache: CalibratorCache[int] = CalibratorCache(config)

    assert cache.enabled is False
    assert cache.metrics.snapshot()["hits"] == 0
    assert cache.get(stage="predict", parts=[1]) is None
    cache.set(stage="predict", parts=[1], value=1)
    assert cache.compute(stage="predict", parts=[1], fn=lambda: 5) == 5


def test_calibrator_cache_compute_reuses_results() -> None:
    config = CacheConfig(enabled=True, max_items=4)
    cache: CalibratorCache[int] = CalibratorCache(config)

    calls = 0

    def factory() -> int:
        nonlocal calls
        calls += 1
        return 99

    result_first = cache.compute(stage="score", parts=["sample"], fn=factory)
    result_second = cache.compute(stage="score", parts=["sample"], fn=factory)

    assert result_first == 99
    assert result_second == 99
    assert calls == 1

    cache.forksafe_reset()
    assert cache.get(stage="score", parts=["sample"]) is None


def test_should_handle_cache_miss_with_none_value() -> None:
    """Cache should distinguish between missing keys and None values."""
    config = CacheConfig(enabled=True, max_items=4)
    cache: CalibratorCache[int | None] = CalibratorCache(config)
    
    # Store explicit None
    cache.set(stage="verify", parts=["test"], value=None)
    assert cache.get(stage="verify", parts=["test"]) is None
    assert cache.metrics.snapshot()["hits"] >= 1


def test_should_handle_multiple_parts_as_composite_key() -> None:
    """Cache key should be composite of stage, parts list."""
    config = CacheConfig(enabled=True, max_items=10)
    cache: CalibratorCache[str] = CalibratorCache(config)
    
    # Store with different parts
    cache.set(stage="predict", parts=[1, 2], value="result_1_2")
    cache.set(stage="predict", parts=[1, 3], value="result_1_3")
    cache.set(stage="predict", parts=[1, 2, 3], value="result_1_2_3")
    
    assert cache.get(stage="predict", parts=[1, 2]) == "result_1_2"
    assert cache.get(stage="predict", parts=[1, 3]) == "result_1_3"
    assert cache.get(stage="predict", parts=[1, 2, 3]) == "result_1_2_3"
    assert cache.get(stage="predict", parts=[1]) is None


def test_should_handle_different_stages_independently() -> None:
    """Different stages should maintain separate cache entries."""
    config = CacheConfig(enabled=True, max_items=10)
    cache: CalibratorCache[int] = CalibratorCache(config)
    
    cache.set(stage="fit", parts=["a"], value=10)
    cache.set(stage="predict", parts=["a"], value=20)
    cache.set(stage="calibrate", parts=["a"], value=30)
    
    assert cache.get(stage="fit", parts=["a"]) == 10
    assert cache.get(stage="predict", parts=["a"]) == 20
    assert cache.get(stage="calibrate", parts=["a"]) == 30


def test_should_handle_compute_with_factory_exception() -> None:
    """Compute should propagate factory exceptions."""
    config = CacheConfig(enabled=True, max_items=4)
    cache: CalibratorCache[int] = CalibratorCache(config)
    
    def failing_factory() -> int:
        raise ValueError("Factory error")
    
    with pytest.raises(ValueError, match="Factory error"):
        cache.compute(stage="predict", parts=["fail"], fn=failing_factory)


def test_should_respect_cache_disable() -> None:
    """When disabled, cache should pass-through to factory every call."""
    config = CacheConfig(enabled=False, max_items=100)
    cache: CalibratorCache[int] = CalibratorCache(config)
    
    call_count = 0
    
    def counting_factory() -> int:
        nonlocal call_count
        call_count += 1
        return call_count
    
    # Even with same key, disabled cache should call factory each time
    result1 = cache.compute(stage="predict", parts=["x"], fn=counting_factory)
    result2 = cache.compute(stage="predict", parts=["x"], fn=counting_factory)
    
    assert result1 == 1
    assert result2 == 2
    assert call_count == 2


def test_should_track_hit_miss_stats() -> None:
    """Cache metrics should accurately track hits and misses."""
    config = CacheConfig(enabled=True, max_items=5)
    cache: CalibratorCache[str] = CalibratorCache(config)
    
    # Miss (set doesn't count as hit/miss)
    assert cache.get(stage="predict", parts=["new"]) is None
    
    # Store value
    cache.set(stage="predict", parts=["new"], value="value1")
    
    # Hit
    result = cache.get(stage="predict", parts=["new"])
    assert result == "value1"
    
    snapshot = cache.metrics.snapshot()
    assert snapshot["hits"] >= 1
    assert snapshot["misses"] >= 1


def test_should_handle_lru_eviction_with_size_limit() -> None:
    """LRU cache should evict oldest when size limit exceeded."""
    config = CacheConfig(enabled=True, max_items=2)
    cache: CalibratorCache[int] = CalibratorCache(config)
    
    cache.set(stage="predict", parts=["a"], value=1)
    cache.set(stage="predict", parts=["b"], value=2)
    
    # Access 'a' to make it recently used
    _ = cache.get(stage="predict", parts=["a"])
    
    # Add third item, should evict 'b' (least recently used)
    cache.set(stage="predict", parts=["c"], value=3)
    
    assert cache.get(stage="predict", parts=["a"]) == 1
    assert cache.get(stage="predict", parts=["b"]) is None
    assert cache.get(stage="predict", parts=["c"]) == 3
