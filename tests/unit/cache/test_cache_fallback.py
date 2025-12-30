import sys
import time
from unittest.mock import patch

import numpy as np
import pytest

from calibrated_explanations.cache.cache import (
    CacheConfig,
    CalibratorCache,
    LRUCache,
    default_size_estimator,
    hash_part,
    make_key,
)


def test_cache_fallback_logic():
    # We want to test the fallback classes even if cachetools is installed.
    # They are hidden in the except block and assigned to cachetools shim.

    old_modules = sys.modules.copy()
    if "calibrated_explanations.cache.cache" in sys.modules:
        del sys.modules["calibrated_explanations.cache.cache"]

    try:
        with patch.dict(sys.modules, {"cachetools": None}):
            import calibrated_explanations.cache.cache as cache_mod

            # The fallback classes are in cache_mod.cachetools
            fallback_lru = cache_mod.cachetools.LRUCache
            fallback_ttl = cache_mod.cachetools.TTLCache

            # Test LRUCache fallback
            cache = fallback_lru(2)
            cache["a"] = 1
            cache["b"] = 2
            assert cache["a"] == 1
            cache["c"] = 3
            # "b" should be evicted as "a" was accessed last
            assert "b" not in cache
            assert "a" in cache
            assert "c" in cache

            assert cache.get("a") == 1
            assert cache.get("d", 4) == 4

            cache["a"] = 10
            assert cache["a"] == 10

            # Test TTLCache fallback
            cache_ttl = fallback_ttl(maxsize=10, ttl=0.1)
            cache_ttl["a"] = 1
            assert cache_ttl["a"] == 1
            time.sleep(0.2)
            with pytest.raises(KeyError):
                _ = cache_ttl["a"]

            cache_ttl["b"] = 2
            assert cache_ttl.get("b") == 2
            time.sleep(0.2)
            assert cache_ttl.get("b") is None
    finally:
        sys.modules.clear()
        sys.modules.update(old_modules)

        cache_ttl["c"] = 3
        cache_ttl.pop("c")
        assert "c" not in cache_ttl

        cache_ttl["d"] = 4
        cache_ttl.clear()
        assert len(cache_ttl) == 0

    # Re-import to restore normal state
    if "calibrated_explanations.cache.cache" in sys.modules:
        del sys.modules["calibrated_explanations.cache.cache"]


def test_hash_part_should_produce_stable_hashable_keys():
    """hash_part should normalize complex inputs into stable hashable values."""
    payload = {
        "array": np.array([1, 2, 3], dtype=int),
        "mapping": {"b": 2, "a": 1},
        "set": {"x", "y"},
    }

    first = hash_part(payload)
    second = hash_part(payload)

    assert first == second
    assert isinstance(first, tuple)


def test_make_key_should_include_namespace_version_and_parts():
    """make_key should compose a deterministic namespaced cache key."""
    parts = [np.array([1.0, 2.0]), {"alpha": 1}]
    key_one = make_key("ns", "v1", parts)
    key_two = make_key("ns", "v1", parts)

    assert key_one == key_two
    assert key_one[0] == "ns"
    assert key_one[1] == "v1"


def test_default_size_estimator_should_fallback_when_inputs_unfriendly():
    """default_size_estimator should return a conservative fallback when needed."""

    class ExplodingNbytes:
        @property
        def nbytes(self):
            raise AttributeError("no size")

    assert default_size_estimator(np.zeros((2, 2), dtype=float)) > 0
    assert default_size_estimator(ExplodingNbytes()) == 256


def test_lru_cache_should_track_hits_misses_and_none_values():
    """LRUCache should distinguish missing entries from cached None values."""
    cache = LRUCache(
        namespace="test",
        version="v1",
        max_items=4,
        max_bytes=16,
        ttl_seconds=None,
        telemetry=None,
        size_estimator=lambda _: 1,
    )

    cache.set(("a",), None)
    assert cache.get(("a",)) is None
    assert cache.metrics.hits == 1
    assert cache.metrics.misses == 0

    assert cache.get(("missing",)) is None
    assert cache.metrics.misses == 1


def test_lru_cache_should_evict_when_exceeding_memory_budget():
    """LRUCache should evict the oldest entry when max_bytes is exceeded."""
    cache = LRUCache(
        namespace="budget",
        version="v1",
        max_items=10,
        max_bytes=2,
        ttl_seconds=None,
        telemetry=None,
        size_estimator=lambda _: 1,
    )

    cache.set(("one",), "a")
    cache.set(("two",), "b")
    cache.set(("three",), "c")

    assert cache.get(("one",)) is None
    assert cache.metrics.evictions >= 1


def test_lru_cache_should_skip_oversized_values():
    """LRUCache should refuse to store values larger than the cache budget."""
    cache = LRUCache(
        namespace="oversize",
        version="v1",
        max_items=5,
        max_bytes=1,
        ttl_seconds=None,
        telemetry=None,
        size_estimator=lambda _: 10,
    )

    cache.set(("big",), "payload")

    assert cache.get(("big",)) is None
    assert cache.metrics.misses >= 1


def test_calibrator_cache_should_compute_flush_and_reset_version():
    """CalibratorCache should cache computed values and support invalidation."""
    config = CacheConfig(enabled=True, namespace="cal", version="v1", max_items=5, max_bytes=1000)
    cache = CalibratorCache(config)

    value_one = cache.compute(stage="stage", parts=("a",), fn=lambda: "value")
    value_two = cache.compute(stage="stage", parts=("a",), fn=lambda: "new")

    assert value_one == "value"
    assert value_two == "value"

    cache.flush()
    assert cache.metrics.resets == 1

    cache.reset_version("v2")
    assert cache.version == "v2"


def test_calibrator_cache_should_support_pickle_state_restoration():
    """CalibratorCache should recreate its lock when restoring state."""
    cache = CalibratorCache(CacheConfig(enabled=False))
    state = cache.__getstate__()

    assert "_version_lock" not in state

    restored = CalibratorCache.__new__(CalibratorCache)
    restored.__setstate__(state)

    assert hasattr(restored, "_version_lock")
