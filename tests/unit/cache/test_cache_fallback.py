import sys
import time
from unittest.mock import patch

import numpy as np
import pytest

from calibrated_explanations.cache.cache import (
    LRUCache,
    make_key,
)


def test_cache_fallback_logic(monkeypatch):
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
            # Avoid real sleep during tests
            monkeypatch.setattr(time, "sleep", lambda _s: None)
            assert cache_ttl["a"] == 1
            time.sleep(0.2)
            with pytest.raises(KeyError):
                _ = cache_ttl["a"]

            cache_ttl["b"] = 2
            assert cache_ttl.get("b") == 2
            time.sleep(0.2)
            assert cache_ttl.get("b") is None

            # Test additional operations on fallback
            cache_ttl["c"] = 3
            cache_ttl.pop("c")
            assert "c" not in cache_ttl

            cache_ttl["d"] = 4
            cache_ttl.clear()
            assert len(cache_ttl) == 0
    finally:
        sys.modules.clear()
        sys.modules.update(old_modules)

    # Re-import to restore normal state
    if "calibrated_explanations.cache.cache" in sys.modules:
        del sys.modules["calibrated_explanations.cache.cache"]


def test_make_key_should_include_namespace_version_and_parts():
    """make_key should compose a deterministic namespaced cache key."""
    parts = [np.array([1.0, 2.0]), {"alpha": 1}]
    key_one = make_key("ns", "v1", parts)
    key_two = make_key("ns", "v1", parts)

    assert key_one == key_two
    assert key_one[0] == "ns"
    assert key_one[1] == "v1"


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
