"""Tests for the lightweight LRU cache utilities."""

from __future__ import annotations

import pytest

from calibrated_explanations.perf.cache import LRUCache, make_key


@pytest.mark.parametrize("max_items", [0, -1])
def test_lru_cache_rejects_non_positive_capacity(max_items: int) -> None:
    """LRUCache enforces a strictly positive capacity."""
    with pytest.raises(ValueError, match="max_items must be positive"):
        LRUCache(max_items=max_items)


def test_lru_cache_updates_existing_entries_without_growth() -> None:
    """Updating an existing key keeps the cache size stable and bumps it to MRU."""
    cache: LRUCache[str, int] = LRUCache(max_items=2)
    cache.set("alpha", 1)
    cache.set("beta", 2)
    cache.set("alpha", 10)  # update existing key; should stay at size 2

    assert len(cache._store) == 2  # internal OrderedDict size remains bounded
    assert cache.get("alpha") == 10
    assert list(cache._store.keys())[-1] == "alpha"


def test_lru_cache_get_default_preserves_state() -> None:
    """Missing lookups return the provided default and do not mutate the cache."""
    cache: LRUCache[str, int] = LRUCache(max_items=1)
    cache.set("exists", 42)

    sentinel = object()
    assert cache.get("missing", sentinel) is sentinel
    assert "missing" not in cache._store
    assert len(cache) == 1  # __len__ hook


def test_make_key_handles_generators() -> None:
    """``make_key`` coerces any iterable (including generators) to a tuple."""
    parts = (i * 2 for i in range(3))
    key = make_key(parts)

    assert key == (0, 2, 4)
    assert isinstance(key, tuple)


def test_lru_cache_updates_existing_keys_without_eviction() -> None:
    """Updating a cached key should refresh recency and preserve the entry."""
    cache: LRUCache[str, int] = LRUCache(max_items=2)
    cache.set("a", 1)
    cache.set("b", 2)

    # Updating an existing key hits the "already present" branch and moves it to the end.
    cache.set("a", 42)

    # Adding a new item now evicts the other key rather than the refreshed one.
    cache.set("c", 3)

    assert cache.get("a") == 42
    assert cache.get("b") is None
    assert cache.get("c") == 3


def test_cache_helpers_cover_len_contains_and_make_key() -> None:
    """Convenience helpers should behave consistently for typical usage."""
    cache: LRUCache[str, str] = LRUCache(max_items=1)
    cache.set("feature", "value")

    assert "feature" in cache
    assert len(cache) == 1
    assert make_key(["feature", 1]) == ("feature", 1)


def test_lru_cache_evicts_least_recently_used_entry():
    cache = LRUCache(max_items=2)
    cache.set("a", 1)
    cache.set("b", 2)
    # Touch "a" so it becomes most recently used.
    assert cache.get("a") == 1
    cache.set("c", 3)

    assert cache.get("a") == 1
    assert cache.get("c") == 3
    # "b" should have been evicted because it was the least recently used.
    assert cache.get("b") is None


def test_lru_cache_get_returns_default_when_missing():
    cache = LRUCache(max_items=1)
    assert cache.get("missing") is None
    assert cache.get("missing", default=42) == 42


def test_make_key_returns_tuple_of_hashable_parts():
    key = make_key(["a", 1, ("nested", 2)])
    assert key == ("a", 1, ("nested", 2))
    assert isinstance(key, tuple)
