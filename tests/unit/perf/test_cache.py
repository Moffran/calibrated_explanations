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
