import pickle

import numpy as np

from calibrated_explanations.cache.cache import (
    CacheConfig,
    CacheMetrics,
    LRUCache,
    default_size_estimator,
)
from calibrated_explanations.core.config_manager import ConfigManager


def test_pickle_and_restore_cache_config_and_metrics():
    cfg = CacheConfig(
        enabled=True, namespace="ns", version="v2", max_items=2, max_bytes=1024, ttl_seconds=None
    )
    # default estimator marker should round-trip via the reduce implementation
    data = pickle.loads(pickle.dumps(cfg))
    # The reconstructed object may come from the module-level helper and
    # therefore not be identity-equal to the imported `CacheConfig` symbol.
    # Assert on public attributes instead of strict isinstance checks.
    assert getattr(data, "enabled", None) is True
    assert getattr(data, "namespace", None) == "ns"
    assert getattr(data, "version", None) == "v2"

    m = CacheMetrics(hits=1, misses=2, sets=3, evictions=4, expirations=5, resets=6)
    m2 = pickle.loads(pickle.dumps(m))
    # The same identity caveat applies to CacheMetrics; check field values.
    assert (
        getattr(m2, "hits", None),
        getattr(m2, "misses", None),
        getattr(m2, "sets", None),
        getattr(m2, "evictions", None),
        getattr(m2, "expirations", None),
        getattr(m2, "resets", None),
    ) == (
        1,
        2,
        3,
        4,
        5,
        6,
    )


def test_estimate_numpy_and_fallback_size():
    arr = np.zeros((8, 8), dtype=np.float64)
    assert default_size_estimator(arr) == arr.nbytes

    class Dummy:
        pass

    # objects without array interface or nbytes should fall back to constant
    assert default_size_estimator(Dummy()) == 256


def test_wrap_none_and_evict_least_recently_used():
    cache = LRUCache(
        namespace="t",
        version="v",
        max_items=1,
        max_bytes=None,
        ttl_seconds=None,
        telemetry=None,
        size_estimator=default_size_estimator,
    )
    cache.set("a", None)
    assert cache.get("a") is None
    assert cache.metrics.sets == 1

    # inserting a second item forces eviction under capacity=1
    cache.set("b", 123)
    assert "a" not in cache
    assert cache.metrics.evictions >= 1


def test_parse_cache_config_from_env(monkeypatch):
    monkeypatch.delenv("CE_CACHE", raising=False)
    cfg = CacheConfig.from_env(None, config_manager=ConfigManager.from_sources())
    assert isinstance(cfg, CacheConfig)

    monkeypatch.setenv("CE_CACHE", "namespace=foo,version=v9,max_items=3,max_bytes=1024,ttl=5")
    cfg2 = CacheConfig.from_env(None, config_manager=ConfigManager.from_sources())
    assert cfg2.namespace == "foo"
    assert cfg2.version == "v9"
    assert cfg2.max_items == 3
    assert cfg2.max_bytes == 1024
    assert cfg2.ttl_seconds == 5.0
