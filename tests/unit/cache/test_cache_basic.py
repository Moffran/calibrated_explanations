import os
import numpy as np

from calibrated_explanations.cache.cache import CacheConfig, default_size_estimator


def test_cacheconfig_from_env_parsing(monkeypatch):
    # use canonical '1' token to enable the cache per parser expectations
    monkeypatch.setenv("CE_CACHE", "1,namespace=testns,version=v2,max_items=10,max_bytes=1024,ttl=1.0")
    cfg = CacheConfig.from_env()
    assert cfg.enabled is True
    assert cfg.namespace == "testns"
    assert cfg.version == "v2"
    assert cfg.max_items == 10
    assert cfg.max_bytes == 1024


def test_default_size_estimator_array_like():
    arr = np.ones((10, 10), dtype=np.float64)
    size = default_size_estimator(arr)
    assert isinstance(size, int) and size > 0
