import importlib
import warnings

from calibrated_explanations.cache import cache as canonical


def test_perf_cache_shim_warns_and_forwards(monkeypatch):
    monkeypatch.syspath_prepend(".")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        perf_cache = importlib.reload(importlib.import_module("calibrated_explanations.perf.cache"))
    assert any(isinstance(w.message, DeprecationWarning) for w in caught)
    assert perf_cache.LRUCache is canonical.LRUCache
    assert perf_cache.CacheConfig is canonical.CacheConfig
    assert perf_cache.CalibratorCache is canonical.CalibratorCache
    assert perf_cache._default_size_estimator is canonical._default_size_estimator
    assert perf_cache._hash_part is canonical._hash_part
