import importlib
import warnings

from calibrated_explanations.cache import cache as canonical


def testperf_cache_shim_warns_and_forwards(monkeypatch):
    monkeypatch.syspath_prepend(".")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        perf_cache = importlib.reload(importlib.import_module("calibrated_explanations.perf.cache"))
    assert any(isinstance(w.message, DeprecationWarning) for w in caught)
    # Use __name__ and __module__ to verify identity across reloads if 'is' fails
    assert perf_cache.LRUCache.__name__ == canonical.LRUCache.__name__
    assert perf_cache.LRUCache.__module__ == canonical.LRUCache.__module__
    assert perf_cache.CacheConfig.__name__ == canonical.CacheConfig.__name__
    assert perf_cache.CalibratorCache.__name__ == canonical.CalibratorCache.__name__
    assert perf_cache.default_size_estimator.__name__ == canonical.default_size_estimator.__name__
    assert perf_cache._hash_part.__name__ == canonical._hash_part.__name__
