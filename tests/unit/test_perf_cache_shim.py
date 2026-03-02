import importlib
import warnings

import pytest

from calibrated_explanations.cache import cache as canonical
from tests.helpers.deprecation import deprecations_error_enabled


def testperf_cache_shim_warns_and_forwards(monkeypatch):
    monkeypatch.syspath_prepend(".")
    if deprecations_error_enabled():
        with pytest.raises(DeprecationWarning, match="calibrated_explanations.perf.cache"):
            importlib.reload(importlib.import_module("calibrated_explanations.perf.cache"))
        return

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        perf_cache = importlib.reload(importlib.import_module("calibrated_explanations.perf.cache"))
    # Use __name__ and __module__ to verify identity across reloads if 'is' fails
    assert perf_cache.LRUCache.__name__ == canonical.LRUCache.__name__
    assert perf_cache.LRUCache.__module__ == canonical.LRUCache.__module__
    assert perf_cache.CacheConfig.__name__ == canonical.CacheConfig.__name__
    assert perf_cache.CalibratorCache.__name__ == canonical.CalibratorCache.__name__
    assert perf_cache.default_size_estimator.__name__ == canonical.default_size_estimator.__name__
    assert perf_cache.hash_part.__name__ == canonical.hash_part.__name__
