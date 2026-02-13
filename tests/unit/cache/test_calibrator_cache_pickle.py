import pickle

import numpy as np

from calibrated_explanations.cache.cache import CacheConfig, CalibratorCache


def test_calibrator_cache_pickle_roundtrip():
    cfg = CacheConfig()
    cfg.enabled = True
    cfg.namespace = "ns"
    cfg.version = "v1"
    cfg.max_items = 4
    cache = CalibratorCache(config=cfg)
    # store a numpy value to hit size estimator paths
    cache.set(stage="s", parts=[np.array([1, 2, 3])], value=np.array([1.0, 2.0]))

    dumped = pickle.dumps(cache)
    loaded = pickle.loads(dumped)
    assert hasattr(loaded, "version")
    assert loaded.version == "v1"
    # ensure lock was recreated
    assert hasattr(loaded, "_version_lock")
