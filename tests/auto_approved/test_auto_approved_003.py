from calibrated_explanations.cache.cache import hash_part, make_key, default_size_estimator
import numpy as np


def test_cache_hash_and_size_estimator():
    # simple scalars
    assert hash_part(1) == 1
    assert isinstance(hash_part((1, 2)), tuple)
    arr = np.arange(10, dtype=np.int64)
    h = hash_part(arr)
    assert isinstance(h, tuple)
    # size estimator handles numpy arrays
    assert default_size_estimator(arr) > 0
    # make_key composes a tuple key
    key = make_key("ns", "v1", [1, "a", arr])
    assert key[0] == "ns" and key[1] == "v1"
