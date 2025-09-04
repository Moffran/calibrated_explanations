from calibrated_explanations.perf import LRUCache, make_key, JoblibBackend, sequential_map


def test_lru_cache_basic_eviction():
    cache = LRUCache[int, str](max_items=2)
    cache.set(1, "a")
    cache.set(2, "b")
    assert cache.get(1) == "a"  # touch 1 to make it MRU
    cache.set(3, "c")  # evicts key 2
    assert cache.get(2) is None
    assert cache.get(1) == "a"
    assert cache.get(3) == "c"


def test_make_key_determinism():
    k1 = make_key(["x", 1, (2, 3)])
    k2 = make_key(["x", 1, (2, 3)])
    assert k1 == k2


def test_parallel_joblib_fallback_and_sequential():
    data = [1, 2, 3, 4]

    def fn(x):
        return x * x

    # sequential
    assert sequential_map(fn, data) == [1, 4, 9, 16]
    # joblib backend (works even if joblib missing)
    jb = JoblibBackend()
    out = jb.map(fn, data, workers=1)
    assert out == [1, 4, 9, 16]
