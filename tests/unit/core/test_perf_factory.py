from types import SimpleNamespace


from calibrated_explanations.perf import PerfFactory, from_config


def test_perf_factory_defaults_sequential_map():
    pf = PerfFactory()
    # cache disabled by default
    assert pf.make_cache() is None
    backend = pf.make_parallel_backend()
    assert hasattr(backend, "map")
    res = backend.map(lambda x: x * 2, [1, 2, 3])
    assert res == [2, 4, 6]


def test_perf_factory_cache_eviction():
    pf = PerfFactory(cache_enabled=True, cache_max_items=2)
    cache = pf.make_cache()
    assert cache is not None
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)
    # 'a' should have been evicted (max_items == 2)
    assert "a" not in cache
    assert len(cache) == 2


def test_from_config_factory_behavior():
    cfg = SimpleNamespace(
        perf_cache_enabled=True,
        perf_cache_max_items=3,
        perf_parallel_enabled=False,
        perf_parallel_backend="auto",
    )
    pf = from_config(cfg)
    cache = pf.make_cache()
    assert cache is not None and cache.max_items == 3
    backend = pf.make_parallel_backend()
    assert hasattr(backend, "map")


def test_lru_eviction_order():
    # ensure least-recently-used eviction behavior
    pf = PerfFactory(cache_enabled=True, cache_max_items=2)
    cache = pf.make_cache()
    assert cache is not None
    cache.set("a", 1)
    cache.set("b", 2)
    # touch 'a' so 'b' should be evicted when we add 'c'
    assert cache.get("a") == 1
    cache.set("c", 3)
    assert "b" not in cache
    assert "a" in cache and "c" in cache


def test_joblib_backend_fallback(monkeypatch):
    # simulate missing joblib: ensure JoblibBackend gracefully falls back to sequential
    import sys

    saved = sys.modules.get("joblib")
    if "joblib" in sys.modules:
        monkeypatch.setitem(sys.modules, "joblib", None)
    try:
        from calibrated_explanations.perf.parallel import JoblibBackend

        jb = JoblibBackend()
        res = jb.map(lambda x: x + 1, [1, 2, 3])
        assert res == [2, 3, 4]
    finally:
        if saved is not None:
            monkeypatch.setitem(sys.modules, "joblib", saved)
