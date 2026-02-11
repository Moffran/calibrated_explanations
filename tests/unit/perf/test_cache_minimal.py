def test_cache_module_basic_classes():
    import importlib

    cache_module = importlib.import_module("calibrated_explanations.cache.cache")

    # Prefer the minimal backend in cachemodule.cachetools when present
    # otherwise fall back to the wrapper LRUCache that requires kwargs.
    LRU = getattr(getattr(cache_module, "cachetools", {}), "LRUCache", None) or getattr(
        cache_module, "LRUCache", None
    )
    TTL = getattr(getattr(cache_module, "cachetools", {}), "TTLCache", None) or getattr(
        cache_module, "TTLCache", None
    )

    assert LRU is not None

    # instantiate LRU tolerantly: some fallback implementations accept no args
    try:
        cache = LRU(2)
    except TypeError:
        try:
            cache = LRU()
        except TypeError:
            cache = LRU(maxsize=2)

    cache["key1"] = "value1"
    assert cache["key1"] == "value1"

    # TTL may not be available in all environments; if present ensure basic API
    if TTL is not None:
        try:
            tc = TTL(2, 0.1)
        except TypeError:
            try:
                tc = TTL()
            except TypeError:
                tc = TTL(maxsize=2, ttl=0.1)
        tc["k"] = "v"
        assert tc.get("k") == "v"
