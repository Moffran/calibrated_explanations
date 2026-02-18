from calibrated_explanations.cache.cache import (
    CacheConfig,
    LRUCache,
    default_size_estimator,
)


def test_lru_cache_hit_path_and_cache_config_default_estimator_marker():
    events = []

    def telemetry(event, payload):
        events.append((event, payload["namespace"]))

    cache = LRUCache(
        namespace="gap",
        version="v1",
        max_items=4,
        max_bytes=None,
        ttl_seconds=None,
        telemetry=telemetry,
        size_estimator=lambda _: 3,
    )
    cache.set(("k",), "value")
    assert cache.get(("k",)) == "value"
    assert cache.metrics.hits == 1
    assert ("cache_hit", "gap") in events

    cfg = CacheConfig(size_estimator=default_size_estimator)
    reduce_fn, reduce_args = cfg.__reduce__()
    assert reduce_fn.__name__ == "_reconstruct_cache_config"
    assert reduce_args[0]["size_estimator"] == "__default_size_estimator__"
