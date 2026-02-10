import pickle

import numpy as np

from calibrated_explanations.cache import cache as cache_mod
from calibrated_explanations.cache.explanation_cache import ExplanationCacheFacade


def boom_size_estimator(_value):
    raise RuntimeError("nope")


def test_default_size_estimator_with_numpy():
    arr = np.arange(10, dtype=np.int64)
    size = cache_mod.default_size_estimator(arr)
    assert isinstance(size, int)
    assert size == arr.nbytes


def test_default_size_estimator_fallback_for_plain_object():
    class Dummy:
        pass

    obj = Dummy()
    size = cache_mod.default_size_estimator(obj)
    assert size == 256


def test_hash_part_various_types():
    assert cache_mod.hash_part(None) is None
    assert cache_mod.hash_part(123) == 123
    assert cache_mod.hash_part("abc") == "abc"

    arr = np.array([[1, 2], [3, 4]], dtype=np.int32)
    token = cache_mod.hash_part(arr)
    assert isinstance(token, tuple) and token[0] == "nd"

    lst = [3, 1, 2]
    assert cache_mod.hash_part(lst) == tuple(sorted(cache_mod.hash_part(x) for x in lst))

    s = {"b", "a"}
    hp = cache_mod.hash_part(s)
    assert isinstance(hp, tuple)

    mapping = {"x": 1, "y": 2}
    mp = cache_mod.hash_part(mapping)
    assert isinstance(mp, tuple) and all(isinstance(t, tuple) for t in mp)


def test_make_key_and_metrics_snapshot(monkeypatch):
    parts = ["a", 1, None]
    key = cache_mod.make_key("ns", "v1", parts)
    assert key[0] == "ns" and key[1] == "v1"

    m = cache_mod.CacheMetrics()
    m.hits = 1
    m.misses = 2
    snap = m.snapshot()
    assert snap["hits"] == 1 and snap["misses"] == 2


def test_cacheconfig_from_env_and_calibrator_cache(monkeypatch, tmp_path):
    # Ensure env parsing toggles enabled
    monkeypatch.setenv("CE_CACHE", "1")
    cfg = cache_mod.CacheConfig.from_env()
    assert cfg.enabled is True

    monkeypatch.setenv("CE_CACHE", "namespace=foo,version=v2,max_items=3,max_bytes=1024,ttl=0.01")
    cfg2 = cache_mod.CacheConfig.from_env()
    assert cfg2.namespace == "foo"
    assert cfg2.version == "v2"
    assert cfg2.max_items == 3
    assert cfg2.max_bytes == 1024
    assert cfg2.ttl_seconds is not None

    # Instantiate CalibratorCache and exercise set/get/compute/flush/reset
    cfg2.enabled = True
    cal = cache_mod.CalibratorCache(cfg2)
    assert cal.enabled

    # Simple set/get
    cal.set(stage="s", parts=("p",), value={"v": 1})
    val = cal.get(stage="s", parts=("p",))
    assert val == {"v": 1}

    # compute should return cached value
    called = {}

    def producer():
        called["ok"] = True
        return "computed"

    res = cal.compute(stage="s", parts=("p",), fn=producer)
    assert res == {"v": 1}

    # new key compute
    res2 = cal.compute(stage="s", parts=("q",), fn=producer)
    assert res2 == "computed" and called.get("ok")

    # flush clears entries
    cal.flush()
    assert cal.get(stage="s", parts=("p",)) is None

    # reset_version modifies the version string
    old = cal.version
    cal.reset_version("newv")
    assert cal.version == "newv"

    # forksafe_reset should not error
    cal.forksafe_reset()

    # pickling roundtrip recreates lock
    state = pickle.dumps(cal)
    new = pickle.loads(state)
    assert hasattr(new, "__dict__")


def test_default_size_estimator_error_paths():
    class BadNbytes:
        @property
        def nbytes(self):
            class BadInt:
                def __int__(self):
                    raise ValueError("boom")

            return BadInt()

        @property
        def __array_interface__(self):
            return "invalid-array-interface"

    size = cache_mod.default_size_estimator(BadNbytes())
    assert size == 256


def test_lru_cache_helpers_and_pickle_roundtrip():
    cache = cache_mod.LRUCache(
        namespace="ns",
        version="v1",
        max_items=8,
        max_bytes=None,
        ttl_seconds=None,
        telemetry=None,
        size_estimator=cache_mod.default_size_estimator,
    )

    assert cache.telemetry_handler is None
    cache.telemetry_handler = lambda *_args, **_kwargs: None
    assert callable(cache.telemetry_handler)

    cache = cache_mod.LRUCache(
        namespace="ns",
        version="v1",
        max_items=8,
        max_bytes=None,
        ttl_seconds=None,
        telemetry=None,
        size_estimator=boom_size_estimator,
    )
    cache.set(("k",), {"v": 1})
    assert cache.get(("k",)) == {"v": 1}

    state = pickle.dumps(cache)
    restored = pickle.loads(state)
    assert restored.get(("k",)) == {"v": 1}


def test_lru_cache_telemetry_errors_and_none_values():
    def bad_telemetry(_event, _payload):
        raise RuntimeError("boom")

    cache = cache_mod.LRUCache(
        namespace="ns",
        version="v1",
        max_items=8,
        max_bytes=None,
        ttl_seconds=None,
        telemetry=bad_telemetry,
        size_estimator=cache_mod.default_size_estimator,
    )
    cache.set(("none",), None)
    assert cache.get(("none",)) is None


def test_explanation_cache_facade_disabled():
    facade = ExplanationCacheFacade(None)
    assert facade.enabled is False

    assert facade.get_calibration_summaries(explainer_id="x", x_cal_hash="h") is None
    assert facade.get_feature_names_cache(explainer_id="x") is None

    called = {}

    def compute_fn():
        called["ok"] = True
        return ({0: {"a": 1}}, {0: np.array([1.0])})

    result = facade.compute_calibration_summaries(
        explainer_id="x", x_cal_hash="h", compute_fn=compute_fn
    )
    assert called.get("ok") is True
    assert result[0][0]["a"] == 1

    facade.invalidate_all()
    facade.reset_version("v2")


def test_explanation_cache_facade_enabled_roundtrip():
    cfg = cache_mod.CacheConfig(enabled=True)
    cal = cache_mod.CalibratorCache(cfg)
    facade = ExplanationCacheFacade(cal)
    assert facade.enabled is True

    categorical_counts = {0: {"a": 1}}
    numeric_sorted = {0: np.array([1.0, 2.0])}

    facade.set_calibration_summaries(
        explainer_id="x",
        x_cal_hash="h",
        categorical_counts=categorical_counts,
        numeric_sorted=numeric_sorted,
    )
    cached = facade.get_calibration_summaries(explainer_id="x", x_cal_hash="h")
    assert cached == (categorical_counts, numeric_sorted)

    feature_names = ("f1", "f2")
    facade.set_feature_names_cache(explainer_id="x", feature_names=feature_names)
    assert facade.get_feature_names_cache(explainer_id="x") == feature_names

    facade.invalidate_all()
    assert facade.get_calibration_summaries(explainer_id="x", x_cal_hash="h") is None

    facade.reset_version("v2")
    assert cal.version == "v2"


def test_calibrator_cache_disabled_noops():
    cfg = cache_mod.CacheConfig(enabled=False)
    cal = cache_mod.CalibratorCache(cfg)
    assert cal.enabled is False
    assert cal.metrics.snapshot()["hits"] == 0

    assert cal.get(stage="s", parts=("p",)) is None
    cal.set(stage="s", parts=("p",), value=1)
    assert cal.get(stage="s", parts=("p",)) is None


def test_lru_cache_eviction_and_budget_paths():
    oversize_cache = cache_mod.LRUCache(
        namespace="ns",
        version="v1",
        max_items=8,
        max_bytes=1,
        ttl_seconds=None,
        telemetry=None,
        size_estimator=lambda _value: 10,
    )
    oversize_cache.set(("big",), "value")
    assert oversize_cache.get(("big",)) is None

    update_cache = cache_mod.LRUCache(
        namespace="ns",
        version="v1",
        max_items=8,
        max_bytes=10,
        ttl_seconds=None,
        telemetry=None,
        size_estimator=lambda _value: 1,
    )
    update_cache.set(("k",), "a")
    update_cache.set(("k",), "b")
    assert update_cache.get(("k",)) == "b"

    byte_budget_cache = cache_mod.LRUCache(
        namespace="ns",
        version="v1",
        max_items=8,
        max_bytes=1,
        ttl_seconds=None,
        telemetry=None,
        size_estimator=lambda _value: 1,
    )
    byte_budget_cache.set(("k1",), "v1")
    byte_budget_cache.set(("k2",), "v2")
    assert byte_budget_cache.get(("k2",)) == "v2"

    max_items_cache = cache_mod.LRUCache(
        namespace="ns",
        version="v1",
        max_items=1,
        max_bytes=None,
        ttl_seconds=None,
        telemetry=None,
        size_estimator=lambda _value: 1,
    )
    max_items_cache.set(("k1",), "v1")
    max_items_cache.set(("k2",), "v2")
    assert max_items_cache.get(("k2",)) == "v2"
