import importlib
import pickle
import sys

import numpy as np

from calibrated_explanations.cache.explanation_cache import ExplanationCacheFacade


def fresh_cache_module():
    return importlib.import_module("calibrated_explanations.cache.cache")


def boom_size_estimator(_value):
    raise RuntimeError("nope")


def test_cacheconfig_from_env_and_calibrator_cache(monkeypatch, tmp_path):
    cache_mod = fresh_cache_module()
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
    # Avoid flakiness from very short TTL during CI runs.
    cfg2.ttl_seconds = None
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
    cal.reset_version("newv")
    assert cal.version == "newv"

    # forksafe_reset should not error
    cal.forksafe_reset()

    # pickling roundtrip recreates lock
    sys.modules["calibrated_explanations.cache.cache"] = cache_mod
    state = pickle.dumps(cal)
    new = pickle.loads(state)
    assert hasattr(new, "__dict__")


def test_default_size_estimator_error_paths():
    cache_mod = fresh_cache_module()

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
    cache_mod = fresh_cache_module()
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

    sys.modules["calibrated_explanations.cache.cache"] = cache_mod
    state = pickle.dumps(cache)
    restored = pickle.loads(state)
    assert restored.get(("k",)) == {"v": 1}


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
    cache_mod = fresh_cache_module()
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


def test_explanation_cache_facade_enabled_compute_uses_cache_compute():
    cache_mod = fresh_cache_module()
    cfg = cache_mod.CacheConfig(enabled=True)
    cal = cache_mod.CalibratorCache(cfg)
    facade = ExplanationCacheFacade(cal)

    result = facade.compute_calibration_summaries(
        explainer_id="x",
        x_cal_hash="h",
        compute_fn=lambda: ({0: {"a": 1}}, {0: np.array([1.0])}),
    )
    assert result[0][0]["a"] == 1
