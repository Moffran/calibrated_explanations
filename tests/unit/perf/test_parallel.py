from __future__ import annotations

import builtins
from types import SimpleNamespace

import pytest

from calibrated_explanations.perf import PerfFactory, from_config
from calibrated_explanations.perf import parallel as parallel_module


def test_perf_factory_cache_toggles_and_uses_max_items():
    disabled = PerfFactory(cache_enabled=False)
    assert disabled.make_cache() is None

    enabled = PerfFactory(cache_enabled=True, cache_max_items=3)
    cache = enabled.make_cache()
    assert cache is not None
    assert cache.max_items == 3
    cache.set("a", 1)
    cache.set("b", 2)
    assert cache.get("a") == 1


def test_perf_factory_parallel_disabled_returns_sequential_backend():
    factory = PerfFactory(parallel_enabled=False)
    backend = factory.make_parallel_backend()
    result = backend.map(lambda x: x + 1, [1, 2, 3])
    assert result == [2, 3, 4]


@pytest.mark.parametrize("backend_name", ["joblib", "auto", "unknown"])
def test_perf_factory_parallel_enables_joblib_backend(backend_name):
    factory = PerfFactory(parallel_enabled=True, parallel_backend=backend_name)
    backend = factory.make_parallel_backend()
    assert isinstance(backend, parallel_module.JoblibBackend)
    assert backend.map(lambda x: x * 2, [1, 2, 3]) == [2, 4, 6]


def test_joblib_backend_falls_back_when_import_fails(monkeypatch):
    backend = parallel_module.JoblibBackend()

    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("joblib"):
            raise ImportError("joblib not installed")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    values = backend.map(lambda x: x + 1, [10, 11])
    assert values == [11, 12]


def test_from_config_reads_perf_attributes():
    cfg = SimpleNamespace(
        perf_cache_enabled=True,
        perf_cache_max_items=5,
        perf_parallel_enabled=True,
        perf_parallel_backend="joblib",
    )
    factory = from_config(cfg)
    assert factory.cache_enabled is True
    assert factory.cache_max_items == 5
    assert factory.parallel_enabled is True
    assert factory.parallel_backend == "joblib"


def test_from_config_uses_defaults_when_missing():
    factory = from_config(SimpleNamespace())
    assert factory.cache_enabled is False
    assert factory.cache_max_items == 128
    assert factory.parallel_enabled is False
    assert factory.parallel_backend == "auto"
