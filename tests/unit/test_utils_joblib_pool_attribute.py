import importlib
import sys
import types


def test_utils_import_adds_joblib_pool_attribute(monkeypatch):
    fake_backends = types.ModuleType("joblib._parallel_backends")

    class DummyPoolManagerMixin:
        pass

    fake_backends.PoolManagerMixin = DummyPoolManagerMixin

    joblib_pkg = types.ModuleType("joblib")
    joblib_pkg.__path__ = []

    monkeypatch.setitem(sys.modules, "joblib", joblib_pkg)
    monkeypatch.setitem(sys.modules, "joblib._parallel_backends", fake_backends)

    import calibrated_explanations.utils as utils

    importlib.reload(utils)

    assert hasattr(DummyPoolManagerMixin, "pool")
    instance = DummyPoolManagerMixin()
    assert instance.pool is None
    instance.pool = "ready"
    assert instance.pool == "ready"
