from __future__ import annotations

import importlib
import importlib.util
import runpy
import sys
import types
from pathlib import Path

import pytest

from calibrated_explanations.explanations.adapters import domain_to_legacy, legacy_to_domain
from calibrated_explanations.explanations.models import Explanation


def test_api_quick_explain_delegates_to_core(monkeypatch: pytest.MonkeyPatch) -> None:
    api_quick = importlib.import_module("calibrated_explanations.api.quick")
    captured = {}

    def fake_quick_explain(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    fake_module = types.SimpleNamespace(quick_explain=fake_quick_explain)
    monkeypatch.setattr(api_quick.importlib, "import_module", lambda _: fake_module)

    result = api_quick.quick_explain(
        model="m",
        x_train="x_train",
        y_train="y_train",
        x_cal="x_cal",
        y_cal="y_cal",
        x="x",
        task="classification",
        threshold=0.5,
        low_high_percentiles=(10, 90),
        preprocessor="pp",
    )

    assert result == {"ok": True}
    assert captured["model"] == "m"
    assert captured["task"] == "classification"
    assert captured["threshold"] == 0.5
    # Keep perf-shim warning tests deterministic by clearing cached shim modules.
    sys.modules.pop("calibrated_explanations.perf", None)
    sys.modules.pop("calibrated_explanations.perf.cache", None)
    sys.modules.pop("calibrated_explanations.perf.parallel", None)


def test_core_quick_explain_drives_fit_calibrate_and_explain(monkeypatch: pytest.MonkeyPatch) -> None:
    core_quick = importlib.import_module("calibrated_explanations.core.quick")
    calls = {"fit": None, "calibrate": None}

    class FakeWrapper:
        def fit(self, x_train, y_train):
            calls["fit"] = (x_train, y_train)

        def calibrate(self, x_cal, y_cal, **kwargs):
            calls["calibrate"] = (x_cal, y_cal, kwargs)

        def explain_factual(self, x):
            return {"x": x}

    monkeypatch.setattr(
        core_quick.ExplainerConfig,
        "__init__",
        lambda self, **kwargs: setattr(self, "_cfg", kwargs),
    )
    monkeypatch.setattr(
        core_quick.WrapCalibratedExplainer,
        "from_config",
        lambda cfg: FakeWrapper(),
    )

    result = core_quick.quick_explain(
        model="model",
        x_train=[1],
        y_train=[0],
        x_cal=[2],
        y_cal=[1],
        x=[3],
        task="regression",
    )

    assert result == {"x": [3]}
    assert calls["fit"] == ([1], [0])
    assert calls["calibrate"] == ([2], [1], {"mode": "regression"})


def test_explanations_adapters_roundtrip_shapes() -> None:
    payload = {
        "task": "classification",
        "prediction": {"predict": 0.9, "low": 0.8, "high": 1.0},
        "rules": {"rule": ["r1", "r2"], "feature": [0, [1, 2]]},
        "feature_weights": {"predict": [0.3, -0.2]},
        "feature_predict": {"predict": [0.9, 0.7]},
    }
    domain = legacy_to_domain(7, payload)
    assert domain.index == 7
    assert len(domain.rules) == 2

    out = domain_to_legacy(domain)
    assert out["task"] == "classification"
    assert out["rules"]["rule"] == ["r1", "r2"]
    assert out["feature_weights"]["predict"] == [0.3, -0.2]






def test_utils_module_reload_handles_joblib_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    utils_mod = importlib.import_module("calibrated_explanations.utils")
    real_import = __import__

    def guarded_import(name, *args, **kwargs):
        if name == "joblib._parallel_backends":
            raise ImportError("simulated missing joblib")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded_import)
    reloaded = importlib.reload(utils_mod)
    assert hasattr(reloaded, "set_rng_seed")


def test_utils_module_reload_returns_when_pool_property_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    utils_mod = importlib.import_module("calibrated_explanations.utils")

    class PoolMixin:
        pool = property(lambda self: None)

    fake_backends = types.SimpleNamespace(PoolManagerMixin=PoolMixin)
    monkeypatch.setitem(sys.modules, "joblib._parallel_backends", fake_backends)
    reloaded = importlib.reload(utils_mod)
    assert hasattr(reloaded, "perturb_dataset")


def test_schema_module_getattr_rejects_unknown_attribute() -> None:
    schema_mod = importlib.import_module("calibrated_explanations.schema")
    with pytest.raises(AttributeError):
        _ = schema_mod.not_exported_name
