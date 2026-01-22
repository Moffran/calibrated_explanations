from __future__ import annotations

import importlib
import types

import pytest

from calibrated_explanations import utils as utils_module
from calibrated_explanations.api import params, quick
from calibrated_explanations.utils.exceptions import ConfigurationError


def test_canonicalize_kwargs_maps_aliases_without_overwriting():
    payload = {"alpha": (10, 90), "parallel_workers": 4}
    canonical = params.canonicalize_kwargs(payload)
    assert canonical["low_high_percentiles"] == payload["alpha"]
    assert canonical["alpha"] == payload["alpha"]
    assert canonical["parallel_workers"] == 4


def test_validate_param_combination_reports_conflicts():
    with pytest.raises(ConfigurationError, match="mutually exclusive"):
        params.validate_param_combination({"threshold": 0.1, "confidence_level": 0.5})


def test_warn_on_aliases_emits_user_warning(monkeypatch):
    recorded = []

    def fake_deprecate(alias, canonical, *, stacklevel):
        recorded.append((alias, canonical, stacklevel))

    monkeypatch.setattr(utils_module, "deprecate_alias", fake_deprecate)
    with pytest.warns(UserWarning, match="alpha"):
        params.warn_on_aliases({"alpha": 0.2})
    assert recorded


def test_quick_explain_forwards_to_core(monkeypatch):
    sentinel = object()

    stub_module = types.SimpleNamespace(
        quick_explain=lambda **kwargs: (sentinel, kwargs),
    )

    def fake_import(name):
        assert name == "calibrated_explanations.core.quick"
        return stub_module

    monkeypatch.setattr(importlib, "import_module", fake_import)

    value, kwargs = quick.quick_explain(
        model="m",
        x_train=[1],
        y_train=[0],
        x_cal=[1],
        y_cal=[0],
        x=[2],
    )

    assert value is sentinel
    assert kwargs["model"] == "m"
