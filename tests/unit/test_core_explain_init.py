from __future__ import annotations

import importlib

import pytest

from calibrated_explanations.core import explain as core_explain


def test_explain_delegates_to_legacy_module(monkeypatch):
    sentinel = object()

    def fake_explain(*args, **kwargs):
        return (sentinel, args, kwargs)

    legacy_module = importlib.import_module("calibrated_explanations.core.explain._legacy_explain")
    monkeypatch.setattr(legacy_module, "explain", fake_explain)

    with pytest.warns(DeprecationWarning, match="deprecated"):
        result = core_explain.explain("subject", key="value")

    assert result[0] is sentinel
