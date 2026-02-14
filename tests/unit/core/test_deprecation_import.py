import pytest


def test_should_resolve_core_lazy_exports_when_accessed() -> None:
    """Exercise calibrated_explanations.core.__getattr__ branches for coverage."""
    import importlib
    import types

    core = importlib.import_module("calibrated_explanations.core")

    # Arrange/Act: Access lazy exports.
    _ = core.CalibratedExplainer
    _ = core.WrapCalibratedExplainer
    _ = core.assign_threshold
    _ = core.CalibratedError
    _ = core.ValidationError
    _ = core.explain_exception
    explain_mod = core.explain

    # Assert: explain resolves to a module.
    assert isinstance(explain_mod, types.ModuleType)


def test_core_explain_explain_warns_and_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure calibrated_explanations.core.explain.explain proxies to legacy implementation."""
    import sys
    import types

    from calibrated_explanations.core import explain as explain_module

    calls = {}

    def fake_legacy_mock(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return "legacy-result"

    stub = types.SimpleNamespace(explain=fake_legacy_mock)
    monkeypatch.setitem(sys.modules, "calibrated_explanations.core.explain._legacy_explain", stub)

    with pytest.warns(DeprecationWarning):
        result = explain_module.explain("payload", answer=42)

    assert result == "legacy-result"
    assert calls["args"] == ("payload",)
    assert calls["kwargs"] == {"answer": 42}
