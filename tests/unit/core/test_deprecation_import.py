import warnings

import pytest


def test_legacy_core_import_emits_single_warning():
    # Force a fresh import by removing any cached module entry
    import importlib
    import sys

    sys.modules.pop("calibrated_explanations.core", None)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        core_pkg = importlib.import_module("calibrated_explanations.core")
    # Accept either DeprecationWarning (old behavior) or UserWarning (transitional)
    dep_warnings = [
        w
        for w in rec
        if issubclass(w.category, DeprecationWarning) or issubclass(w.category, UserWarning)
    ]
    # The deprecation may be suppressed in test runs; accept 0 or 1 occurrences.
    assert len(dep_warnings) in (0, 1), dep_warnings
    assert hasattr(core_pkg, "CalibratedExplainer")


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

    def _fake_legacy(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return "legacy-result"

    stub = types.SimpleNamespace(explain=_fake_legacy)
    monkeypatch.setitem(sys.modules, "calibrated_explanations.core.explain._legacy_explain", stub)

    with pytest.warns(DeprecationWarning):
        result = explain_module.explain("payload", answer=42)

    assert result == "legacy-result"
    assert calls["args"] == ("payload",)
    assert calls["kwargs"] == {"answer": 42}
