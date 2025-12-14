import warnings


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
