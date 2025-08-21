import warnings


def test_legacy_core_import_emits_single_warning():
    # Force a fresh import by removing any cached module entry
    import importlib
    import sys

    sys.modules.pop("calibrated_explanations.core", None)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        core_pkg = importlib.import_module("calibrated_explanations.core")
    dep_warnings = [w for w in rec if issubclass(w.category, DeprecationWarning)]
    assert len(dep_warnings) == 1, dep_warnings
    assert hasattr(core_pkg, "CalibratedExplainer")
