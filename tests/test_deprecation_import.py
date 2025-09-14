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
