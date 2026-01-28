def test_import_package_does_not_eagerly_import_matplotlib():
    import sys
    import importlib

    # Remove any existing matplotlib entries to simulate a clean import
    for key in list(sys.modules.keys()):
        if key.startswith("matplotlib"):
            del sys.modules[key]

    importlib.invalidate_caches()
    import calibrated_explanations as ce  # noqa: F401

    # After importing the package, matplotlib should not have been imported
    assert not any(k.startswith("matplotlib") for k in sys.modules)
