def test_utils_mass_imports():
    # Import several utility modules to execute module-level initialization code.
    import importlib

    modules = [
        "calibrated_explanations.utils.helper",
        "calibrated_explanations.utils.int_utils",
        "calibrated_explanations.utils.discretizers",
        "calibrated_explanations.utils.rng",
        "calibrated_explanations.utils.exceptions",
        "calibrated_explanations.utils.deprecation",
    ]

    for mod in modules:
        importlib.import_module(mod)

    assert True
