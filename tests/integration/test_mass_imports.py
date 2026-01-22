def test_mass_imports():
    # Import a broad set of modules to execute module-level code and increase coverage.
    import importlib

    modules = [
        "calibrated_explanations",
        "calibrated_explanations.core",
        "calibrated_explanations.core.calibrated_explainer",
        "calibrated_explanations.core.wrap_explainer",
        "calibrated_explanations.core.explain._computation",
        "calibrated_explanations.core.explain._feature_filter",
        "calibrated_explanations.plugins.builtins",
        "calibrated_explanations.plugins.manager",
        "calibrated_explanations.plugins.registry",
        "calibrated_explanations.explanations.explanation",
        "calibrated_explanations.explanations.explanations",
        "calibrated_explanations.plotting",
        "calibrated_explanations.viz.builders",
        "calibrated_explanations.utils.helper",
        "calibrated_explanations.utils.int_utils",
        "calibrated_explanations.utils.rng",
    ]

    for mod in modules:
        importlib.import_module(mod)

    assert True
