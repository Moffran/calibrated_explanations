def test_broader_imports_and_attrs():
    # Import additional modules and access attributes to exercise more module-level code.
    import importlib

    modules = [
        "calibrated_explanations.plotting",
        "calibrated_explanations.plugins.cli",
        "calibrated_explanations.plugins.plots",
        "calibrated_explanations.plugins.registry",
        "calibrated_explanations.viz.builders",
        "calibrated_explanations.explanations.explanation",
        "calibrated_explanations.explanations.explanations",
        "calibrated_explanations.core.wrap_explainer",
        "calibrated_explanations.core.calibrated_explainer",
        "calibrated_explanations.core.explain._helpers",
        "calibrated_explanations.core.explain._shared",
        "calibrated_explanations.core.explain.orchestrator",
        "calibrated_explanations.core.prediction.orchestrator",
        "calibrated_explanations.plugins.manager",
        "calibrated_explanations.plugins.explanations",
    ]

    for mod in modules:
        m = importlib.import_module(mod)
        # Try to access some common attributes safely.
        for attr in ("__all__", "__version__", "__name__"):
            _ = getattr(m, attr, None)

    assert True
