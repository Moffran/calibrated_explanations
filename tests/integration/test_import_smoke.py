import importlib


def test_import_smoke_modules():
    """Smoke test: import a set of public/impl modules to exercise module-level code.

    This is intentionally small and deterministic — it only imports modules and
    asserts they load. It helps increase coverage for lazy-loaded code paths
    without exercising heavy plotting backends.
    """
    modules = [
        "calibrated_explanations.core.calibrated_explainer",
        "calibrated_explanations.core.explain._computation",
        "calibrated_explanations.core.explain._feature_filter",
        "calibrated_explanations.plugins.builtins",
        "calibrated_explanations.explanations.explanation",
        "calibrated_explanations.viz.builders",
        "calibrated_explanations.plugins.registry",
    ]

    for name in modules:
        mod = importlib.import_module(name)
        assert mod is not None
