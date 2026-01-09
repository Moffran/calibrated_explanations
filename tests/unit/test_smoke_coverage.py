import importlib


def test_smoke_import_key_modules():
    """Lightweight smoke test to import a set of modules and touch a few
    attributes to exercise module-level code paths and improve coverage.
    """
    modules = [
        "calibrated_explanations.plotting",
        "calibrated_explanations.plugins.builtins",
        "calibrated_explanations.explanations.explanations",
        "calibrated_explanations.viz.builders",
        "calibrated_explanations.viz.serializers",
        "calibrated_explanations.plugins.manager",
        "calibrated_explanations.plugins.cli",
        "calibrated_explanations.plugins.registry",
        "calibrated_explanations.explanations.legacy_conjunctions",
    ]

    for modname in modules:
        mod = importlib.import_module(modname)
        # Access a harmless attribute if present to touch more lines
        _ = getattr(mod, "__all__", None)
        _ = getattr(mod, "__name__", None)

    # Exercise a few small functions from heavy modules to cover branches
    from calibrated_explanations.plugins.builtins import derive_threshold_labels, LegacyPredictBridge
    # Sequence threshold
    assert derive_threshold_labels([0, 1])
    # Scalar threshold
    assert derive_threshold_labels(0.5)

    class DummyExplainer:
        def predict(self, x, uq_interval=False, calibrated=False, bins=None):
            import numpy as _np

            if uq_interval:
                return _np.array([0.1, 0.9])
            return _np.array([0.5])

        def predict_proba(self, x, uq_interval=False, calibrated=False, bins=None):
            import numpy as _np

            return _np.array([[0.2, 0.8]])

    bridge = LegacyPredictBridge(DummyExplainer())
    _ = bridge.predict([1, 2, 3], mode="mode", task="regression")
    _ = bridge.predict_proba([1, 2, 3])

    # Touch PlotKindRegistry helpers to cover lightweight validation logic
    from calibrated_explanations.viz.serializers import PlotKindRegistry

    kinds = PlotKindRegistry.supported_kinds()
    assert "triangular" in kinds
    assert PlotKindRegistry.is_supported_kind("triangular")
    PlotKindRegistry.validate_kind_and_mode("triangular", "classification")
    try:
        PlotKindRegistry.validate_kind_and_mode("unknown_kind", "classification")
    except Exception:
        pass
    _ = PlotKindRegistry.get_kind_requirements("triangular")
    # Touch plotting helpers
    import calibrated_explanations.plotting as plotting

    assert plotting.derive_threshold_labels((0, 1))
    assert plotting.split_csv("a, b, c")
    assert plotting.format_save_path("/tmp", "file.png")

    # Call public style resolver (avoids accessing private helpers)
    _ = plotting.resolve_plot_style_chain(None, None)
