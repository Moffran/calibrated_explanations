import inspect


def test_should_have_docstring_for_combined_plugin_when_style_registered():
    from calibrated_explanations.plugins import registry
    from calibrated_explanations.plugins.plots import CombinedPlotPlugin

    class DummyBuilder:
        plugin_meta = {
            "schema_version": 1,
            "name": "dummy.builder",
            "version": "1.0.0",
            "provider": "test",
            "style": "dummy-style",
            "capabilities": ("plot",),
            "output_formats": ("png",),
            "legacy_compatible": True,
            "dependencies": (),
        }

        def build(self, context):
            return {"dummy": "artifact"}

    class DummyRenderer:
        plugin_meta = {
            "schema_version": 1,
            "name": "dummy.renderer",
            "version": "1.0.0",
            "provider": "test",
            "capabilities": ("render",),
            "output_formats": ("png",),
            "supports_interactive": False,
            "dependencies": (),
        }

        def render(self, artifact, *, context):
            return None

    # Ensure a clean registry for the test
    registry.clear_plot_plugins()

    # Register builder, renderer and style
    registry.register_plot_builder("dummy.builder", DummyBuilder, metadata=DummyBuilder.plugin_meta)
    registry.register_plot_renderer(
        "dummy.renderer", DummyRenderer, metadata=DummyRenderer.plugin_meta
    )
    registry.register_plot_style(
        "dummy-style",
        metadata={
            "style": "dummy-style",
            "builder_id": "dummy.builder",
            "renderer_id": "dummy.renderer",
            "fallbacks": (),
            "legacy_compatible": True,
        },
    )

    plugin = registry.find_plot_plugin("dummy-style")
    assert plugin is not None
    # The registry should return an instance of the documented wrapper
    assert isinstance(plugin, CombinedPlotPlugin)
    # ADR-018: dynamically composed plugin classes must have non-empty docstrings
    doc = inspect.getdoc(plugin.__class__) or inspect.getdoc(plugin)
    assert (
        isinstance(doc, str) and doc.strip()
    ), "CombinedPlotPlugin must have a non-empty docstring"
