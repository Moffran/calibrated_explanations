from __future__ import annotations

from calibrated_explanations.plugins.plots import CombinedPlotPlugin


class DummyPlotBuilder:
    plugin_meta = {"builder": True}

    def build(self, *args, **kwargs):
        return "artifact"


class DummyPlotRenderer:
    plugin_meta = {"renderer": True}

    def render(self, artifact, *, context=None):
        return f"rendered {artifact} {context}"


def test_combined_plot_plugin_delegates():
    builder = DummyPlotBuilder()
    renderer = DummyPlotRenderer()
    plugin = CombinedPlotPlugin(builder, renderer)

    assert plugin.plugin_meta == builder.plugin_meta
    assert plugin.build("context") == "artifact"
    assert plugin.render("artifact", context="ctx") == "rendered artifact ctx"
