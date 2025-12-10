"""Plugin helpers used by tests."""

from calibrated_explanations.plugins import (
    unregister,
    clear_explanation_plugins,
    ensure_builtin_plugins,
)
from calibrated_explanations.plugins.plots import PlotRenderContext


def cleanup_plugin(plugin) -> None:
    """Reset the plugin registry after a test modifies it."""
    unregister(plugin)
    clear_explanation_plugins()
    ensure_builtin_plugins()


def make_plot_context(**overrides) -> PlotRenderContext:
    """Construct a PlotRenderContext populated with test-friendly defaults."""
    context = {
        "explanation": None,
        "instance_metadata": {},
        "style": "plot_spec.default",
        "intent": {},
        "show": False,
        "path": None,
        "save_ext": None,
        "options": {},
    }
    context.update(overrides)
    return PlotRenderContext(**context)
