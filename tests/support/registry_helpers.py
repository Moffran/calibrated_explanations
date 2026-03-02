"""Registry reset helpers used by tests."""

from calibrated_explanations.plugins import registry


def clear_explanation_plugins() -> None:
    """Reset explanation plugin descriptors for tests."""
    registry.reset_plugin_catalog(kind="explanation")


def clear_interval_plugins() -> None:
    """Reset interval plugin descriptors for tests."""
    registry.reset_plugin_catalog(kind="interval")


def clear_plot_plugins() -> None:
    """Reset plot plugin descriptors for tests."""
    registry.reset_plugin_catalog(kind="plot")
