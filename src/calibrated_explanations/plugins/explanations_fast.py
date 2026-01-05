"""In-tree FAST explanation plugin implementation (ADR-015)."""

from __future__ import annotations

from typing import Any

from .. import __version__ as package_version
from ..explanations.explanation import FastExplanation
from .registry import find_explanation_descriptor, register_explanation_plugin


def register_fast_explanation_plugin() -> None:
    """Register the in-tree FAST explanation plugin when missing."""
    if find_explanation_descriptor("core.explanation.fast") is not None:
        return

    from .builtins import _LegacyExplanationBase  # pylint: disable=import-outside-toplevel

    class BuiltinFastExplanationPlugin(_LegacyExplanationBase):
        """Legacy wrapper delegating fast explanations to the explainer."""

        plugin_meta: dict[str, Any] = {
            "name": "core.explanation.fast",
            "schema_version": 1,
            "version": package_version,
            "provider": "calibrated_explanations",
            "capabilities": [
                "explain",
                "explanation:fast",
                "task:classification",
                "task:regression",
            ],
            "modes": ("fast",),
            "tasks": ("classification", "regression"),
            "dependencies": ("core.interval.fast", "legacy"),
            "interval_dependency": "core.interval.fast",
            "plot_dependency": "legacy",
            "trusted": True,
            "trust": {"trusted": True},
        }

        def __init__(self) -> None:
            super().__init__(
                _mode="fast",
                _explanation_attr="explain_fast",
                _expected_cls=FastExplanation,
                plugin_meta=self.plugin_meta,
            )

    register_explanation_plugin(
        "core.explanation.fast",
        BuiltinFastExplanationPlugin(),
        source="builtin",
    )


__all__ = ["register_fast_explanation_plugin"]
