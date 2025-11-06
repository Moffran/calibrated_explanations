"""Tiny example plugin used by unit tests.

This module exports a single `PLUGIN` instance that conforms to the
`ExplainerPlugin` protocol used by the registry. It's intentionally
minimal and deterministic for testing.
"""

from __future__ import annotations

from typing import Any


class ExamplePlugin:
    """Deterministic plugin implementation for registry tests."""

    plugin_meta = {
        "schema_version": 1,
        "capabilities": ["explain"],
        "name": "tests.example_plugin",
        "version": "1.0-test",
        "provider": "tests",
        "trusted": False,
        "trust": False,
    }

    def supports(self, model: Any) -> bool:
        """Return True when *model* matches the supported test sentinel."""
        # Support a simple sentinel model value or a dict marker for tests
        return model == "supported-model" or (
            isinstance(model, dict) and model.get("kind") == "example"
        )

    def explain(self, model: Any, x: Any, **kwargs: Any) -> dict[str, Any]:
        """Produce a deterministic explanation payload for test assertions."""
        # Return a tiny deterministic payload for assertions
        return {"plugin": "example", "model": str(model), "n": (len(x) if x is not None else 0)}


PLUGIN = ExamplePlugin()
