"""Tiny example plugin used by unit tests.

This module exports a single `PLUGIN` instance that conforms to the
`ExplainerPlugin` protocol used by the registry. It's intentionally
minimal and deterministic for testing.
"""

from __future__ import annotations

from typing import Any


class ExamplePlugin:
    plugin_meta = {"schema_version": 1, "capabilities": ["explain"], "name": "tests.example_plugin"}

    def supports(self, model: Any) -> bool:
        # Support a simple sentinel model value or a dict marker for tests
        return model == "supported-model" or (
            isinstance(model, dict) and model.get("kind") == "example"
        )

    def explain(self, model: Any, X: Any, **kwargs: Any) -> dict[str, Any]:
        # Return a tiny deterministic payload for assertions
        return {"plugin": "example", "model": str(model), "n": (len(X) if X is not None else 0)}


PLUGIN = ExamplePlugin()
