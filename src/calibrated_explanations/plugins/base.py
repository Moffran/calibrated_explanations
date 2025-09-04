"""Plugin base interfaces (ADR-006 skeleton).

Minimal interfaces to support a registry of third-party explainers. This is an
opt-in surface; users should understand that loading external plugins executes
arbitrary code. We will document risks and keep the registry explicit.

Contract (v0.1, unstable):
- Each plugin module exposes a ``plugin_meta`` dict with at least:
    {"schema_version": 1, "capabilities": ["explain"], "name": str}
- Each plugin exposes two callables:
    supports(model) -> bool
    explain(model, X, **kwargs) -> Any  # typically an Explanation or legacy dict

This mirrors ADR-006 minimal capability metadata and keeps behavior opt-in.
"""

from __future__ import annotations

from typing import Any, Dict, Protocol


class ExplainerPlugin(Protocol):
    """Protocol for explainer plugins.

    Implementations are expected to provide:
    - plugin_meta: Dict[str, Any]
    - supports(model) -> bool
    - explain(model, X, **kwargs) -> Any
    """

    plugin_meta: Dict[str, Any]

    def supports(self, model: Any) -> bool:  # pragma: no cover - protocol
        ...

    def explain(self, model: Any, X: Any, **kwargs: Any) -> Any:  # pragma: no cover - protocol
        ...


def validate_plugin_meta(meta: Dict[str, Any]) -> None:
    """Validate minimal plugin metadata.

    Required keys: schema_version (int), capabilities (list[str]), name (str)
    """

    if not isinstance(meta, dict):
        raise ValueError("plugin_meta must be a dict")
    for key, typ in ("schema_version", int), ("capabilities", list), ("name", str):
        if key not in meta:
            raise ValueError(f"plugin_meta missing required key: {key}")
        if not isinstance(meta[key], typ):
            raise ValueError(f"plugin_meta[{key!r}] must be {typ.__name__}")


__all__ = ["ExplainerPlugin", "validate_plugin_meta"]
