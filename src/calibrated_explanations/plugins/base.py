"""Plugin base interfaces (ADR-006 skeleton).

Minimal interfaces to support a registry of third-party explainers. This is an
opt-in surface; users should understand that loading external plugins executes
arbitrary code. We will document risks and keep the registry explicit.

Contract (v0.1, unstable):
- Each plugin module exposes a ``plugin_meta`` dict with at least::

      {
          "schema_version": int,
          "capabilities": Sequence[str],
          "name": str,
          "version": str,
          "provider": str,
      }

  Optional integrity metadata can include a ``checksum`` entry (currently
  SHA256). Plugins must also surface a ``trusted`` boolean so the registry can
  enforce ADR-006's opt-in trust model. Older plugins that only expose the
  ``trust`` key continue to be normalised for backwards compatibility.
- Each plugin exposes two callables::

      supports(model) -> bool
      explain(model, X, **kwargs) -> Any  # typically an Explanation or legacy dict

This mirrors ADR-006 minimal capability metadata and keeps behaviour opt-in.

ADR-015 refines this layer with dedicated explanation, interval, and plotting
protocols. They build on the lightweight ``PluginMeta`` typing alias and the
validation helper exported from this module.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Protocol, Sequence

try:  # Python < 3.10 compatibility
    from typing import TypeAlias
except ImportError:  # pragma: no cover - fallback when TypeAlias is unavailable
    TypeAlias = object  # type: ignore[assignment]


PluginMeta: TypeAlias = Mapping[str, Any]


class ExplainerPlugin(Protocol):
    """Protocol for explainer plugins.

    Implementations are expected to provide:
    - plugin_meta: Dict[str, Any]
    - supports(model) -> bool
    - explain(model, X, **kwargs) -> Any
    """

    plugin_meta: PluginMeta

    def supports(self, model: Any) -> bool:  # pragma: no cover - protocol
        ...

    def explain(self, model: Any, X: Any, **kwargs: Any) -> Any:  # pragma: no cover - protocol
        ...


def _ensure_sequence_of_strings(value: Any, *, key: str) -> Sequence[str]:
    """Return *value* as a sequence of strings or raise ``ValueError``."""

    if isinstance(value, str) or not isinstance(value, Iterable):
        raise ValueError(f"plugin_meta[{key!r}] must be a sequence of strings")

    result: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"plugin_meta[{key!r}] must contain only string values")
        if not item:
            raise ValueError(f"plugin_meta[{key!r}] must contain non-empty string values")
        result.append(item)
    if not result:
        raise ValueError(f"plugin_meta[{key!r}] must not be empty")
    return tuple(result)


def validate_plugin_meta(meta: Dict[str, Any]) -> None:
    """Validate minimal plugin metadata required by ADR-006."""

    if not isinstance(meta, dict):
        raise ValueError("plugin_meta must be a dict")

    required_scalars = (
        ("schema_version", int),
        ("name", str),
        ("version", str),
        ("provider", str),
    )
    for key, typ in required_scalars:
        if key not in meta:
            raise ValueError(f"plugin_meta missing required key: {key}")
        value = meta[key]
        if not isinstance(value, typ) or (isinstance(value, str) and not value):
            raise ValueError(f"plugin_meta[{key!r}] must be a non-empty {typ.__name__}")

    capabilities = meta.get("capabilities")
    if capabilities is None:
        raise ValueError("plugin_meta missing required key: capabilities")
    meta["capabilities"] = _ensure_sequence_of_strings(capabilities, key="capabilities")

    checksum = meta.get("checksum")
    if checksum is not None and not isinstance(checksum, (str, Mapping)):
        raise ValueError("plugin_meta['checksum'] must be a string or mapping")

    if "trusted" in meta:
        trusted_value = meta["trusted"]
        if not isinstance(trusted_value, bool):
            raise ValueError("plugin_meta['trusted'] must be a boolean")
    elif "trust" in meta:
        # Backwards compatibility with earlier drafts that exposed ``trust``.
        trust_value = meta["trust"]
        if isinstance(trust_value, Mapping) and "trusted" in trust_value:
            meta["trusted"] = bool(trust_value["trusted"])
        else:
            meta["trusted"] = bool(trust_value)
    else:
        # Default to False for clarity; registry callers can still override.
        meta["trusted"] = False


__all__ = ["ExplainerPlugin", "validate_plugin_meta"]
