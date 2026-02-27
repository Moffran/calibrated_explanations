"""Plugin trust-policy protocol and default implementation (ADR-006)."""

from __future__ import annotations

from typing import Any, Mapping, Protocol


class PluginTrustPolicy(Protocol):
    """Policy interface for deny/trust decisions."""

    def is_denied(self, identifier: str, *, denylist: set[str]) -> bool:
        """Return whether *identifier* is denied."""

    def is_trusted(
        self,
        *,
        meta: Mapping[str, Any],
        identifier: str,
        source: str,
        trusted_identifiers: set[str],
    ) -> bool:
        """Return whether plugin metadata should be trusted."""


class DefaultPluginTrustPolicy:
    """Default policy matching CE's existing trust model."""

    def is_denied(self, identifier: str, *, denylist: set[str]) -> bool:
        """Return whether *identifier* appears in *denylist*."""
        return identifier in denylist

    def is_trusted(
        self,
        *,
        meta: Mapping[str, Any],
        identifier: str,
        source: str,
        trusted_identifiers: set[str],
    ) -> bool:
        """Return trust decision for plugin metadata and source context."""
        # Builtins remain trusted by default.
        if source == "builtin":
            return True
        return identifier in trusted_identifiers


__all__ = ["PluginTrustPolicy", "DefaultPluginTrustPolicy"]
