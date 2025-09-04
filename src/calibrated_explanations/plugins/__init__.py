"""Explainer plugin interfaces and registry (ADR-006, minimal skeleton).

This subpackage exposes:
- ``ExplainerPlugin`` protocol and ``validate_plugin_meta`` helper
- ``registry`` utilities: register, unregister, list_plugins, find_for

Security note: Registering/using third-party plugins executes arbitrary code.
Only use plugins you trust. This API is opt-in and intentionally explicit.
"""

from . import registry  # re-export module for convenience
from .base import ExplainerPlugin, validate_plugin_meta  # noqa: F401

__all__ = [
    "ExplainerPlugin",
    "validate_plugin_meta",
    "registry",
]
