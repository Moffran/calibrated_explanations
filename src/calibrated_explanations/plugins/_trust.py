"""Atomic trust-state mutation helpers for plugin registry internals."""

from __future__ import annotations

import logging
import os
from threading import RLock, local
from typing import Any, Callable

from ..logging import ensure_logging_context_filter, logging_context
from ..utils.exceptions import ValidationError

_TRUST_LOCK = RLock()
_TRUST_CONTEXT = local()
_LOGGER = logging.getLogger("calibrated_explanations.governance.registry")
ensure_logging_context_filter("calibrated_explanations.governance.registry")


def trust_debug_checks_enabled() -> bool:
    """Return ``True`` when debug invariant checks should be enforced."""
    value = os.getenv("CE_DEBUG_TRUST_INVARIANTS", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def mutate_trust_atomic(
    identifier: str,
    trusted: bool,
    *,
    actor: str | None = None,
    kind: str = "unknown",
    source: str = "registry",
    mutation: Callable[[], Any],
    verify: Callable[[], None] | None = None,
) -> Any:
    """Run a trust mutation under a shared lock and emit a governance payload.

    Parameters
    ----------
    identifier:
        Plugin identifier being mutated.
    trusted:
        Target trust state.
    actor:
        Optional actor string (for example ``"register_interval_plugin"``).
    kind:
        Plugin kind associated with the mutation.
    source:
        Source string for observability payloads.
    mutation:
        Closure that performs the actual write operations.
    verify:
        Optional invariant check callback executed before releasing the lock.
    """
    with _TRUST_LOCK:
        _TRUST_CONTEXT.depth = getattr(_TRUST_CONTEXT, "depth", 0) + 1
        try:
            result = mutation()
            if verify is not None:
                verify()
            with logging_context(plugin_identifier=identifier):
                _LOGGER.info(
                    "Plugin trust state mutated",
                    extra={
                        "event_name": "trust.mutation",
                        "identifier": identifier,
                        "trusted": bool(trusted),
                        "kind": kind,
                        "source": source,
                        "actor": actor or "unknown",
                    },
                )
            return result
        finally:
            _TRUST_CONTEXT.depth = max(getattr(_TRUST_CONTEXT, "depth", 1) - 1, 0)


def _assert_mutation_context() -> None:
    depth = getattr(_TRUST_CONTEXT, "depth", 0)
    if depth <= 0:
        raise ValidationError(
            "Trust-state mutation helper called outside mutate_trust_atomic context.",
            details={"helper": "_assert_mutation_context"},
        )


def update_trusted_identifier(
    trusted_identifiers: set[str], identifier: str, trusted: bool
) -> None:
    """Update trusted identifier membership for *identifier*."""
    _assert_mutation_context()
    if trusted:
        trusted_identifiers.add(identifier)
    else:
        trusted_identifiers.discard(identifier)


def clear_trusted_identifiers(trusted_identifiers: set[str]) -> None:
    """Clear trusted identifier membership in-place."""
    _assert_mutation_context()
    trusted_identifiers.clear()
