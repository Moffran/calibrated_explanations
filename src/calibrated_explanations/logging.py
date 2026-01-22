"""Logging utilities for calibrated explanations.

This module provides logging configuration, context management, and telemetry
support for the calibrated explanations library.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import os
from typing import Any, Dict, Iterator

from .core.config_helpers import read_pyproject_section

# Context keys expected by ADR-028 / Standard-005
_CONTEXT_KEYS = (
    "request_id",
    "tenant_id",
    "explainer_id",
    "checkpoint_id",
    "plugin_identifier",
    "mode",
)

_context_vars = {key: contextvars.ContextVar(key, default=None) for key in _CONTEXT_KEYS}


def _coerce_bool(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on", "enable"}


def coerce_bool(value: str | bool | None) -> bool:
    """Public wrapper for boolean coercion used in telemetry config parsing."""
    return _coerce_bool(value)


def telemetry_diagnostic_mode() -> bool:
    """Return whether telemetry should emit full diagnostic payloads.

    Reads ``CE_TELEMETRY_DIAGNOSTIC_MODE`` or ``[tool.calibrated_explanations.telemetry]``
    ``diagnostic_mode`` from pyproject.toml. Env var takes precedence.
    """
    env_value = os.environ.get("CE_TELEMETRY_DIAGNOSTIC_MODE")
    if env_value is not None:
        return coerce_bool(env_value)

    config = read_pyproject_section(("tool", "calibrated_explanations", "telemetry"))
    return coerce_bool(config.get("diagnostic_mode")) if config else False


def get_logging_context() -> Dict[str, Any]:
    """Return current structured logging context."""
    return {key: var.get() for key, var in _context_vars.items() if var.get() is not None}


def update_logging_context(**kwargs: Any) -> None:
    """Update structured logging context fields present in kwargs."""
    for key, value in kwargs.items():
        if key in _context_vars:
            _context_vars[key].set(value)


@contextlib.contextmanager
def logging_context(**kwargs: Any) -> Iterator[None]:
    """Context manager to temporarily set logging context fields."""
    tokens = {}
    for key, value in kwargs.items():
        if key in _context_vars:
            tokens[key] = _context_vars[key].set(value)
    try:
        yield
    finally:
        for key, token in tokens.items():
            _context_vars[key].reset(token)


class LoggingContextFilter(logging.Filter):
    """Logging filter that injects structured context into records."""

    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - simple pass-through
        """Inject structured context into the log record."""
        context = get_logging_context()
        for key in _CONTEXT_KEYS:
            value = context.get(key)
            setattr(record, key, value)
        return True


def ensure_logging_context_filter(logger_name: str = "calibrated_explanations") -> None:
    """Attach the context filter to the root calibrated_explanations logger once."""
    logger = logging.getLogger(logger_name)
    # Avoid duplicate filters
    for existing in logger.filters:
        if isinstance(existing, LoggingContextFilter):
            return
    logger.addFilter(LoggingContextFilter())


__all__ = [
    "telemetry_diagnostic_mode",
    "get_logging_context",
    "update_logging_context",
    "logging_context",
    "ensure_logging_context_filter",
    "LoggingContextFilter",
    "coerce_bool",
]
