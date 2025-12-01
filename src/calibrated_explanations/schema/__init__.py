"""Schema validation entry points (ADR-001 Stage 5 API tightening).

Only the stable validation helper is exposed from the package root to keep the
surface area predictable for import graph linting.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import-time only
    from .validation import validate_payload


__all__ = ("validate_payload",)


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(name)

    module = import_module(f"{__name__}.validation")
    value = getattr(module, name)
    globals()[name] = value
    return value
