"""API decorators for surface classification (ADR-038)."""

from __future__ import annotations

import functools
from typing import Callable, TypeVar

_F = TypeVar("_F", bound=Callable)


def experimental(func: _F) -> _F:
    """Mark a function or method as experimental.

    Experimental surfaces may use ``**kwargs`` for parameter-contract flexibility
    while under active development (ADR-038 §3 exception). They MUST graduate to
    explicit typed arguments before leaving experimental status.

    Sets ``func.__experimental__ = True`` for programmatic introspection.
    Does not emit a runtime warning — the ``[EXPERIMENTAL]`` tag in the docstring
    is the caller-visible signal.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    wrapper.__experimental__ = True  # type: ignore[attr-defined]
    return wrapper  # type: ignore[return-value]
