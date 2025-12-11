"""Lightweight parallel execution helpers used by optional perf scaffolding."""

from __future__ import annotations

import sys
from typing import Callable, Iterable, List, Protocol, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")


class ParallelBackend(Protocol):
    """Protocol describing a map-style parallel backend."""

    def map(
        self, fn: Callable[[T], R], items: Sequence[T], *, workers: int | None = None
    ) -> List[R]:
        """Apply *fn* to *items* using the backend's execution strategy."""
        ...


class JoblibBackend:
    """Thin adapter over joblib. Falls back to sequential if joblib is missing."""

    def map(
        self, fn: Callable[[T], R], items: Sequence[T], *, workers: int | None = None
    ) -> List[R]:
        """Execute *fn* over *items* using joblib when available."""
        try:
            from joblib import Parallel, delayed  # type: ignore
        except:
            if not isinstance(sys.exc_info()[1], Exception):
                raise
            return [fn(x) for x in items]
        n_jobs = workers if workers is not None else -1
        return Parallel(n_jobs=n_jobs)(delayed(fn)(x) for x in items)


def sequential_map(fn: Callable[[T], R], items: Iterable[T]) -> List[R]:
    """Apply *fn* to *items* sequentially and return the collected results."""
    return [fn(x) for x in items]
