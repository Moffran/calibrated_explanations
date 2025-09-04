from __future__ import annotations

from typing import Callable, Iterable, List, Protocol, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")


class ParallelBackend(Protocol):
    def map(
        self, fn: Callable[[T], R], items: Sequence[T], *, workers: int | None = None
    ) -> List[R]: ...


class JoblibBackend:
    """Thin adapter over joblib. Falls back to sequential if joblib is missing."""

    def map(
        self, fn: Callable[[T], R], items: Sequence[T], *, workers: int | None = None
    ) -> List[R]:
        try:
            from joblib import Parallel, delayed  # type: ignore
        except Exception:
            return [fn(x) for x in items]
        n_jobs = workers if workers is not None else -1
        return Parallel(n_jobs=n_jobs)(delayed(fn)(x) for x in items)


def sequential_map(fn: Callable[[T], R], items: Iterable[T]) -> List[R]:
    return [fn(x) for x in items]
