from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Generic, Hashable, Iterable, Tuple, TypeVar

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


@dataclass
class LRUCache(Generic[K, V]):
    """A tiny LRU cache with max item count eviction.

    Feature-flag intended usage; safe to use standalone.
    """

    max_items: int = 128

    def __post_init__(self) -> None:
        if self.max_items <= 0:
            raise ValueError("max_items must be positive")
        self._store: OrderedDict[K, V] = OrderedDict()

    def get(self, key: K, default: V | None = None) -> V | None:
        if key in self._store:
            self._store.move_to_end(key, last=True)
            return self._store[key]
        return default

    def set(self, key: K, value: V) -> None:
        if key in self._store:
            self._store.move_to_end(key, last=True)
        self._store[key] = value
        if len(self._store) > self.max_items:
            self._store.popitem(last=False)

    def __contains__(self, key: K) -> bool:  # pragma: no cover - convenience
        return key in self._store

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._store)


def make_key(parts: Iterable[Hashable]) -> Tuple[Hashable, ...]:
    """Deterministic composite cache key from hashable parts.

    Consumers are responsible for pre-hashing large arrays/bytes to a hashable.
    """

    return tuple(parts)
