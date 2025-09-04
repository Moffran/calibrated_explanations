"""Performance foundations (feature-flagged): cache and parallel backends.

These utilities are scaffolded for Phase 3 (ADR-003/ADR-004) and are not wired
into the main flows by default. They can be imported and used explicitly, and
will be integrated behind feature flags in future steps.
"""

from .cache import LRUCache, make_key  # noqa: F401
from .parallel import JoblibBackend, ParallelBackend, sequential_map  # noqa: F401

__all__ = [
    "LRUCache",
    "make_key",
    "ParallelBackend",
    "JoblibBackend",
    "sequential_map",
]
