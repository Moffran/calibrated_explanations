from __future__ import annotations

from time import monotonic
from typing import Dict

import numpy as np
import pytest

from calibrated_explanations.perf.cache import CalibratorCache, CacheConfig, LRUCache, make_key


def test_lru_cache_rejects_invalid_limits() -> None:
    with pytest.raises(ValueError):
        LRUCache(
            namespace="test",
            version="v1",
            max_items=0,
            max_bytes=1024,
            ttl_seconds=None,
            telemetry=None,
            size_estimator=lambda _: 1,
        )
    with pytest.raises(ValueError):
        LRUCache(
            namespace="test",
            version="v1",
            max_items=1,
            max_bytes=0,
            ttl_seconds=None,
            telemetry=None,
            size_estimator=lambda _: 1,
        )


def test_cache_respects_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = {"now": monotonic()}

    def fake_monotonic() -> float:
        return clock["now"]

    cache = LRUCache[
        str,
        int,
    ](
        namespace="test",
        version="v1",
        max_items=4,
        max_bytes=None,
        ttl_seconds=1.0,
        telemetry=None,
        size_estimator=lambda _: 1,
    )
    monkeypatch.setattr("calibrated_explanations.perf.cache.monotonic", fake_monotonic)

    cache.set("alpha", 42)
    assert cache.get("alpha") == 42
    clock["now"] += 2.0  # expire entry
    assert cache.get("alpha") is None
    assert cache.metrics.expirations == 1


def test_cache_respects_memory_budget() -> None:
    cache = LRUCache[
        str,
        np.ndarray,
    ](
        namespace="test",
        version="v1",
        max_items=10,
        max_bytes=16,
        ttl_seconds=None,
        telemetry=None,
        size_estimator=lambda value: int(getattr(value, "nbytes", 0)),
    )

    small = np.zeros(2, dtype=np.uint8)
    large = np.zeros(64, dtype=np.uint8)

    cache.set("small", small)
    assert cache.get("small") is small

    cache.set("too_big", large)
    assert cache.get("too_big") is None
    assert cache.metrics.misses >= 1


def test_calibrator_cache_namespaces() -> None:
    config = CacheConfig(enabled=True, namespace="calib", version="v2", max_items=8)
    cache: CalibratorCache[Dict[str, int]] = CalibratorCache(config)

    payload = {"value": 7}
    cache.set(stage="predict", parts=[("sample", 1)], value=payload)
    assert cache.get(stage="predict", parts=[("sample", 1)]) == payload
    assert cache.get(stage="train", parts=[("sample", 1)]) is None


def test_make_key_normalises_arrays() -> None:
    array = np.arange(5)
    key = make_key("ns", "v", [("payload", array)])
    assert key[0] == "ns"
    assert key[1] == "v"
    assert "payload" in str(key[-1])
