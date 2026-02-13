"""Tests for lightweight parallel helper shims."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from calibrated_explanations.core import test as perf_helpers


def test_joblib_backend_falls_back_when_missing(monkeypatch: pytest.MonkeyPatch):
    """Ensure the backend handles ImportError gracefully."""
    original_import = __import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("joblib"):
            raise ImportError("forced")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    backend = perf_helpers.JoblibBackend()
    result = backend.map(lambda x: x * 2, [1, 2, 3], workers=1)
    assert result == [2, 4, 6]


def test_joblib_backend_uses_explicit_workers(monkeypatch: pytest.MonkeyPatch):
    calls: dict[str, object] = {}

    def delayed(fn):
        return lambda item: lambda: fn(item)

    class ParallelStub:
        def __init__(self, *, n_jobs):
            calls["n_jobs"] = n_jobs

        def __call__(self, tasks):
            return [task() for task in tasks]

    monkeypatch.setitem(sys.modules, "joblib", SimpleNamespace(Parallel=ParallelStub, delayed=delayed))
    backend = perf_helpers.JoblibBackend()

    result = backend.map(lambda x: x + 1, [1, 2, 3], workers=2)

    assert calls["n_jobs"] == 2
    assert result == [2, 3, 4]




def test_sequential_map_returns_collection():
    assert perf_helpers.sequential_map(lambda x: x * x, range(3)) == [0, 1, 4]
