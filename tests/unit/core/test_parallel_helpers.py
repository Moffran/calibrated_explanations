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


def test_joblib_backend_uses_parallel_module(monkeypatch: pytest.MonkeyPatch):
    """Verify that a provided joblib module is invoked."""

    class FakeParallel:
        def __init__(self, *, n_jobs):
            self.n_jobs = n_jobs

        def __call__(self, iterator):
            return [task() for task in iterator]

    fake_joblib = SimpleNamespace(
        Parallel=lambda n_jobs: FakeParallel(n_jobs=n_jobs),
        delayed=lambda fn: (lambda value: (lambda: fn(value))),
    )
    monkeypatch.setitem(sys.modules, "joblib", fake_joblib)

    backend = perf_helpers.JoblibBackend()
    assert backend.map(lambda x: x + 1, [0, 1], workers=2) == [1, 2]


def test_sequential_map_returns_collection():
    assert perf_helpers.sequential_map(lambda x: x * x, range(3)) == [0, 1, 4]
