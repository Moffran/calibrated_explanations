"""Common test fixtures exported for the test suite.

This file exposes lightweight fixtures used across multiple test modules so
that standalone test files (for example parity tests) can import them via
pytest fixtures without duplicating large data-loading logic.

The fixtures intentionally delegate to existing helpers where available to
avoid duplicating parsing logic.
"""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path

import pytest
from ._fixtures import regression_dataset, binary_dataset, multiclass_dataset

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_PATH = _REPO_ROOT / "src"
if _SRC_PATH.exists():
    _SRC_STR = str(_SRC_PATH)
    if _SRC_STR not in sys.path:
        sys.path.insert(0, _SRC_STR)

# Ensure non-interactive backend is selected early so tests never require a GUI.
import os as _os

_os.environ.setdefault("MPLBACKEND", "Agg")

# Defer importing matplotlib until needed; some CI runs do not install viz extras.
_matplotlib = None

# Reference imported fixtures so static analyzers know they are used by pytest
_IMPORTED_FIXTURES = (regression_dataset, binary_dataset, multiclass_dataset)


def _env_flag(name: str) -> bool:
    """Return a boolean for environment flags treating common truthy strings as True.

    This treats '1', 'true', 'yes', 'y' (case-insensitive) as True. Everything
    else (including '0', 'false', empty string or unset) is False. Using this
    avoids Python's truthiness where a non-empty string like '0' evaluates to True.
    """
    v = os.getenv(name, "").strip().lower()
    return v in ("1", "true", "yes", "y")


@pytest.fixture(scope="session")
def sample_limit():
    """Return an integer sample limit for tests.

    Priority:
    - If SAMPLE_LIMIT env var is set, use it.
    - Else if FAST_TESTS is enabled, return a small limit (100).
    - Otherwise return default 500.
    """
    MINIMUM_LIMIT = 20
    val = os.getenv("SAMPLE_LIMIT")
    if val:
        try:
            v = int(val)
            # enforce a sensible lower bound to avoid dataset-splitting errors
            return v if v >= MINIMUM_LIMIT else MINIMUM_LIMIT
        except Exception:
            pass
    if _env_flag("FAST_TESTS"):
        return max(100, MINIMUM_LIMIT)
    return max(500, MINIMUM_LIMIT)


def _ensure_matplotlib():
    """Lazily import matplotlib and set a non-interactive backend.

    Returns the imported matplotlib module or None if not available.
    """
    global _matplotlib
    if _matplotlib is not None:
        return _matplotlib
    try:
        import matplotlib as _m

        # ensure non-interactive backend
        with contextlib.suppress(Exception):
            _m.use("Agg")
        _matplotlib = _m
    except Exception:
        _matplotlib = None
    return _matplotlib


@pytest.fixture(autouse=True)
def disable_plot_show(monkeypatch):
    """Monkeypatch common plotting entrypoints to be no-ops for speed."""
    mpl = _ensure_matplotlib()
    if not mpl:
        return
    # If pyplot can't be imported, continue silently for core-only runs.
    with contextlib.suppress(Exception):
        import matplotlib.pyplot as plt

        # Only disable interactive show to keep tests headless and fast.
        # Do NOT stub out Figure.savefig — some tests assert files are written.
        monkeypatch.setattr(plt, "show", lambda *a, **k: None)


@pytest.fixture(scope="session")
def small_random_forest():
    """Return a small, cheap RandomForestRegressor for reuse across tests."""
    from sklearn.ensemble import RandomForestRegressor
    import numpy as np

    # tiny deterministic dataset
    X = np.vstack([np.arange(8), np.arange(8) + 1]).T
    y = np.arange(8) / 8.0
    rf = RandomForestRegressor(n_estimators=3, random_state=42)
    rf.fit(X, y)
    return rf


@pytest.fixture(scope="session")
def small_random_forest_classifier():
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np

    X = np.vstack([np.arange(8), np.arange(8) + 1]).T
    y = np.arange(8) % 2
    rf = RandomForestClassifier(n_estimators=3, random_state=42)
    rf.fit(X, y)
    return rf


@pytest.fixture(scope="session")
def small_decision_tree():
    """Return a tiny DecisionTree classifier for reuse across tests."""
    from sklearn.tree import DecisionTreeClassifier
    import numpy as np

    X = np.vstack([np.arange(8), np.arange(8) + 1]).T
    y = np.arange(8) % 2
    dt = DecisionTreeClassifier(max_depth=2, random_state=0)
    dt.fit(X, y)
    return dt


@pytest.fixture(autouse=True)
def patch_difficulty_estimator(monkeypatch):
    """In FAST_TESTS mode, replace crepes DifficultyEstimator.fit with a lightweight stub to avoid KNN costs."""
    if not _env_flag("FAST_TESTS"):
        return

    with contextlib.suppress(Exception):
        from crepes import extras as _extras

        class _StubDifficulty:
            def fit(self, *a, **k):
                return self

            def predict(self, X):
                import numpy as _np

                return _np.zeros(len(X))

            def __call__(self, X):
                return self.predict(X)

        monkeypatch.setattr(_extras, "DifficultyEstimator", lambda *a, **k: _StubDifficulty())


def pytest_collection_modifyitems(config, items):
    """Skip tests marked slow when FAST_TESTS is enabled."""
    if not _env_flag("FAST_TESTS"):
        return
    skip_marker = pytest.mark.skip(reason="skipped slow test in FAST_TESTS mode")
    for item in items:
        if item.get_closest_marker("slow") is not None:
            item.add_marker(skip_marker)
