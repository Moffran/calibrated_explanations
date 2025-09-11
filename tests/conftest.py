"""Pytest configuration for test suite.

Force a non-interactive matplotlib backend to avoid sporadic Tk/Tcl errors
(e.g., tkinter.TclError: Can't find a usable init.tcl) on local Windows runs.
CI already runs headless, but local environments may sometimes pick TkAgg.
"""

from __future__ import annotations

import os

# Set backend before any pyplot import happens.
os.environ.setdefault("MPLBACKEND", "Agg")

import pytest

import matplotlib

# Use non-interactive backend for tests to avoid rendering overhead
matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def disable_plot_show(monkeypatch):
    """Monkeypatch common plotting entrypoints to be no-ops for speed."""
    try:
        import matplotlib.pyplot as plt

        # Only disable interactive show to keep tests headless and fast.
        # Do NOT stub out Figure.savefig — some tests assert files are written.
        monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    except Exception:
        pass


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
    if not bool(os.getenv("FAST_TESTS")):
        return

    try:
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
    except Exception:
        # crepes not installed or other issue — ignore
        pass


def pytest_collection_modifyitems(config, items):
    """Skip tests marked slow when FAST_TESTS is enabled."""
    if not bool(os.getenv("FAST_TESTS")):
        return
    skip_marker = pytest.mark.skip(reason="skipped slow test in FAST_TESTS mode")
    for item in items:
        if item.get_closest_marker("slow") is not None:
            item.add_marker(skip_marker)


# (Optional future spot) Add global fixtures or marks.
