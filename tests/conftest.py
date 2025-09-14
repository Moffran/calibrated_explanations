"""Common test fixtures exported for the test suite.

This file exposes lightweight fixtures used across multiple test modules so
that standalone test files (for example parity tests) can import them via
pytest fixtures without duplicating large data-loading logic.

The fixtures intentionally delegate to existing helpers where available to
avoid duplicating parsing logic.
"""

import os

import pytest

import matplotlib
from ._fixtures import regression_dataset, binary_dataset, multiclass_dataset

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


# Set a non-interactive backend before any matplotlib.pyplot import happens.
os.environ.setdefault("MPLBACKEND", "Agg")

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
    if not _env_flag("FAST_TESTS"):
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
    if not _env_flag("FAST_TESTS"):
        return
    skip_marker = pytest.mark.skip(reason="skipped slow test in FAST_TESTS mode")
    for item in items:
        if item.get_closest_marker("slow") is not None:
            item.add_marker(skip_marker)
