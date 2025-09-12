"""Common test fixtures exported for the test suite.

This file exposes lightweight fixtures used across multiple test modules so
that standalone test files (for example parity tests) can import them via
pytest fixtures without duplicating large data-loading logic.

The fixtures intentionally delegate to existing helpers where available to
avoid duplicating parsing logic.
"""

import os

from tests.test_explanation_parity import _make_binary_dataset
import pytest

import matplotlib


def _env_flag(name: str) -> bool:
    """Return a boolean for environment flags treating common truthy strings as True.

    This treats '1', 'true', 'yes', 'y' (case-insensitive) as True. Everything
    else (including '0', 'false', empty string or unset) is False. Using this
    avoids Python's truthiness where a non-empty string like '0' evaluates to True.
    """
    v = os.getenv(name, "").strip().lower()
    return v in ("1", "true", "yes", "y")


# Set a non-interactive backend before any matplotlib.pyplot import happens.
os.environ.setdefault("MPLBACKEND", "Agg")

# Use non-interactive backend for tests to avoid rendering overhead
matplotlib.use("Agg")


@pytest.fixture
def binary_dataset():
    """Lightweight binary dataset fixture used by parity tests.

    Delegates to the internal helper in `tests/test_explanation_parity.py` to
    keep parity tests fast and isolated.
    """
    return _make_binary_dataset()


@pytest.fixture
def regression_dataset():
    """Generate a regression dataset used by regression parity tests.

    This reproduces the minimal behavior of the original `regression_dataset`
    fixture but as a plain function (not delegating to another pytest
    fixture), avoiding calling fixtures directly.
    """
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split

    num_to_test = 2
    dataset = "abalone.txt"

    ds = pd.read_csv(f"data/reg/{dataset}")
    fast = _env_flag("FAST_TESTS")
    max_rows = 500 if fast else 2000
    X = ds.drop("REGRESSION", axis=1).values[:max_rows, :]
    y = ds["REGRESSION"].values[:max_rows]
    # calibration_size must be smaller than the available training rows
    calibration_size = min(1000, max(2, max_rows - num_to_test - 2))
    y = (y - np.min(y)) / (np.max(y) - np.min(y))
    no_of_features = X.shape[1]
    categorical_features = [i for i in range(no_of_features) if len(np.unique(X[:, i])) < 10]
    columns = ds.drop("REGRESSION", axis=1).columns

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=num_to_test, random_state=42
    )
    X_prop_train, X_cal, y_prop_train, y_cal = train_test_split(
        X_train, y_train, test_size=calibration_size, random_state=42
    )
    return (
        X_prop_train,
        y_prop_train,
        X_cal,
        y_cal,
        X_test,
        y_test,
        no_of_features,
        categorical_features,
        columns,
    )


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
