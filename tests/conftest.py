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

try:  # pragma: no cover - exercised indirectly when pytest-cov is absent
    import pytest_cov as _pytest_cov  # type: ignore[attr-defined]

    _HAS_PYTEST_COV = True
    _pytest_cov  # placate linters
    _coverage_module = None
except Exception:  # pragma: no cover - executed in minimalist environments
    _HAS_PYTEST_COV = False
    try:
        import coverage as _coverage_module  # type: ignore[import-not-found]
    except Exception:  # pragma: no cover - defensive
        _coverage_module = None

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
    minimum_limit = 20
    val = os.getenv("SAMPLE_LIMIT")
    if val:
        try:
            v = int(val)
            # enforce a sensible lower bound to avoid dataset-splitting errors
            return v if v >= minimum_limit else minimum_limit
        except Exception:
            pass
    if _env_flag("FAST_TESTS"):
        return max(100, minimum_limit)
    return max(500, minimum_limit)


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
    x = np.vstack([np.arange(8), np.arange(8) + 1]).T
    y = np.arange(8) / 8.0
    rf = RandomForestRegressor(n_estimators=3, random_state=42)
    rf.fit(x, y)
    return rf


@pytest.fixture(scope="session")
def small_random_forest_classifier():
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np

    x = np.vstack([np.arange(8), np.arange(8) + 1]).T
    y = np.arange(8) % 2
    rf = RandomForestClassifier(n_estimators=3, random_state=42)
    rf.fit(x, y)
    return rf


@pytest.fixture(scope="session")
def small_decision_tree():
    """Return a tiny DecisionTree classifier for reuse across tests."""
    from sklearn.tree import DecisionTreeClassifier
    import numpy as np

    x = np.vstack([np.arange(8), np.arange(8) + 1]).T
    y = np.arange(8) % 2
    dt = DecisionTreeClassifier(max_depth=2, random_state=0)
    dt.fit(x, y)
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

            def predict(self, x):
                import numpy as _np

                return _np.zeros(len(x))

            __call__ = predict

        monkeypatch.setattr(_extras, "DifficultyEstimator", lambda *a, **k: _StubDifficulty())


def pytest_addoption(parser):  # pragma: no cover - simple option registration
    """Provide coverage CLI flags when pytest-cov is unavailable."""

    if _HAS_PYTEST_COV:
        return

    group = parser.getgroup("coverage")
    group.addoption(
        "--cov",
        action="append",
        default=[],
        help="Measure coverage for the specified package or path.",
    )
    group.addoption(
        "--cov-report",
        action="append",
        default=[],
        help="Generate coverage reports (supports term, term-missing, html, xml).",
    )
    group.addoption(
        "--cov-fail-under",
        action="store",
        type=float,
        default=None,
        help="Fail if coverage percentage is below this value.",
    )


def pytest_configure(config):  # pragma: no cover - integration glue
    """Start coverage measurement when the real pytest-cov plugin is unavailable."""

    if _HAS_PYTEST_COV or _coverage_module is None:
        return

    targets = config.getoption("--cov")
    reports = config.getoption("--cov-report")
    fail_under = config.getoption("--cov-fail-under")

    if not targets and not reports and fail_under is None:
        return

    cov_kwargs = {"source": targets or None}
    coverage_controller = _coverage_module.Coverage(**cov_kwargs)
    coverage_controller.start()
    config._ce_cov_controller = coverage_controller  # type: ignore[attr-defined]
    config._ce_cov_reports = reports  # type: ignore[attr-defined]
    config._ce_cov_fail_under = fail_under  # type: ignore[attr-defined]


def pytest_unconfigure(config):  # pragma: no cover - integration glue
    """Emit coverage reports for the stub coverage integration."""

    coverage_controller = getattr(config, "_ce_cov_controller", None)
    if coverage_controller is None:
        return

    reports = getattr(config, "_ce_cov_reports", []) or []
    fail_under = getattr(config, "_ce_cov_fail_under", None)

    coverage_controller.stop()
    coverage_controller.save()

    report_kind = next((rep for rep in reports if rep in ("term", "term-missing")), None)
    show_missing = report_kind == "term-missing"
    coverage_controller.report(
        show_missing=show_missing,
        fail_under=fail_under,
    )

    for rep in reports:
        if rep.startswith("html"):
            _, _, directory = rep.partition(":")
            directory = directory or "htmlcov"
            coverage_controller.html_report(directory=directory)
        elif rep.startswith("xml"):
            _, _, outfile = rep.partition(":")
            outfile = outfile or "coverage.xml"
            coverage_controller.xml_report(outfile=outfile)


def pytest_collection_modifyitems(config, items):
    """Skip tests marked slow when FAST_TESTS is enabled."""
    if not _env_flag("FAST_TESTS"):
        return
    skip_marker = pytest.mark.skip(reason="skipped slow test in FAST_TESTS mode")
    for item in items:
        if item.get_closest_marker("slow") is not None:
            item.add_marker(skip_marker)


_MODULE_COVERAGE_THRESHOLDS = {
    "src/calibrated_explanations/_interval_regressor.py": 85.0,
    "src/calibrated_explanations/core/calibrated_explainer.py": 85.0,
    "src/calibrated_explanations/plugins/registry.py": 85.0,
    "src/calibrated_explanations/plugins/cli.py": 85.0,
    "src/calibrated_explanations/api/config.py": 85.0,
    "src/calibrated_explanations/utils/helper.py": 85.0,
    "src/calibrated_explanations/utils/perturbation.py": 85.0,
    "src/calibrated_explanations/_plots.py": 85.0,
    "src/calibrated_explanations/_plots_legacy.py": 85.0,
    "src/calibrated_explanations/viz/matplotlib_adapter.py": 85.0,
    "src/calibrated_explanations/explanations/explanation.py": 85.0,
    "src/calibrated_explanations/perf/cache.py": 85.0,
    "src/calibrated_explanations/perf/parallel.py": 85.0,
    "src/calibrated_explanations/perf/__init__.py": 85.0,
}


def _get_active_coverage_controller(config):
    plugin = config.pluginmanager.get_plugin("_cov")
    if plugin is not None:
        controller = getattr(plugin, "cov_controller", None)
        if controller is not None:
            return controller.cov
    controller = getattr(config, "_ce_cov_controller", None)
    if controller is not None:
        return controller
    return None


def pytest_sessionfinish(session, exitstatus):  # pragma: no cover - exercised indirectly
    cov = _get_active_coverage_controller(session.config)
    if cov is None:
        return

    try:
        data = cov.get_data()
    except Exception:  # pragma: no cover - defensive
        return
    if data is None:
        return

    root = Path(session.config.rootpath).resolve()
    measured = {Path(filename).resolve() for filename in data.measured_files()}
    failures: list[str] = []

    for relative, threshold in _MODULE_COVERAGE_THRESHOLDS.items():
        target = (root / relative).resolve()
        if target not in measured:
            continue
        try:
            analysis = cov._analyze(str(target))  # pylint: disable=protected-access
            percent = analysis.numbers.pc_covered
        except Exception:  # pragma: no cover - fallback for coverage API changes
            try:
                _, statements, _, missing, _ = cov.analysis2(str(target))
            except Exception:
                continue
            if not statements:
                percent = 100.0
            else:
                percent = 100.0 * (len(statements) - len(missing)) / len(statements)
        if percent + 1e-6 < threshold:
            failures.append(
                f"Coverage for {relative} is {percent:.1f}% (required {threshold:.1f}%)"
            )

    if failures:
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        for message in failures:
            if reporter:
                reporter.write_line(message, red=True)  # pragma: no cover - side effect
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
