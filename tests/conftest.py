"""Common test fixtures exported for the test suite.

This file exposes lightweight fixtures used across multiple test modules so
that standalone test files (for example parity tests) can import them via
pytest fixtures without duplicating large data-loading logic.

The fixtures intentionally delegate to existing helpers where available to
avoid duplicating parsing logic.
"""

from __future__ import annotations

# CRITICAL: Preload matplotlib submodules BEFORE pytest-cov instruments code.
# This is placed immediately after __future__ imports (which must be first).
# matplotlib 3.8+ uses lazy loading that breaks when coverage instruments __getattr__.
try:
    import matplotlib
    import matplotlib.image  # noqa: F401
    import matplotlib.axes  # noqa: F401
    import matplotlib.artist  # noqa: F401
    import matplotlib.pyplot  # noqa: F401 - Force full pyplot initialization
except Exception:  # pragma: no cover
    pass  # matplotlib not installed

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


if not _HAS_PYTEST_COV:

    def pytest_addoption(parser):  # pragma: no cover - exercised when pytest-cov absent
        """Register stub coverage options so pytest.ini remains compatible."""

        group = parser.getgroup("cov", "coverage reporting")
        group.addoption(
            "--cov",
            action="append",
            default=[],
            metavar="path",
            help="stub option accepted when pytest-cov is unavailable",
        )
        group.addoption(
            "--cov-report",
            action="append",
            default=[],
            metavar="type",
            help="stub option accepted when pytest-cov is unavailable",
        )
        group.addoption(
            "--cov-config",
            action="store",
            default=None,
            metavar="path",
            help="stub option accepted when pytest-cov is unavailable",
        )
        group.addoption(
            "--cov-fail-under",
            action="store",
            default=None,
            metavar="float",
            help="stub option accepted when pytest-cov is unavailable",
        )


_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_PATH = _REPO_ROOT / "src"
if _SRC_PATH.exists():
    _SRC_STR = str(_SRC_PATH)
    if _SRC_STR not in sys.path:
        sys.path.insert(0, _SRC_STR)

# Ensure non-interactive backend is selected early so tests never require a GUI.
os.environ.setdefault("MPLBACKEND", "Agg")

# Defer importing matplotlib until needed; some CI runs do not install viz extras.
_matplotlib = None


def _matplotlib_available() -> bool:
    """Return True when matplotlib can be imported."""

    return _ensure_matplotlib() is not None


def _should_relax_coverage(config) -> bool:
    option = getattr(config, "option", None)
    keyword = getattr(option, "keyword", "") if option is not None else ""
    if keyword and "viz" in keyword:
        return True
    return not _matplotlib_available()


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


def _should_enforce_cov_threshold(config) -> bool:
    """Return True when coverage fail-under thresholds should be enforced."""

    try:
        args = tuple(config.args)
    except AttributeError:
        return True

    if not args:
        return True

    root = Path(getattr(config, "rootpath", Path.cwd())).resolve()
    tests_root = (root / "tests").resolve()

    for raw in args:
        text = str(raw)
        if not text:
            continue
        path_text, _, _ = text.partition("::")
        try:
            candidate = Path(path_text)
        except TypeError:
            continue
        if not candidate.is_absolute():
            candidate = (root / candidate).resolve()
        else:
            candidate = candidate.resolve()
        if candidate == tests_root:
            return True

    return False


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
        # Preload lazy-loaded submodules to avoid AttributeError in coverage context
        with contextlib.suppress(Exception):
            import matplotlib.image
            import matplotlib.axes
            import matplotlib.artist
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
    group.addoption(
        "--cov-config",
        action="store",
        default=None,
        help="Read coverage configuration from the given file.",
    )


def pytest_configure(config):  # pragma: no cover - integration glue
    """Start coverage measurement when the real pytest-cov plugin is unavailable."""

    enforce_fail_under = _should_enforce_cov_threshold(config)
    setattr(config, "_ce_cov_enforce_thresholds", enforce_fail_under)

    if _HAS_PYTEST_COV:
        if not enforce_fail_under:
            with contextlib.suppress(AttributeError):
                config.option.cov_fail_under = 0.0
            plugin = config.pluginmanager.get_plugin("_cov")
            if plugin is not None:
                for attr in ("options", "cov_controller"):
                    target = getattr(plugin, attr, None)
                    if target is not None:
                        with contextlib.suppress(AttributeError):
                            target.options.cov_fail_under = 0.0
        return

    if _coverage_module is None:
        return

    targets = config.getoption("--cov")
    reports = config.getoption("--cov-report")
    fail_under = config.getoption("--cov-fail-under")
    cov_config = config.getoption("--cov-config")

    if not enforce_fail_under:
        fail_under = None

    if not targets and not reports and fail_under is None and cov_config is None:
        return

    if fail_under is not None and _should_relax_coverage(config):
        fail_under = min(fail_under, _MATPLOTLIB_OPTIONAL_FAIL_UNDER)
        with contextlib.suppress(Exception):
            config.option.cov_fail_under = fail_under

    cov_kwargs = {"source": targets or None}
    if cov_config:
        cov_kwargs["config_file"] = cov_config
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
    if fail_under is not None and _should_relax_coverage(config):
        fail_under = min(fail_under, _MATPLOTLIB_OPTIONAL_FAIL_UNDER)

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


_MATPLOTLIB_OPTIONAL_FAIL_UNDER = 60.0

_MATPLOTLIB_OPTIONAL_MODULES = {
    "src/calibrated_explanations/_plots.py",
    "src/calibrated_explanations/_plots_legacy.py",
    "src/calibrated_explanations/viz/matplotlib_adapter.py",
}


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
    if not getattr(session.config, "_ce_cov_enforce_thresholds", True):
        return

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

    skip_matplotlib_modules = _should_relax_coverage(session.config)

    for relative, threshold in _MODULE_COVERAGE_THRESHOLDS.items():
        if skip_matplotlib_modules and relative in _MATPLOTLIB_OPTIONAL_MODULES:
            continue
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


def pytest_sessionstart(session):  # pragma: no cover - simple configuration tweak
    option = getattr(session.config, "option", None)
    if option is None:
        return
    current = getattr(option, "cov_fail_under", None)
    if current is None:
        return
    if not _should_relax_coverage(session.config):
        return
    option.cov_fail_under = min(current, _MATPLOTLIB_OPTIONAL_FAIL_UNDER)
