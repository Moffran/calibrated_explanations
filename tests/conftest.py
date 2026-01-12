"""Shared pytest fixtures for CalibratedExplainer tests."""

from __future__ import annotations
import json
import re
import warnings
from datetime import datetime
from pathlib import Path
import os
from typing import Any, Callable

import numpy as np
import pytest
import importlib
from importlib.util import find_spec
from _pytest.fixtures import FixtureRequest


@pytest.fixture(autouse=True)
def skip_viz_if_missing(request: FixtureRequest):
    """Auto-skip tests marked with ``viz`` or ``viz_render`` when matplotlib is missing.

    This keeps the test-suite runnable in minimal environments while allowing
    viz tests to run when the optional `[viz]` extras are installed in CI.
    """
    if request.node.get_closest_marker("viz") or request.node.get_closest_marker("viz_render"):
        if find_spec("matplotlib") is None:
            pytest.skip("matplotlib not installed; skipping viz tests")


@pytest.fixture(scope="session", autouse=True)
def debug_matplotlib_session_state(request: FixtureRequest):
    """Write a small debug file describing matplotlib import state for debugging.

    This is a temporary diagnostic helper to reproduce a failing import that
    appears only under pytest (with coverage). It records `sys.modules` keys
    related to matplotlib and a few attributes that tests rely on.
    """
    import sys
    import json
    from pathlib import Path

    root = Path(request.config.rootpath)
    out = root / ".pytest_matplotlib_debug.json"
    try:
        found = find_spec("matplotlib") is not None
        mods = [k for k in sys.modules.keys() if k.startswith("matplotlib")]
        modules = {}
        for m in mods:
            mod = sys.modules.get(m)
            try:
                modules[m] = {
                    "repr": repr(mod),
                    "file": getattr(mod, "__file__", None),
                    "has_artist": hasattr(mod, "artist"),
                    "has_figure": hasattr(mod, "figure"),
                    "spec": getattr(mod, "__spec__", None).name
                    if getattr(mod, "__spec__", None)
                    else None,
                }
            except Exception:
                modules[m] = {"repr": "<uninspectable>"}

        data = {
            "timestamp": datetime.utcnow().isoformat(),
            "matplotlib_findable": found,
            "sys_path": list(sys.path),
            "matplotlib_modules": modules,
        }
        out.write_text(json.dumps(data, indent=2), encoding="utf8")
    except Exception:
        # Best-effort only; don't fail the test session because of debugging.
        pass
    yield


"""Avoid heavy top-level imports (CalibratedExplainer) to keep pytest
collection fast. Imports of the explainer class are performed lazily inside
fixtures that require it."""
from tests.helpers.analysis_utils import extract_private_symbols_from_ast, parse_version_token
from tests.helpers.model_utils import DummyLearner, DummyIntervalLearner
from tests.helpers.utils import get_env_flag

# Import additional fixtures from _fixtures module
from tests.helpers.fixtures import binary_dataset, regression_dataset, multiclass_dataset  # noqa: F401, pylint: disable=unused-import

ATTR_RE = re.compile(r"\._(?!_)[A-Za-z0-9_]+")
GETATTR_RE = re.compile(r"getattr\([^,]+,\s*'(?!__)(_[A-Za-z0-9_]+)'")


def load_allowlist(path: Path):
    """Return allowlist entries from a JSON file, or an empty list when missing."""
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf8"))
        return raw.get("allowlist", [])
    except Exception:
        return []


def is_expired(entry: dict) -> bool:
    """Return True when the allowlist entry expiry is in the past.

    Supports two expiry formats:
    - ISO date strings parsable by ``datetime.fromisoformat``
    - Semantic version tokens like ``v0.11.0`` which expire when the
      package version is greater-or-equal to the token.
    """
    expiry = entry.get("expiry")
    if not expiry:
        return False

    # Try ISO date first
    try:
        dt = datetime.fromisoformat(expiry)
        return dt.date() < datetime.utcnow().date()
    except Exception:
        pass

    # Fallback: accept semantic version tokens like 'v0.11.0'
    target = parse_version_token(expiry)
    if target is None:
        return False

    try:
        # Import package version in a minimal way
        from calibrated_explanations import __version__ as pkg_version  # local import

        current = parse_version_token(pkg_version)
        if current is None:
            return False
        return current >= target
    except Exception:
        return False


def scan_and_check(root: Path):
    """Scan test files for private-member usage and return any matches."""
    tests_dir = root / "tests"
    if not tests_dir.exists():
        return []
    findings = []
    for p in tests_dir.rglob("*.py"):
        try:
            txt = p.read_text(encoding="utf8")
        except Exception:
            continue
        syms = extract_private_symbols_from_ast(txt, p)
        if syms is None:
            syms = {m.lstrip(".") for m in ATTR_RE.findall(txt)}
            syms.update(m for m in GETATTR_RE.findall(txt))
        if syms:
            findings.append((p, sorted(syms)))
    return findings


def pytest_sessionstart(session):
    """Enforce private-member policy at session start with warnings or errors."""
    tracer_enabled = os.environ.get("CE_MPL_IMPORT_TRACER") == "1"
    if tracer_enabled:
        # Install a temporary import tracer for matplotlib to capture import-time
        # activity during test collection and early test execution. This helps
        # identify which test/module triggers partial or fake `matplotlib` entries
        # in `sys.modules` that later cause AttributeError in pyplot imports.
        try:
            import builtins
            import sys

            orig_import = builtins.__import__

            def tracing_import(name, globals=None, locals=None, fromlist=(), level=0):
                # If importing any matplotlib submodule, proactively import a
                # small set of commonly-used submodules first so that import-time
                # attribute lookups (e.g., decorators in `pyplot`) succeed even
                # when tests or other code have manipulated `sys.modules`.
                try:
                    if isinstance(name, str) and name.startswith("matplotlib"):
                        preload = (
                            "matplotlib.artist",
                            "matplotlib.figure",
                            "matplotlib.axes",
                            "matplotlib.image",
                            "matplotlib.cm",
                            "matplotlib.colors",
                            "matplotlib.transforms",
                            "matplotlib.path",
                            "matplotlib.collections",
                            "matplotlib.lines",
                            "matplotlib.patches",
                            "matplotlib.text",
                            "matplotlib.textpath",
                            "matplotlib.font_manager",
                            "matplotlib.backend_bases",
                            "matplotlib.backends",
                        )
                        for sub in preload:
                            try:
                                try:
                                    orig_import(sub, None, None, (), 0)
                                except Exception:
                                    try:
                                        importlib.import_module(sub)
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                except Exception:
                    pass
                mod = orig_import(name, globals, locals, fromlist, level)
                try:
                    if name == "matplotlib" or (
                        isinstance(name, str) and name.startswith("matplotlib.")
                    ):
                        root = Path(session.config.rootpath)
                        out = root / ".pytest_matplotlib_imports.log"
                        entry = {
                            "timestamp": datetime.utcnow().isoformat(),
                            "import_name": name,
                            "module_repr": repr(mod),
                            "module_file": getattr(mod, "__file__", None),
                            "sys_modules_keys": [
                                k for k in sys.modules.keys() if k.startswith("matplotlib")
                            ],
                        }
                        try:
                            with out.open("a", encoding="utf8") as fh:
                                fh.write(json.dumps(entry) + "\n")
                        except Exception:
                            pass
                except Exception:
                    pass
                return mod

            builtins.__import__ = tracing_import
        except Exception:
            # Best-effort tracing only
            pass
        try:
            if find_spec("matplotlib") is not None:
                mpl = importlib.import_module("matplotlib")
                preload = (
                    "matplotlib.artist",
                    "matplotlib.figure",
                    "matplotlib.axes",
                    "matplotlib.image",
                    "matplotlib.cm",
                    "matplotlib.colors",
                    "matplotlib.transforms",
                    "matplotlib.path",
                    "matplotlib.collections",
                    "matplotlib.lines",
                    "matplotlib.patches",
                    "matplotlib.text",
                    "matplotlib.textpath",
                    "matplotlib.font_manager",
                    "matplotlib.backend_bases",
                    "matplotlib.backends",
                )
                for name in preload:
                    try:
                        sub = importlib.import_module(name)
                        parts = name.split(".")
                        if len(parts) >= 2 and parts[0] == "matplotlib":
                            top = parts[1]
                            if not hasattr(mpl, top):
                                try:
                                    setattr(mpl, top, sub)
                                except Exception:
                                    pass
                    except Exception:
                        # best-effort; do not fail the test session
                        pass
        except Exception:
            pass
        root = Path(session.config.rootpath)
    else:
        root = Path(session.config.rootpath)
    findings = scan_and_check(root)
    if not findings:
        return

    allowlist_path = root / ".github" / "private_member_allowlist.json"
    allowlist = load_allowlist(allowlist_path)
    non_legacy_allowlist = []
    allowed = set()
    expired_allowed = []
    for e in allowlist:
        f = e.get("file")
        if f:
            f = f.replace("\\", "/")
        key = (f, e.get("symbol"))
        if key[0] and key[1]:
            if is_expired(e):
                expired_allowed.append(e)
            else:
                allowed.add(key)
                if key[0] and "/legacy/" not in key[0]:
                    non_legacy_allowlist.append(e)

    if non_legacy_allowlist:
        msg_lines = [
            "Private-member allowlist entries exist outside tests/legacy/.",
            "Please refactor the tests to use public APIs and remove these entries.",
        ]
        for entry in non_legacy_allowlist:
            msg_lines.append(
                f"- {entry.get('file')} ({entry.get('symbol')}) expires {entry.get('expiry')}"
            )
        raise pytest.UsageError("\n".join(msg_lines))

    violations = []
    for p, syms in findings:
        rel = p.relative_to(root).as_posix()
        for s in syms:
            if (rel, s) in allowed or (os.path.basename(rel), s) in allowed:
                continue
            violations.append((rel, s))

    if not violations:
        if expired_allowed:
            warnings.warn("Some allowlist entries are expired; please review them.")
        if non_legacy_allowlist:
            warnings.warn(
                "Private-member allowlist entries exist outside tests/legacy; please refactor and prune."
            )
        return

    msg_lines = [
        "Private-member usage detected in tests/",
        "The repository enforces using public APIs in tests. Found the following:",
    ]
    for f, s in violations:
        msg_lines.append(f"- {f}: {s}")
    msg_lines.append("")
    msg_lines.append(
        "If this is a temporary exception, add an entry to .github/private_member_allowlist.json with an expiry date."
    )
    msg = "\n".join(msg_lines)

    # Always fail when violations are present to keep local runs aligned with CI.
    raise pytest.UsageError(msg)


def pytest_collection_modifyitems(config, items):
    """Skip visualization tests when optional viz extras are not installed.

    Tests marked with `@pytest.mark.viz` will be skipped if `matplotlib`
    cannot be imported. This keeps the test suite green for core-only installs
    while still running viz tests when the `viz` extras are available.
    """
    try:
        importlib.import_module("matplotlib")
        return
    except Exception:
        skip_marker = pytest.mark.skip(reason="matplotlib is not installed; skipping viz tests")
        for item in items:
            if "viz" in item.keywords or "viz_render" in item.keywords:
                item.add_marker(skip_marker)


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
            limit = int(val)
            # enforce a sensible lower bound to avoid dataset-splitting errors
            return limit if limit >= minimum_limit else minimum_limit
        except Exception:  # pylint: disable=broad-except
            pass
    if get_env_flag("FAST_TESTS"):
        return max(100, minimum_limit)
    return max(500, minimum_limit)


@pytest.fixture
def explainer_factory(monkeypatch: pytest.MonkeyPatch) -> Callable[..., "CalibratedExplainer"]:
    """Return a factory that builds fully initialized CalibratedExplainer instances."""

    def initialize_interval(explainer: CalibratedExplainer, *_args: Any, **_kwargs: Any) -> None:
        explainer.interval_learner = DummyIntervalLearner()
        setattr(explainer, "_CalibratedExplainer__initialized", True)  # noqa: SLF001

    monkeypatch.setattr(
        "calibrated_explanations.calibration.interval_learner.initialize_interval_learner",
        initialize_interval,
    )
    monkeypatch.setattr(
        "calibrated_explanations.calibration.interval_learner.initialize_interval_learner_for_fast_explainer",
        initialize_interval,
    )

    def factory(
        *,
        mode: str = "classification",
        learner: Any | None = None,
        x_cal: np.ndarray | None = None,
        y_cal: np.ndarray | None = None,
        **kwargs: Any,
    ) -> CalibratedExplainer:
        # Local import to avoid importing the full calibrated_explainer module
        # during pytest collection. This keeps `pytest` responsive when run
        # without needing the full plotting/calibration stack.
        from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer

        if learner is None:
            learner = DummyLearner(mode=mode)
        if x_cal is None:
            x_cal = np.asarray([[0.0, 1.0], [1.0, 2.0]], dtype=float)
        if y_cal is None:
            if mode == "classification":
                y_cal = np.asarray([0, 1], dtype=int)
            else:
                y_cal = np.asarray([0.1, 0.9], dtype=float)
        return CalibratedExplainer(learner, x_cal, y_cal, mode=mode, **kwargs)

    return factory


@pytest.fixture(autouse=True)
def disable_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable all plugin fallback chains by default.

    This fixture is applied automatically to all tests (autouse=True) to enforce
    the Fallback Chain Enforcement policy from .github/tests-guidance.md.

    Tests that explicitly need to validate fallback behavior must use the
    `enable_fallbacks` fixture to opt in.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest monkeypatch fixture for setting environment variables.

    Notes
    -----
    This ensures tests run against the primary code path without falling back
    to legacy implementations. If a test triggers a fallback warning, it will
    fail in CI unless it explicitly uses `enable_fallbacks`.

    See Also
    --------
    enable_fallbacks : Fixture to explicitly enable fallbacks for specific tests
    tests.helpers.fallback_control : Helper functions for fallback management
    """
    from tests.helpers.fallback_control import disable_all_fallbacks

    disable_all_fallbacks(monkeypatch)
    # Tests expect CI detection to be opt-in; clear CI/GitHub markers so auto strategy tests run as intended.
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)


@pytest.fixture
def enable_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enable fallback chains for tests that explicitly validate fallback behavior.

    This fixture removes the fallback restrictions imposed by the autouse
    `disable_fallbacks` fixture, allowing tests to validate that fallback
    chains work correctly.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest monkeypatch fixture for setting environment variables.

    Notes
    -----
    Only use this fixture when the test is explicitly validating fallback behavior:
    - Testing error recovery paths
    - Testing graceful degradation
    - Testing compatibility with missing optional dependencies
    - Testing the fallback mechanism itself

    Do NOT use this fixture for normal feature tests.

    Examples
    --------
    >>> def test_explanation_plugin_fallback(enable_fallbacks):
    ...     '''Verify fallback chain activates when primary plugin fails.'''
    ...     explainer = CalibratedExplainer(
    ...         model, x_cal, y_cal,
    ...         _explanation_plugin_override="intentionally-missing"
    ...     )
    ...     with pytest.warns(UserWarning, match="fallback"):
    ...         explanation = explainer.explain_factual(x_test)
    ...     assert explanation is not None

    See Also
    --------
    disable_fallbacks : Autouse fixture that disables fallbacks by default
    tests.helpers.fallback_control : Helper functions for fallback management
    """
    # Remove the environment variables set by disable_fallbacks
    monkeypatch.delenv("CE_EXPLANATION_PLUGIN_FACTUAL_FALLBACKS", raising=False)
    monkeypatch.delenv("CE_EXPLANATION_PLUGIN_ALTERNATIVE_FALLBACKS", raising=False)
    monkeypatch.delenv("CE_EXPLANATION_PLUGIN_FAST_FALLBACKS", raising=False)
    monkeypatch.delenv("CE_INTERVAL_PLUGIN_FALLBACKS", raising=False)
    monkeypatch.delenv("CE_INTERVAL_PLUGIN_FAST_FALLBACKS", raising=False)
    monkeypatch.delenv("CE_PLOT_STYLE_FALLBACKS", raising=False)
    monkeypatch.delenv("CE_PARALLEL_MIN_BATCH_SIZE", raising=False)
    # Restore runtime warning behaviour so tests can observe fallback warnings
    from tests.helpers.fallback_control import restore_runtime_warnings

    restore_runtime_warnings(monkeypatch)
