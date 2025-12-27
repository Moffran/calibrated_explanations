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

from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer
from tests.helpers.model_utils import DummyLearner, DummyIntervalLearner
from tests.helpers.utils import get_env_flag

# Import additional fixtures from _fixtures module
from ._fixtures import binary_dataset, regression_dataset, multiclass_dataset  # noqa: F401, pylint: disable=unused-import

ATTR_RE = re.compile(r"\._[A-Za-z0-9_]+")
GETATTR_RE = re.compile(r"getattr\([^,]+,\s*'(_[A-Za-z0-9_]+)'")


def _load_allowlist(path: Path):
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf8"))
        return raw.get("allowlist", [])
    except Exception:
        return []


def _is_expired(entry: dict) -> bool:
    expiry = entry.get("expiry")
    if not expiry:
        return False
    try:
        dt = datetime.fromisoformat(expiry)
        return dt.date() < datetime.utcnow().date()
    except Exception:
        return False


def _scan_and_check(root: Path):
    tests_dir = root / "tests"
    if not tests_dir.exists():
        return []
    findings = []
    for p in tests_dir.rglob("*.py"):
        try:
            txt = p.read_text(encoding="utf8")
        except Exception:
            continue
        syms = {m.lstrip(".") for m in ATTR_RE.findall(txt)}
        syms.update(m for m in GETATTR_RE.findall(txt))
        if syms:
            findings.append((p, sorted(syms)))
    return findings


def pytest_sessionstart(session):
    root = Path(session.config.rootpath)
    findings = _scan_and_check(root)
    if not findings:
        return

    allowlist_path = root / ".github" / "private_member_allowlist.json"
    allowlist = _load_allowlist(allowlist_path)
    allowed = set()
    expired_allowed = []
    for e in allowlist:
        key = (e.get("file"), e.get("symbol"))
        if key[0] and key[1]:
            if _is_expired(e):
                expired_allowed.append(e)
            else:
                allowed.add(key)

    violations = []
    for p, syms in findings:
        rel = str(p).replace("\\\\", "/")
        for s in syms:
            if (rel, s) in allowed or (os.path.basename(rel), s) in allowed:
                continue
            violations.append((rel, s))

    if not violations:
        if expired_allowed:
            warnings.warn("Some allowlist entries are expired; please review them.")
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

    # Fail in CI (GitHub Actions) or when explicitly requested via env var
    ci = os.environ.get("GITHUB_ACTIONS") == "true"
    hard_fail = os.environ.get("CE_FAIL_PRIVATE_TESTS") == "1"
    if ci or hard_fail:
        raise pytest.UsageError(msg)
    else:
        warnings.warn(msg)


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
def explainer_factory(monkeypatch: pytest.MonkeyPatch) -> Callable[..., CalibratedExplainer]:
    """Return a factory that builds fully initialized CalibratedExplainer instances."""

    def _initialize_interval(explainer: CalibratedExplainer, *_args: Any, **_kwargs: Any) -> None:
        explainer.interval_learner = DummyIntervalLearner()
        explainer._CalibratedExplainer__initialized = True  # noqa: SLF001

    monkeypatch.setattr(
        "calibrated_explanations.calibration.interval_learner.initialize_interval_learner",
        _initialize_interval,
    )
    monkeypatch.setattr(
        "calibrated_explanations.calibration.interval_learner.initialize_interval_learner_for_fast_explainer",
        _initialize_interval,
    )

    def _factory(
        *,
        mode: str = "classification",
        learner: Any | None = None,
        x_cal: np.ndarray | None = None,
        y_cal: np.ndarray | None = None,
        **kwargs: Any,
    ) -> CalibratedExplainer:
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

    return _factory


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
