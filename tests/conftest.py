"""Shared pytest fixtures for CalibratedExplainer tests."""

from __future__ import annotations

import os
from typing import Any, Callable

import numpy as np
import pytest

from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer

# Import additional fixtures from _fixtures module
from ._fixtures import binary_dataset, regression_dataset, multiclass_dataset  # noqa: F401, pylint: disable=unused-import


def _env_flag(name: str) -> bool:
    """Return a boolean for environment flags treating common truthy strings as True.

    This treats '1', 'true', 'yes', 'y' (case-insensitive) as True. Everything
    else (including '0', 'false', empty string or unset) is False. Using this
    avoids Python's truthiness where a non-empty string like '0' evaluates to True.
    """
    val = os.getenv(name, "").strip().lower()
    return val in ("1", "true", "yes", "y")


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
    if _env_flag("FAST_TESTS"):
        return max(100, minimum_limit)
    return max(500, minimum_limit)


class _FixtureLearner:
    """Minimal learner satisfying the CalibratedExplainer contract."""

    def __init__(self, *, mode: str = "classification") -> None:
        self.mode = mode
        self.fitted_ = True  # satisfies check_is_fitted

    def fit(self, x: np.ndarray, y: np.ndarray) -> "_FixtureLearner":  # pragma: no cover - unused
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return np.zeros(x.shape[0])

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        if self.mode == "classification":
            probs = np.zeros((x.shape[0], 2))
            probs[:, 0] = 0.4
            probs[:, 1] = 0.6
            return probs
        return np.zeros((x.shape[0], 1))


class _FixtureIntervalLearner:
    """Deterministic interval learner for lightweight tests."""

    def predict_uncertainty(
        self, x: np.ndarray, *_args: Any, **_kwargs: Any
    ) -> tuple[np.ndarray, ...]:
        n = np.atleast_2d(x).shape[0]
        zeros = np.zeros(n)
        return zeros, zeros, zeros, None

    def predict_probability(
        self, x: np.ndarray, *_args: Any, **_kwargs: Any
    ) -> tuple[np.ndarray, ...]:
        n = np.atleast_2d(x).shape[0]
        zeros = np.zeros(n)
        return zeros, zeros, zeros, None

    def predict_proba(
        self, x: np.ndarray, *_args: Any, **_kwargs: Any
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.atleast_2d(x)
        probs = np.zeros((x.shape[0], 2))
        probs[:, 0] = 0.5
        probs[:, 1] = 0.5
        low = np.zeros_like(probs)
        high = np.ones_like(probs)
        return probs, low, high


@pytest.fixture
def explainer_factory(monkeypatch: pytest.MonkeyPatch) -> Callable[..., CalibratedExplainer]:
    """Return a factory that builds fully initialized CalibratedExplainer instances."""

    def _initialize_interval(explainer: CalibratedExplainer, *_args: Any, **_kwargs: Any) -> None:
        explainer.interval_learner = _FixtureIntervalLearner()
        explainer._CalibratedExplainer__initialized = True  # noqa: SLF001

    monkeypatch.setattr(
        "calibrated_explanations.core.calibration.interval_learner.initialize_interval_learner",
        _initialize_interval,
    )
    monkeypatch.setattr(
        "calibrated_explanations.core.calibration.interval_learner.initialize_interval_learner_for_fast_explainer",
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
            learner = _FixtureLearner(mode=mode)
        if x_cal is None:
            x_cal = np.asarray([[0.0, 1.0], [1.0, 2.0]], dtype=float)
        if y_cal is None:
            if mode == "classification":
                y_cal = np.asarray([0, 1], dtype=int)
            else:
                y_cal = np.asarray([0.1, 0.9], dtype=float)
        return CalibratedExplainer(learner, x_cal, y_cal, mode=mode, **kwargs)

    return _factory
