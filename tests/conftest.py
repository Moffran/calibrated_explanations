"""Shared pytest fixtures for CalibratedExplainer tests."""

from __future__ import annotations

import os
from typing import Any, Callable

import numpy as np
import pytest

from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer
from tests.helpers.model_utils import DummyLearner, DummyIntervalLearner
from tests.helpers.utils import get_env_flag

# Import additional fixtures from _fixtures module
from ._fixtures import binary_dataset, regression_dataset, multiclass_dataset  # noqa: F401, pylint: disable=unused-import


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
