"""Regression tests for WrapCalibratedExplainer keyword defaults and alias handling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from sklearn.base import BaseEstimator

from calibrated_explanations.api.config import ExplainerConfig
from calibrated_explanations.core.wrap_explainer import WrapCalibratedExplainer
from tests.helpers.deprecation import warns_or_raises, deprecations_error_enabled


class RecordingExplainer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.plot_calls: list[tuple[Any, dict[str, Any]]] = []

    def explain_factual(self, x: Any, **kwargs: Any) -> str:
        self.calls.append(("factual", dict(kwargs)))
        return "factual"

    def explore_alternatives(self, x: Any, **kwargs: Any) -> str:
        self.calls.append(("alternative", dict(kwargs)))
        return "alternative"

    def explain_fast(self, x: Any, **kwargs: Any) -> str:
        self.calls.append(("fast", dict(kwargs)))
        return "fast"

    def plot(self, x: Any, *, threshold: float | None = None, **kwargs: Any) -> None:
        payload = dict(kwargs)
        payload["threshold"] = threshold
        self.plot_calls.append((x, payload))


@dataclass
class DummyModel(BaseEstimator):
    """Tiny stand-in compatible with WrapCalibratedExplainer._from_config."""

    value: float = 0.0

    def fit(self, x=None, y=None):
        self.value_ = self.value
        return self


def configured_wrapper(
    threshold: float | None, percentiles: tuple[int, int]
) -> tuple[WrapCalibratedExplainer, RecordingExplainer]:
    cfg = ExplainerConfig(model=DummyModel(), threshold=threshold, low_high_percentiles=percentiles)
    cfg.model.fit(None, None)
    wrapper = WrapCalibratedExplainer.from_config(cfg)
    recorder = RecordingExplainer()
    wrapper.explainer = recorder
    wrapper.fitted = True
    wrapper.calibrated = True
    wrapper.mc = None
    return wrapper, recorder


def test_config_defaults_forwarded_when_missing() -> None:
    wrapper, recorder = configured_wrapper(threshold=0.42, percentiles=(10, 90))
    x = np.ones((3, 2))

    wrapper.explain_factual(x)

    assert recorder.calls, "expected explain_factual to be forwarded"
    mode, kwargs = recorder.calls[-1]
    assert mode == "factual"
    assert kwargs["threshold"] == 0.42
    assert kwargs["low_high_percentiles"] == (10, 90)
    assert "bins" in kwargs


def test_user_overrides_win_and_aliases_are_dropped() -> None:
    wrapper, recorder = configured_wrapper(threshold=0.15, percentiles=(5, 95))
    x = np.ones((2, 2))

    if deprecations_error_enabled():
        with pytest.raises(DeprecationWarning):
            wrapper.explore_alternatives(x, threshold=0.7, alpha=(1, 99))
    else:
        with warns_or_raises():
            wrapper.explore_alternatives(x, threshold=0.7, alpha=(1, 99))

        mode, kwargs = recorder.calls[-1]
        assert mode == "alternative"
        assert kwargs["threshold"] == 0.7
        # Alias keys are stripped; defaults reappear instead of alias payloads.
        assert kwargs["low_high_percentiles"] == (5, 95)
        assert "alpha" not in kwargs


def test_plot_inherits_threshold_and_bins_from_config_and_mc() -> None:
    wrapper, recorder = configured_wrapper(threshold=0.33, percentiles=(20, 80))
    wrapper.mc = lambda x: np.arange(len(x))
    x = np.zeros((4, 2))

    wrapper.plot(x)

    assert recorder.plot_calls, "expected plot call to be forwarded"
    x_payload, kwargs = recorder.plot_calls[-1]
    assert kwargs["threshold"] == 0.33
    assert kwargs["low_high_percentiles"] == (20, 80)
    assert np.array_equal(kwargs["bins"], np.arange(len(x)))
