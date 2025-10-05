"""Regression tests for WrapCalibratedExplainer keyword defaults and alias handling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from calibrated_explanations.api.config import ExplainerConfig
from calibrated_explanations.core.wrap_explainer import WrapCalibratedExplainer


class _RecordingExplainer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.plot_calls: list[tuple[Any, dict[str, Any]]] = []

    def explain_factual(self, X_test: Any, **kwargs: Any) -> str:
        self.calls.append(("factual", dict(kwargs)))
        return "factual"

    def explore_alternatives(self, X_test: Any, **kwargs: Any) -> str:
        self.calls.append(("alternative", dict(kwargs)))
        return "alternative"

    def explain_fast(self, X_test: Any, **kwargs: Any) -> str:
        self.calls.append(("fast", dict(kwargs)))
        return "fast"

    def plot(self, X_test: Any, *, threshold: float | None = None, **kwargs: Any) -> None:
        payload = dict(kwargs)
        payload["threshold"] = threshold
        self.plot_calls.append((X_test, payload))


@dataclass
class _DummyModel:
    """Tiny stand-in compatible with WrapCalibratedExplainer._from_config."""

    value: float = 0.0


def _configured_wrapper(
    threshold: float | None, percentiles: tuple[int, int]
) -> tuple[WrapCalibratedExplainer, _RecordingExplainer]:
    cfg = ExplainerConfig(
        model=_DummyModel(), threshold=threshold, low_high_percentiles=percentiles
    )
    wrapper = WrapCalibratedExplainer._from_config(cfg)
    recorder = _RecordingExplainer()
    wrapper.explainer = recorder
    wrapper.fitted = True
    wrapper.calibrated = True
    wrapper.mc = None
    return wrapper, recorder


def test_config_defaults_forwarded_when_missing() -> None:
    wrapper, recorder = _configured_wrapper(threshold=0.42, percentiles=(10, 90))
    X = np.ones((3, 2))

    wrapper.explain_factual(X)

    assert recorder.calls, "expected explain_factual to be forwarded"
    mode, kwargs = recorder.calls[-1]
    assert mode == "factual"
    assert kwargs["threshold"] == 0.42
    assert kwargs["low_high_percentiles"] == (10, 90)
    assert "bins" in kwargs


def test_user_overrides_win_and_aliases_are_dropped() -> None:
    wrapper, recorder = _configured_wrapper(threshold=0.15, percentiles=(5, 95))
    X = np.ones((2, 2))

    with pytest.deprecated_call():
        wrapper.explore_alternatives(X, threshold=0.7, alpha=(1, 99))

    mode, kwargs = recorder.calls[-1]
    assert mode == "alternative"
    assert kwargs["threshold"] == 0.7
    # Alias keys are stripped; defaults reappear instead of alias payloads.
    assert kwargs["low_high_percentiles"] == (5, 95)
    assert "alpha" not in kwargs


def test_plot_inherits_threshold_and_bins_from_config_and_mc() -> None:
    wrapper, recorder = _configured_wrapper(threshold=0.33, percentiles=(20, 80))
    wrapper.mc = lambda X: np.arange(len(X))
    X = np.zeros((4, 2))

    wrapper.plot(X)

    assert recorder.plot_calls, "expected plot call to be forwarded"
    _X_payload, kwargs = recorder.plot_calls[-1]
    assert kwargs["threshold"] == 0.33
    assert kwargs["low_high_percentiles"] == (20, 80)
    assert np.array_equal(kwargs["bins"], np.arange(len(X)))
