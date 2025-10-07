from __future__ import annotations

import numpy as np
import pytest

from calibrated_explanations import _interval_regressor as interval_module


class DummyCPS:
    """Lightweight conformal predictor stub used to drive deterministic tests."""

    def __init__(self):
        self.fit_calls: list[dict[str, object]] = []
        # Predict invocations pop values from this queue so tests can control outputs.
        self.predict_queue: list[float] = []

    # pylint: disable=too-many-arguments
    def fit(self, *, residuals, sigmas=None, bins=None, seed=None):  # pragma: no cover - smoke
        self.fit_calls.append(
            {
                "residuals": np.array(residuals, copy=True),
                "sigmas": None if sigmas is None else np.array(sigmas, copy=True),
                "bins_provided": bins is not None,
                "seed": seed,
            }
        )
        if sigmas is None:
            sigmas = np.ones_like(residuals)
        self._last_sigmas = np.array(sigmas, copy=True)

    # pylint: disable=too-many-arguments
    def predict(
        self,
        *,
        y_hat,
        sigmas,
        y=None,
        lower_percentiles=None,
        higher_percentiles=None,
        bins=None,
    ):
        if y is None and lower_percentiles is not None and higher_percentiles is not None:
            n = y_hat.shape[0]
            # Return four columns mimicking conformal predictive system outputs.
            return np.column_stack(
                [
                    np.full(n, -2.0),
                    np.full(n, -1.0),
                    np.full(n, 1.0),
                    np.full(n, 2.0),
                ]
            )

        value = self.predict_queue.pop(0) if self.predict_queue else 0.5
        return np.full(y_hat.shape[0], value)


class DummyVennAbers:
    """Minimal Venn-Abers implementation that records the latest calibration call."""

    last_init: dict[str, object] | None = None

    def __init__(self, _model, labels, interval_regressor, *, bins=None, cprobs=None):
        DummyVennAbers.last_init = {
            "labels": np.array(labels, copy=True),
            "interval_regressor": interval_regressor,
            "bins": None if bins is None else np.array(bins, copy=True),
            "cprobs": None if cprobs is None else np.array(cprobs, copy=True),
        }

    def predict_proba(self, x, *, output_interval=False, bins=None):  # pragma: no cover - trivial
        n = x.shape[0]
        proba = np.tile(np.array([[0.3, 0.7]]), (n, 1))
        interval_low = np.full((n, 1), 0.1)
        interval_high = np.full((n, 1), 0.9)
        return proba, interval_low, interval_high


class DummyExplainer:
    """Test double exposing the attributes required by :class:`IntervalRegressor`."""

    def __init__(self, *, bins=None):
        self.bins = bins
        self.seed = 1234
        self.x_cal = np.array(
            [
                [0.1, 0.2],
                [0.2, 0.3],
                [0.3, 0.4],
                [0.4, 0.5],
            ]
        )
        self.y_cal = np.array([0.1, 0.2, 0.3, 0.4])
        self.difficulty_estimator = None

    def predict_calibration(self):
        return self.y_cal + 0.05

    def _get_sigma_test(self, x):  # pylint: disable=unused-argument
        return np.ones(len(x))

    def predict_function(self, x):
        x = np.atleast_2d(x)
        return np.sum(x, axis=1)


def _make_regressor(monkeypatch: pytest.MonkeyPatch, *, bins=None):
    monkeypatch.setattr(interval_module.crepes, "ConformalPredictiveSystem", DummyCPS)
    monkeypatch.setattr(interval_module, "VennAbers", DummyVennAbers)
    DummyVennAbers.last_init = None
    explainer = DummyExplainer(bins=bins)
    return interval_module.IntervalRegressor(explainer)


def test_predict_probability_scalar_threshold(monkeypatch):
    regressor = _make_regressor(monkeypatch)
    regressor.split["cps"].predict_queue = [0.6]
    x = np.array([[0.2, 0.1], [0.4, 0.3]])

    proba, low, high, extra = regressor.predict_probability(x, y_threshold=0.5)

    assert np.allclose(proba, 0.7)
    assert np.allclose(low, 0.1)
    assert np.allclose(high, 0.9)
    assert extra is None
    assert DummyVennAbers.last_init is not None
    assert DummyVennAbers.last_init["cprobs"].shape[1] == 2


def test_predict_probability_sequence_threshold(monkeypatch):
    regressor = _make_regressor(monkeypatch)
    # Each tuple threshold consumes two CPS predictions (lower + upper bound).
    regressor.split["cps"].predict_queue = [0.2, 0.8, 0.2, 0.8]
    x = np.array([[0.5, 0.1], [0.6, 0.2]])
    thresholds = [(0.0, 0.5), (0.0, 0.5)]

    proba, low, high, extra = regressor.predict_probability(x, y_threshold=thresholds)

    assert np.allclose(proba, 0.7)
    assert np.allclose(low, 0.1)
    assert np.allclose(high, 0.9)
    assert extra is None


def test_predict_uncertainty_uses_interval_outputs(monkeypatch):
    regressor = _make_regressor(monkeypatch)
    x = np.array([[0.3, 0.2]])

    median, low, high, extra = regressor.predict_uncertainty(x, low_high_percentiles=(5, 95))

    assert np.allclose(median, 0.5)
    assert np.allclose(low, -2.0)
    assert np.allclose(high, 1.0)
    assert extra is None


def test_insert_calibration_requires_bins_when_existing_none(monkeypatch):
    regressor = _make_regressor(monkeypatch)
    xs = np.array([[0.1, 0.2], [0.2, 0.3]])
    ys = np.array([0.5, 0.6])

    with pytest.raises(ValueError, match="Cannot mix calibration instances with and without bins"):
        regressor.insert_calibration(xs, ys, bins=np.array([0, 1]))


def test_insert_calibration_validates_bin_length(monkeypatch):
    base_bins = np.zeros(4, dtype=int)
    regressor = _make_regressor(monkeypatch, bins=base_bins)
    xs = np.array([[0.1, 0.2], [0.2, 0.3]])
    ys = np.array([0.5, 0.6])

    with pytest.raises(ValueError, match="length of bins"):
        regressor.insert_calibration(xs, ys, bins=np.array([0]))
