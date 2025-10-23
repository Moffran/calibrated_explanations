from __future__ import annotations

import builtins

import numpy as np
import pytest

from calibrated_explanations.core import interval_regressor as interval_module
from calibrated_explanations.utils import helper as helper_module


class DummyCPS:
    """Lightweight conformal predictor stub used to drive deterministic tests."""

    def __init__(self):
        self.fit_calls: list[dict[str, object]] = []
        # Predict invocations pop values from this queue so tests can control outputs.
        self.predict_queue: list[float] = []
        self.alphas = np.array([], dtype=float)

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
        residuals = np.array(residuals, copy=True)
        if bins is None:
            self.alphas = np.sort(residuals.astype(float))
        else:
            bins = np.array(bins)
            mapping: dict[object, np.ndarray] = {}
            for value in np.unique(bins):
                mapping[value] = np.sort(residuals[bins == value].astype(float))
            self.alphas = (None, mapping)

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
    last_predict_bins: np.ndarray | None = None

    def __init__(self, _model, labels, interval_regressor, *, bins=None, cprobs=None):
        DummyVennAbers.last_init = {
            "labels": np.array(labels, copy=True),
            "interval_regressor": interval_regressor,
            "bins": None if bins is None else np.array(bins, copy=True),
            "cprobs": None if cprobs is None else np.array(cprobs, copy=True),
        }

    def predict_proba(self, x, *, output_interval=False, bins=None):  # pragma: no cover - trivial
        DummyVennAbers.last_predict_bins = None if bins is None else np.array(bins, copy=True)
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
    DummyVennAbers.last_predict_bins = None
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


def test_predict_probability_requires_calibration_bins_when_test_bins_provided(monkeypatch):
    regressor = _make_regressor(monkeypatch)
    x = np.array([[0.2, 0.1]])

    with pytest.raises(ValueError, match="Calibration bins must be assigned"):
        regressor.predict_probability(x, y_threshold=0.5, bins=np.array([0]))


def test_predict_probability_vector_threshold_uses_absolute_import(monkeypatch):
    regressor = _make_regressor(monkeypatch)
    x = np.array([[0.2, 0.1], [0.4, 0.3]])
    thresholds = np.array([0.4, 0.6])

    import builtins
    import importlib

    helper_module = importlib.import_module("calibrated_explanations.utils.helper")
    import_attempts: list[tuple[str, int]] = []

    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: D401 - test helper
        import_attempts.append((name, level))
        if level > 0 and name.endswith("utils.helper"):
            raise ImportError("relative helper unavailable")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    calls: list[int] = []

    def stub_safe_first_element(values, *, col=0):
        arr = np.asarray(values)
        calls.append(col)
        if arr.ndim == 2:
            return arr[0, col]
        return arr.ravel()[0]

    monkeypatch.setattr(helper_module, "safe_first_element", stub_safe_first_element, raising=False)

    proba, low, high, extra = regressor.predict_probability(x, y_threshold=thresholds)

    assert np.allclose(proba, 0.7)
    assert np.allclose(low, 0.1)
    assert np.allclose(high, 0.9)
    assert extra is None
    assert calls == [1, 0, 0, 1, 0, 0]

    regressor = _make_regressor(monkeypatch)
    regressor.split["cps"].predict_queue = [0.2, 0.8]
    x = np.array([[0.2, 0.1], [0.4, 0.3]])
    thresholds = np.array([0.25, 0.35])

    import builtins
    import importlib

    helper_module = importlib.import_module("calibrated_explanations.utils.helper")
    calls: list[tuple[np.ndarray, int | None]] = []

    def fake_safe_first_element(values, col=None):
        array = np.array(values)
        calls.append((array, col))
        if col is not None:
            return array[0, col]
        return array.flat[0]

    monkeypatch.setattr(helper_module, "safe_first_element", fake_safe_first_element)

    original_import = builtins.__import__

    def failing_import(
        name, globals=None, locals=None, fromlist=(), level=0
    ):  # pragma: no cover - helper
        if name.endswith("utils.helper") and level == 1:
            raise ImportError("forced failure")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", failing_import)

    proba, low, high, extra = regressor.predict_probability(x, y_threshold=thresholds)

    assert proba.shape == (2,)
    assert low.shape == (2,)
    assert high.shape == (2,)
    assert extra is None
    # Three invocations per instance: probability, low bound, high bound.
    assert len(calls) == 6
    assert calls[0][1] == 1  # first call extracts probability column


def test_predict_probability_normalizes_scalar_and_column_bins(monkeypatch):
    calibration_bins = np.array([0, 1, 0, 1])
    regressor = _make_regressor(monkeypatch, bins=calibration_bins)
    x = np.array([[0.2, 0.1], [0.4, 0.3]])

    expanded = np.array([5, 5])
    proba_exp, low_exp, high_exp, _ = regressor.predict_probability(
        x, y_threshold=0.5, bins=expanded
    )
    assert DummyVennAbers.last_predict_bins is not None
    assert DummyVennAbers.last_predict_bins.shape == (2,)
    assert np.all(DummyVennAbers.last_predict_bins == expanded)

    proba_scalar, low_scalar, high_scalar, _ = regressor.predict_probability(
        x, y_threshold=0.5, bins=5
    )
    assert np.allclose(proba_scalar, proba_exp)
    assert np.allclose(low_scalar, low_exp)
    assert np.allclose(high_scalar, high_exp)
    assert DummyVennAbers.last_predict_bins is not None
    assert DummyVennAbers.last_predict_bins.shape == (2,)
    assert np.all(DummyVennAbers.last_predict_bins == expanded)

    expected_column = np.array([7, 7])
    proba_expected_column, low_expected_column, high_expected_column, _ = (
        regressor.predict_probability(x, y_threshold=0.5, bins=expected_column)
    )
    assert DummyVennAbers.last_predict_bins is not None
    assert DummyVennAbers.last_predict_bins.shape == (2,)
    assert np.all(DummyVennAbers.last_predict_bins == expected_column)

    column_bins = np.array([[7], [7]])
    proba_column, low_column, high_column, _ = regressor.predict_probability(
        x, y_threshold=0.5, bins=column_bins
    )
    assert np.allclose(proba_column, proba_expected_column)
    assert np.allclose(low_column, low_expected_column)
    assert np.allclose(high_column, high_expected_column)
    assert DummyVennAbers.last_predict_bins is not None
    assert DummyVennAbers.last_predict_bins.shape == (2,)
    assert np.all(DummyVennAbers.last_predict_bins == expected_column)


def test_predict_probability_requires_calibration_bins(monkeypatch):
    regressor = _make_regressor(monkeypatch)
    x = np.array([[0.1, 0.2]])

    with pytest.raises(ValueError, match="Calibration bins must be assigned"):
        regressor.predict_probability(x, y_threshold=0.5, bins=np.array([0]))


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


def test_predict_probability_uses_fallback_safe_first_element(monkeypatch):
    regressor = _make_regressor(monkeypatch)
    regressor.split["cps"].predict_queue = [0.4, 0.6]
    x = np.array([[0.5, 0.1], [0.6, 0.2]])
    thresholds = np.array([0.3, 0.7])

    calls: list[tuple[int | None, float]] = []
    original_safe_first = helper_module.safe_first_element

    def tracking_safe_first(values, default=0.0, col=None):
        result = original_safe_first(values, default=default, col=col)
        calls.append((col, result))
        return result

    monkeypatch.setattr(helper_module, "safe_first_element", tracking_safe_first, raising=False)

    original_import = builtins.__import__
    failures = {"count": 0}

    def failing_relative_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 1 and name == "utils.helper":
            failures["count"] += 1
            raise ImportError("simulated relative failure")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", failing_relative_import)

    proba, _, _, _ = regressor.predict_probability(x, y_threshold=thresholds)

    assert np.allclose(proba, 0.7)
    assert failures["count"] >= thresholds.size
    assert len(calls) == thresholds.size * 3
    for offset in range(0, len(calls), 3):
        cols = [calls[offset + i][0] for i in range(3)]
        assert cols == [1, None, None]


def test_compute_proba_cal_rejects_invalid_threshold(monkeypatch):
    regressor = _make_regressor(monkeypatch)

    with pytest.raises(TypeError, match="y_threshold must be a float or a tuple"):
        regressor.compute_proba_cal({"not": "supported"})


def test_insert_calibration_updates_predictor_state(monkeypatch):
    base_bins = np.array([0, 1, 0, 1])
    regressor = _make_regressor(monkeypatch, bins=base_bins)
    updates = [
        (np.array([[0.9, 0.2]]), np.array([1.4]), np.array([0])),
        (np.array([[0.1, 0.8]]), np.array([1.2]), np.array([1])),
    ]

    expected_bins0 = np.array(regressor.cps.alphas[1][0], copy=True)
    expected_bins1 = np.array(regressor.cps.alphas[1][1], copy=True)
    appended_bins: list[int] = []

    for xs, ys, new_bins in updates:
        residual = ys - regressor.ce.predict_function(xs)
        regressor.insert_calibration(xs, ys, bins=new_bins)
        appended_bins.extend(new_bins.tolist())
        if new_bins[0] == 0:
            expected_bins0 = np.sort(np.concatenate([expected_bins0, residual]))
        else:
            expected_bins1 = np.sort(np.concatenate([expected_bins1, residual]))

    assert regressor.bins.shape[0] == base_bins.shape[0] + len(appended_bins)
    assert np.all(regressor.bins[-len(appended_bins) :] == np.array(appended_bins))
    assert np.allclose(regressor.cps.alphas[1][0], expected_bins0)
    assert np.allclose(regressor.cps.alphas[1][1], expected_bins1)


def test_insert_calibration_appends_bins_and_updates_alphas(monkeypatch):
    base_bins = np.zeros(4, dtype=int)
    regressor = _make_regressor(monkeypatch, bins=base_bins)
    regressor.split["cps"].alphas = [None, {0: np.array([-0.5, 0.0])}]
    regressor.cps.alphas = [None, {0: np.array([-0.6, -0.2])}]

    xs = np.array([[0.1, 0.2]])
    ys = np.array([0.2])
    new_bins = np.array([0])

    regressor.insert_calibration(xs, ys, bins=new_bins)

    assert np.array_equal(regressor.bins[-1:], new_bins)
    assert np.allclose(regressor.residual_cal[-1], ys[0] - xs.sum())
    assert np.allclose(regressor.split["cps"].alphas[1][0], np.array([-0.5, -0.1, 0.0]))
    assert np.allclose(regressor.cps.alphas[1][0], np.array([-0.6, -0.2, -0.1]))


def test_compute_proba_cal_rejects_unsupported_type(monkeypatch):
    regressor = _make_regressor(monkeypatch)

    with pytest.raises(TypeError, match="y_threshold must be a float or a tuple"):
        regressor.compute_proba_cal(object())


def test_insert_calibration_updates_with_bins(monkeypatch):
    base_bins = np.array([0, 1, 0, 1])
    regressor = _make_regressor(monkeypatch, bins=base_bins)
    xs = np.array([[0.5, 0.5], [0.6, 0.4]])
    ys = np.array([1.2, 1.5])
    new_bins = np.array([0, 1])

    regressor.insert_calibration(xs[[0]], ys[[0]], bins=new_bins[[0]])
    regressor.insert_calibration(xs[[1]], ys[[1]], bins=new_bins[[1]])

    assert np.array_equal(regressor.bins[-2:], new_bins)
    assert regressor.y_cal_hat.shape[0] == 6
    assert regressor.residual_cal.shape[0] == 6
    assert regressor.cps.alphas[1][0][-1] == pytest.approx(0.2)
    assert regressor.cps.alphas[1][1][-1] == pytest.approx(0.5)


def test_insert_calibration_updates_alphas_without_bins(monkeypatch):
    """Residual insertions update both CPS views when no Mondrian bins are used."""

    regressor = _make_regressor(monkeypatch)

    base_split_alphas = np.array(regressor.split["cps"].alphas, copy=True)
    base_cps_alphas = np.array(regressor.cps.alphas, copy=True)

    xs = np.array([[0.2, 0.7], [0.4, 0.1]])
    ys = np.array([0.95, 0.55])

    regressor.insert_calibration(xs, ys)

    residuals = ys - regressor.ce.predict_function(xs)
    expected_split = np.insert(
        base_split_alphas,
        np.searchsorted(base_split_alphas, residuals[0]),
        residuals[0],
    )
    expected_cps = np.insert(
        base_cps_alphas,
        np.searchsorted(base_cps_alphas, residuals),
        residuals,
    )

    assert np.allclose(regressor.split["cps"].alphas, expected_split)
    assert np.allclose(regressor.cps.alphas, expected_cps)


def test_compute_proba_cal_invalid_type(monkeypatch):
    regressor = _make_regressor(monkeypatch)

    with pytest.raises(TypeError, match="y_threshold must be a float or a tuple"):
        regressor.compute_proba_cal([0.1, 0.2])


def test_compute_proba_cal_tuple_threshold(monkeypatch):
    regressor = _make_regressor(monkeypatch)
    regressor.split["cps"].predict_queue = [0.2, 0.8]

    regressor.compute_proba_cal((0.15, 0.35))

    assert DummyVennAbers.last_init is not None
    cprobs = DummyVennAbers.last_init["cprobs"]
    assert cprobs.shape[1] == 2
    assert np.allclose(cprobs[:, 0] + cprobs[:, 1], 1.0)
    assert np.allclose(cprobs[:, 1], 0.6)
    labels = DummyVennAbers.last_init["labels"]
    assert set(np.unique(labels)) <= {0, 1}
