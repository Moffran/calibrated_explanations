from __future__ import annotations

import numpy as np
import pytest

import calibrated_explanations.utils as utils_module
from calibrated_explanations.calibration import interval_regressor as interval_module
from calibrated_explanations.core import ConfigurationError, DataShapeError


class DummyCPS:
    """Lightweight conformal predictor stub used to drive deterministic tests."""

    def __init__(self):
        self.fit_calls: list[dict[str, object]] = []
        # Predict invocations pop values from this queue so tests can control outputs.
        self.predict_queue: list[float] = []
        self.alphas = np.array([], dtype=float)
        self.binned_alphas = None

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
        self.last_sigmas = np.array(sigmas, copy=True)
        residuals = np.array(residuals, copy=True)
        if bins is None:
            self.alphas = np.sort(residuals.astype(float))
            self.binned_alphas = None
        else:
            bins = np.array(bins)
            unique_bins = np.unique(bins)
            alpha_list = []
            mapping: dict[object, np.ndarray] = {}
            for value in unique_bins:
                sorted_residuals = np.sort(residuals[bins == value].astype(float))
                mapping[value] = sorted_residuals
                alpha_list.append(sorted_residuals)
            self.alphas = (None, mapping)
            self.binned_alphas = (unique_bins, alpha_list)

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

    def get_sigma_test(self, x):  # pylint: disable=unused-argument
        return np.ones(len(x))

    def predict_function(self, x):
        x = np.atleast_2d(x)
        return np.sum(x, axis=1)


def make_regressor(monkeypatch: pytest.MonkeyPatch, *, bins=None):
    monkeypatch.setattr(interval_module.crepes, "ConformalPredictiveSystem", DummyCPS)
    monkeypatch.setattr(interval_module, "VennAbers", DummyVennAbers)
    DummyVennAbers.last_init = None
    DummyVennAbers.last_predict_bins = None
    explainer = DummyExplainer(bins=bins)
    return interval_module.IntervalRegressor(explainer)


def test_initializer_flattens_calibration_arrays(monkeypatch):
    class ColumnExplainer(DummyExplainer):
        def __init__(self):
            super().__init__(bins=np.array([[0], [1], [0], [1]]))
            self.y_cal = self.y_cal.reshape(-1, 1)

        def predict_calibration(self):
            base = super().predict_calibration()
            return base.reshape(-1, 1)








def test_bins_setter_flattens_column_vectors(monkeypatch):
    regressor = make_regressor(monkeypatch)

    regressor.bins = np.array([[0], [1], [1], [0]])

    assert regressor.bins_storage.ndim == 1
    assert np.array_equal(regressor.bins, np.array([0, 1, 1, 0]))




def test_predict_probability_normalizes_scalar_and_column_bins(monkeypatch):
    calibration_bins = np.array([0, 1, 0, 1])
    regressor = make_regressor(monkeypatch, bins=calibration_bins)
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
    regressor = make_regressor(monkeypatch)
    x = np.array([[0.1, 0.2]])

    with pytest.raises(ConfigurationError, match="Calibration bins must be assigned"):
        regressor.predict_probability(x, y_threshold=0.5, bins=np.array([0]))


def test_predict_probability_rejects_mismatched_bin_length(monkeypatch):
    calibration_bins = np.array([0, 1, 0, 1])
    regressor = make_regressor(monkeypatch, bins=calibration_bins)
    x = np.array([[0.1, 0.2], [0.2, 0.3]])

    with pytest.raises(DataShapeError, match="length of test bins"):
        regressor.predict_probability(x, y_threshold=0.5, bins=np.array([0]))




def test_insert_calibration_requires_bins_when_existing_none(monkeypatch):
    regressor = make_regressor(monkeypatch)
    xs = np.array([[0.1, 0.2], [0.2, 0.3]])
    ys = np.array([0.5, 0.6])

    with pytest.raises(
        ConfigurationError, match="Cannot mix calibration instances with and without bins"
    ):
        regressor.insert_calibration(xs, ys, bins=np.array([0, 1]))


def test_insert_calibration_validates_bin_length(monkeypatch):
    base_bins = np.zeros(4, dtype=int)
    regressor = make_regressor(monkeypatch, bins=base_bins)
    xs = np.array([[0.1, 0.2], [0.2, 0.3]])
    ys = np.array([0.5, 0.6])

    with pytest.raises(DataShapeError, match="length of bins"):
        regressor.insert_calibration(xs, ys, bins=np.array([0]))










def test_insert_calibration_updates_predictor_state(monkeypatch):
    base_bins = np.array([0, 1, 0, 1])
    regressor = make_regressor(monkeypatch, bins=base_bins)
    updates = [
        (np.array([[0.9, 0.2]]), np.array([1.4]), np.array([0])),
        (np.array([[0.1, 0.8]]), np.array([1.2]), np.array([1])),
    ]

    # Use split["cps"] because insert_calibration updates the split CPS, not the main one.
    # Also use binned_alphas because that's what is updated.
    cps = regressor.split["cps"]
    expected_bins0 = np.array(cps.binned_alphas[1][0], copy=True)
    expected_bins1 = np.array(cps.binned_alphas[1][1], copy=True)
    appended_bins: list[int] = []

    for i, (xs, ys, new_bins) in enumerate(updates):
        residual = ys - regressor.ce.predict_function(xs)
        regressor.insert_calibration(xs, ys, bins=new_bins)
        appended_bins.extend(new_bins.tolist())
        # First update goes to CPS (part 0), second goes to VA (part 1) due to split balancing
        if i == 0:
            if new_bins[0] == 0:
                expected_bins0 = np.sort(np.concatenate([expected_bins0, residual]))
            else:
                expected_bins1 = np.sort(np.concatenate([expected_bins1, residual]))

    assert regressor.bins.shape[0] == base_bins.shape[0] + len(appended_bins)
    assert np.all(regressor.bins[-len(appended_bins) :] == np.array(appended_bins))
    assert np.allclose(cps.binned_alphas[1][0], expected_bins0)
    assert np.allclose(cps.binned_alphas[1][1], expected_bins1)


def test_insert_calibration_updates_unbinned_alphas(monkeypatch):
    regressor = make_regressor(monkeypatch)
    split_before = np.array(regressor.split["cps"].alphas, copy=True)
    cps_before = np.array(regressor.cps.alphas, copy=True)
    xs = np.array([[0.0, 0.0], [0.5, 0.5]])
    ys = np.array([0.2, 0.1])
    residuals = ys - regressor.ce.predict_function(xs)

    regressor.insert_calibration(xs, ys)

    expected_split = np.sort(np.concatenate([split_before, residuals[[0]]]))
    expected_cps = np.sort(np.concatenate([cps_before, residuals]))
    assert np.allclose(regressor.split["cps"].alphas, expected_split)
    assert np.allclose(regressor.cps.alphas, expected_cps)


def test_insert_calibration_updates_legacy_binned_alpha_storage(monkeypatch):
    regressor = make_regressor(monkeypatch, bins=np.array([0, 1, 0, 1]))
    split_cps = regressor.split["cps"]
    full_cps = regressor.cps
    split_map = split_cps.alphas[1]
    full_map = full_cps.alphas[1]
    split_before = {k: np.array(v, copy=True) for k, v in split_map.items()}
    full_before = {k: np.array(v, copy=True) for k, v in full_map.items()}
    split_cps.alphas = (
        np.array([0, 1]),
        [np.array(split_before[0], copy=True), np.array(split_before[1], copy=True)],
    )
    full_cps.alphas = (
        np.array([0, 1]),
        [np.array(full_before[0], copy=True), np.array(full_before[1], copy=True)],
    )
    delattr(split_cps, "binned_alphas")
    delattr(full_cps, "binned_alphas")
    xs = np.array([[0.2, 0.1], [0.4, 0.0]])
    ys = np.array([0.8, 0.6])
    bins = np.array([0, 99])
    residuals = ys - regressor.ce.predict_function(xs)

    regressor.insert_calibration(xs, ys, bins=bins)

    expected_split_bin0 = np.sort(np.concatenate([split_before[0], residuals[[0]]]))
    expected_full_bin0 = np.sort(np.concatenate([full_before[0], residuals[bins == 0]]))
    assert np.allclose(regressor.split["cps"].alphas[1][0], expected_split_bin0)
    assert np.allclose(regressor.cps.alphas[1][0], expected_full_bin0)
    assert np.allclose(regressor.cps.alphas[1][1], full_before[1])


def test_append_helpers_ignore_empty_inputs(monkeypatch):
    """Empty calibration inserts should not mutate internal buffers."""

    regressor = make_regressor(monkeypatch, bins=np.array([0, 1, 0, 1]))

    original_y_hat = np.array(regressor.y_cal_hat_storage, copy=True)
    original_y_hat_size = regressor.y_cal_hat_size
    regressor.append_calibration_buffer("y_cal_hat", np.array([]))

    assert regressor.y_cal_hat_size == original_y_hat_size
    assert np.array_equal(
        regressor.y_cal_hat_storage[:original_y_hat_size], original_y_hat[:original_y_hat_size]
    )

    original_bins = np.array(regressor.bins_storage, copy=True)
    original_bins_size = regressor.bins_size
    regressor.append_bins(np.array([]))

    assert regressor.bins_size == original_bins_size
    assert np.array_equal(regressor.bins_storage[:original_bins_size], original_bins)


def test_append_helpers_expand_capacity_and_normalize_shapes(monkeypatch):
    regressor = make_regressor(monkeypatch)

    appended_calibration = np.array([[9.0], [8.0], [7.0], [6.0], [5.0]])
    regressor.append_calibration_buffer("y_cal_hat", appended_calibration)

    assert regressor.y_cal_hat_size == 4 + appended_calibration.shape[0]
    assert np.allclose(regressor.y_cal_hat_storage[:4], np.array([0.15, 0.25, 0.35, 0.45]))
    assert np.allclose(regressor.y_cal_hat_storage[4:9], appended_calibration.reshape(-1))

    regressor.append_bins(np.array([[2], [3]]))
    regressor.append_bins(np.array([[4], [5], [6]]))

    assert regressor.bins_size == 5
    assert np.array_equal(regressor.bins_storage[:5], np.array([2, 3, 4, 5, 6]))








def test_compute_proba_cal_invalid_type(monkeypatch):
    from calibrated_explanations.utils.exceptions import ValidationError

    regressor = make_regressor(monkeypatch)

    with pytest.raises(ValidationError, match="y_threshold must be a float or a tuple"):
        regressor.compute_proba_cal([0.1, 0.2])




def test_init_flattens_calibration_arrays(monkeypatch):
    class ColumnExplainer(DummyExplainer):
        def __init__(self):
            super().__init__()
            self.y_cal = np.array([[0.1], [0.2], [0.3], [0.4]])

        def predict_calibration(self):
            return self.y_cal + 0.05

        def get_sigma_test(self, x):  # pylint: disable=unused-argument
            return np.ones((len(x), 1))

    monkeypatch.setattr(interval_module.crepes, "ConformalPredictiveSystem", DummyCPS)
    monkeypatch.setattr(interval_module, "VennAbers", DummyVennAbers)

    regressor = interval_module.IntervalRegressor(ColumnExplainer())

    assert regressor.y_cal_hat_storage.ndim == 1
    assert regressor.residual_cal_storage.ndim == 1
    assert regressor.sigma_cal_storage.ndim == 1




def test_ensure_capacity_copies_existing_prefix(monkeypatch):
    regressor = make_regressor(monkeypatch)
    original = np.array([5], dtype=float)

    grown = regressor.ensure_capacity(original, 1, 2)

    assert grown.shape[0] >= 3
    assert grown[0] == original[0]




