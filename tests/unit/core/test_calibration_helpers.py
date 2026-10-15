"""Tests for calibration interval learner management functions.

Tests verify the behavior of interval learner initialization, updates, and
threshold assignment for both standard and fast explanation modes.
"""

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split

from calibrated_explanations import WrapCalibratedExplainer
from calibrated_explanations.calibration import interval_learner as ch
from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer
from calibrated_explanations.utils.exceptions import (
    ConfigurationError,
    DataShapeError,
    ValidationError,
)


def test_calibration_helpers_round_trip():
    data = load_iris()
    x_train, x_cal, y_train, y_cal = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )
    clf = RandomForestClassifier(n_estimators=10, random_state=42, max_depth=3)
    clf.fit(x_train, y_train)
    explainer = CalibratedExplainer(clf, x_cal, y_cal, mode="classification", seed=42)

    # assign_threshold wrapper behavior parity
    assert ch.assign_threshold(explainer, None) is None
    assert isinstance(ch.assign_threshold(explainer, [0.1, 0.2]), np.ndarray)

    # initialize_interval_learner wrapper should be callable
    ch.initialize_interval_learner(explainer)
    assert explainer.interval_learner is not None

    # update_interval_learner wrapper should accept new data
    x_new, y_new = x_cal[:5], y_cal[:5]
    ch.update_interval_learner(explainer, x_new, y_new, bins=None)
    assert explainer.interval_learner is not None


class BaseStubExplainer:
    """Utility stub exposing the minimal CalibratedExplainer surface for tests."""

    mode: str

    def __init__(self, *, mode: str):
        self.mode = mode
        self.initialized_state = False  # internal storage for the `initialized` property

    @property
    def initialized(self):
        return self.initialized_state

    @initialized.setter
    def initialized(self, value):
        self.initialized_state = value

    def is_fast(self) -> bool:  # pragma: no cover - concrete subclasses override
        raise NotImplementedError


def test_update_interval_learner__should_reject_fast_mode_updates():
    class FastExplainer(BaseStubExplainer):
        def __init__(self):
            super().__init__(mode="classification")

        def is_fast(self) -> bool:
            return True

    explainer = FastExplainer()

    with pytest.raises(ConfigurationError):
        ch.update_interval_learner(explainer, xs=[1, 2, 3], ys=[1, 1, 2])


def test_update_interval_learner__should_raise_when_regression_interval_is_list():
    class RegressionListExplainer(BaseStubExplainer):
        def __init__(self):
            super().__init__(mode="regression")
            self.interval_learner = []  # fast-mode sentinel from runtime helpers

        def is_fast(self) -> bool:
            return False

    explainer = RegressionListExplainer()

    with pytest.raises(ConfigurationError):
        ch.update_interval_learner(explainer, xs=[1.0], ys=[0.0])


def test_update_interval_learner__should_insert_calibration_for_regression_interval():
    class TrackingInterval:
        def __init__(self):
            self.calls: list[tuple[tuple[float, ...], tuple[float, ...], object]] = []

        def insert_calibration(self, xs, ys, bins=None):
            self.calls.append((tuple(xs), tuple(ys), bins))

    class RegressionExplainer(BaseStubExplainer):
        def __init__(self):
            super().__init__(mode="regression")
            self.interval_learner = TrackingInterval()

        def is_fast(self) -> bool:
            return False

    explainer = RegressionExplainer()

    ch.update_interval_learner(
        explainer,
        xs=[1.0, 2.0, 3.0],
        ys=[0.1, 0.2, 0.3],
        bins={"count": 5},
    )

    assert explainer.interval_learner.calls == [((1.0, 2.0, 3.0), (0.1, 0.2, 0.3), {"count": 5})]
    assert explainer.initialized is True


def test_should_raise_attribute_error_when_deprecated_calibration_helper_accessed() -> None:
    from calibrated_explanations.core import calibration_helpers as ch_helpers

    for name in (
        "assign_threshold",
        "initialize_interval_learner",
        "initialize_interval_learner_for_fast_explainer",
        "update_interval_learner",
    ):
        with pytest.raises(AttributeError):
            getattr(ch_helpers, name)


# --- reinitialize: calibration state must follow the new data / learner -------------


def _reinitialize_data(mode: str) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    x = rng.random((200, 4))
    if mode == "classification":
        y = (x[:, 0] + x[:, 1] > 1.0).astype(int)
        shifted = 1 - y  # inverted labels: any refresh must move the outputs
    else:
        y = 3.0 * x[:, 0] - x[:, 1] + rng.normal(0.0, 0.05, size=x.shape[0])
        shifted = y + 2.0
    bins = (x[:, 2] > 0.5).astype(int)
    return {
        "x_prop": x[:60], "y_prop": y[:60],
        "x_cal": x[60:120], "y_cal": y[60:120], "bins_cal": bins[60:120],
        "x_new": x[120:160], "y_new": shifted[120:160], "bins_new": bins[120:160],
        "x_test": x[160:], "bins_test": bins[160:],
    }  # fmt: skip


def _fitted_learner(mode: str, data: dict[str, np.ndarray]):
    learner = (
        RandomForestClassifier(n_estimators=10, random_state=0)
        if mode == "classification"
        else RandomForestRegressor(n_estimators=10, random_state=0)
    )
    return learner.fit(data["x_prop"], data["y_prop"])


def _calibrated(learner, mode: str, x_cal, y_cal, bins=None) -> WrapCalibratedExplainer:
    wrapper = WrapCalibratedExplainer(learner)
    wrapper.calibrate(x_cal, y_cal, mode=mode, bins=bins, seed=42)
    return wrapper


def _outputs(wrapper: WrapCalibratedExplainer, mode: str, x, bins=None) -> np.ndarray:
    if mode == "classification":
        proba, (low, high) = wrapper.predict_proba(x, uq_interval=True, bins=bins)
        return np.concatenate([np.ravel(proba), np.ravel(low), np.ravel(high)])
    point, (low, high) = wrapper.predict(x, uq_interval=True, bins=bins)
    return np.concatenate([np.ravel(point), np.ravel(low), np.ravel(high)])


@pytest.mark.parametrize("mondrian", [False, True], ids=["standard", "mondrian"])
@pytest.mark.parametrize("mode", ["classification", "regression"])
def test_should_reflect_appended_calibration_data_when_reinitialize_appends(
    mode: str, mondrian: bool
) -> None:
    """Classification rebuilds on the combined set; regression extends the existing split.

    Regression appends keep the existing conformal split (``insert_calibration``), so only
    the point predictions, not the intervals, equal a fresh calibration on the combined set.
    """
    data = _reinitialize_data(mode)
    learner = _fitted_learner(mode, data)
    bins_cal = data["bins_cal"] if mondrian else None
    bins_new = data["bins_new"] if mondrian else None
    bins_test = data["bins_test"] if mondrian else None
    wrapper = _calibrated(learner, mode, data["x_cal"], data["y_cal"], bins_cal)
    before = _outputs(wrapper, mode, data["x_test"], bins_test)

    wrapper.explainer.reinitialize(learner, data["x_new"], data["y_new"], bins=bins_new)

    after = _outputs(wrapper, mode, data["x_test"], bins_test)
    fresh = _calibrated(
        learner,
        mode,
        np.vstack([data["x_cal"], data["x_new"]]),
        np.concatenate([data["y_cal"], data["y_new"]]),
        None if bins_cal is None else np.concatenate([bins_cal, bins_new]),
    )
    expected = _outputs(fresh, mode, data["x_test"], bins_test)
    assert len(wrapper.explainer.y_cal) == len(data["y_cal"]) + len(data["y_new"])
    assert not np.array_equal(after, before)
    if mode == "classification":
        np.testing.assert_array_equal(after, expected)
    else:
        n_test = len(data["x_test"])
        np.testing.assert_array_equal(after[:n_test], expected[:n_test])


@pytest.mark.parametrize("mode", ["classification", "regression"])
def test_should_match_fresh_calibration_when_reinitialize_swaps_learner(mode: str) -> None:
    data = _reinitialize_data(mode)
    wrapper = _calibrated(_fitted_learner(mode, data), mode, data["x_cal"], data["y_cal"])
    before = _outputs(wrapper, mode, data["x_test"])
    replacement = (LogisticRegression() if mode == "classification" else Ridge()).fit(
        data["x_new"], data["y_new"]
    )

    wrapper.explainer.reinitialize(replacement)

    after = _outputs(wrapper, mode, data["x_test"])
    fresh = _calibrated(replacement, mode, data["x_cal"], data["y_cal"])
    assert not np.array_equal(after, before)
    np.testing.assert_array_equal(after, _outputs(fresh, mode, data["x_test"]))


def test_should_match_fresh_calibration_when_wrapper_is_refit_after_calibration() -> None:
    data = _reinitialize_data("classification")
    wrapper = WrapCalibratedExplainer(LogisticRegression())
    wrapper.fit(data["x_prop"], data["y_prop"])
    wrapper.calibrate(data["x_cal"], data["y_cal"], seed=42)

    wrapper.fit(data["x_new"], data["y_new"])

    fresh = WrapCalibratedExplainer(LogisticRegression())
    fresh.fit(data["x_new"], data["y_new"])
    fresh.calibrate(data["x_cal"], data["y_cal"], seed=42)
    np.testing.assert_array_equal(
        _outputs(wrapper, "classification", data["x_test"]),
        _outputs(fresh, "classification", data["x_test"]),
    )


def test_should_not_serve_cached_predictions_when_reinitialize_changes_calibration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CE_CACHE", "1")
    data = _reinitialize_data("classification")
    learner = _fitted_learner("classification", data)
    wrapper = _calibrated(learner, "classification", data["x_cal"], data["y_cal"])
    assert wrapper.explainer.perf_cache is not None and wrapper.explainer.perf_cache.enabled
    _outputs(wrapper, "classification", data["x_test"])  # populate the cache

    wrapper.explainer.reinitialize(learner, data["x_new"], data["y_new"])

    fresh = _calibrated(
        learner,
        "classification",
        np.vstack([data["x_cal"], data["x_new"]]),
        np.concatenate([data["y_cal"], data["y_new"]]),
    )
    np.testing.assert_array_equal(
        _outputs(wrapper, "classification", data["x_test"]),
        _outputs(fresh, "classification", data["x_test"]),
    )


def _fail(*_args, **_kwargs):
    raise ConfigurationError("calibrator construction failed")


@pytest.mark.parametrize(
    ("mode", "appending", "target"),
    [
        ("classification", True, "obtain_interval_calibrator"),
        ("classification", False, "obtain_interval_calibrator"),
        ("regression", True, "refresh_interval_calibration_snapshot"),
    ],
    ids=["classification-append", "classification-new-learner", "regression-append"],
)
def test_should_restore_previous_state_when_reinitialize_fails(
    mode: str, appending: bool, target: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _reinitialize_data(mode)
    learner = _fitted_learner(mode, data)
    wrapper = _calibrated(learner, mode, data["x_cal"], data["y_cal"])
    explainer = wrapper.explainer
    before = _outputs(wrapper, mode, data["x_test"])
    calibrator = explainer.interval_learner
    owner = (
        explainer if target == "obtain_interval_calibrator" else explainer.prediction_orchestrator
    )
    monkeypatch.setattr(owner, target, _fail)
    args = (data["x_new"], data["y_new"]) if appending else ()

    with pytest.raises(ConfigurationError, match="calibrator construction failed"):
        explainer.reinitialize(learner, *args)

    monkeypatch.undo()
    assert len(explainer.y_cal) == len(data["y_cal"])
    assert explainer.learner is learner
    cached = explainer.plugin_manager.interval_context_metadata["default"]["calibrator"]
    assert cached is explainer.interval_learner
    if mode == "classification":
        assert explainer.interval_learner is calibrator
    np.testing.assert_array_equal(_outputs(wrapper, mode, data["x_test"]), before)


def test_should_leave_calibration_set_unchanged_when_appended_bins_have_wrong_length() -> None:
    data = _reinitialize_data("classification")
    learner = _fitted_learner("classification", data)
    wrapper = _calibrated(learner, "classification", data["x_cal"], data["y_cal"], data["bins_cal"])

    with pytest.raises(DataShapeError):
        wrapper.explainer.reinitialize(
            learner, data["x_new"], data["y_new"], bins=data["bins_new"][:-1]
        )

    assert len(wrapper.explainer.y_cal) == len(data["y_cal"])
    assert len(wrapper.explainer.bins) == len(data["bins_cal"])


# --- set_mode: the calibrator must follow the new mode (#241) ------------------------


@pytest.mark.parametrize(
    ("start", "target"),
    [("classification", "regression"), ("regression", "classification")],
)
def test_should_match_fresh_calibration_when_set_mode_switches_mode(
    start: str, target: str
) -> None:
    data = _reinitialize_data("classification")
    learner = _fitted_learner("classification", data)
    wrapper = _calibrated(learner, start, data["x_cal"], data["y_cal"])

    wrapper.explainer.set_mode(target)

    fresh = _calibrated(learner, target, data["x_cal"], data["y_cal"])
    assert type(wrapper.explainer.interval_learner) is type(fresh.explainer.interval_learner)
    assert wrapper.explainer.predict_function == fresh.explainer.predict_function
    np.testing.assert_array_equal(
        _outputs(wrapper, target, data["x_test"]), _outputs(fresh, target, data["x_test"])
    )


def test_should_restore_previous_mode_when_set_mode_rebuild_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _reinitialize_data("classification")
    learner = _fitted_learner("classification", data)
    wrapper = _calibrated(learner, "classification", data["x_cal"], data["y_cal"])
    explainer = wrapper.explainer
    before = _outputs(wrapper, "classification", data["x_test"])
    calibrator, predict_function = explainer.interval_learner, explainer.predict_function
    monkeypatch.setattr(explainer, "obtain_interval_calibrator", _fail)

    with pytest.raises(ConfigurationError, match="calibrator construction failed"):
        explainer.set_mode("regression")

    monkeypatch.undo()
    assert explainer.mode == "classification"
    assert explainer.interval_learner is calibrator
    assert explainer.predict_function == predict_function
    np.testing.assert_array_equal(_outputs(wrapper, "classification", data["x_test"]), before)


# --- reinitialize: supported and unsupported update combinations (#208) ---------------


def _fast_calibrated(learner, data) -> WrapCalibratedExplainer:
    wrapper = _calibrated(learner, "classification", data["x_cal"], data["y_cal"])
    wrapper.explainer.enable_fast_mode()
    return wrapper


def _fast_weights(wrapper: WrapCalibratedExplainer, x) -> np.ndarray:
    explanations = wrapper.explain_fast(x)
    return np.array([np.ravel(e.feature_weights["predict"]) for e in explanations])


@pytest.mark.parametrize(
    ("mondrian", "kwargs", "error", "match"),
    [
        (False, {"xs": "x_new"}, ValidationError, "both xs and ys"),
        (False, {"ys": "y_new"}, ValidationError, "both xs and ys"),
        (False, {"bins": "bins_new"}, ValidationError, "pass xs and ys together with bins"),
        (False, {"xs": "x_new", "ys": "y_new", "bins": "bins_new"}, ValidationError, "mix"),
        (True, {"xs": "x_new", "ys": "y_new"}, ValidationError, "Mondrian calibration"),
    ],
    ids=["xs-only", "ys-only", "bins-only", "bins-without-mondrian", "mondrian-without-bins"],
)
def test_should_reject_unsupported_combination_when_reinitialize_receives_inconsistent_data(
    mondrian: bool, kwargs: dict, error: type, match: str
) -> None:
    data = _reinitialize_data("classification")
    learner = _fitted_learner("classification", data)
    wrapper = _calibrated(
        learner,
        "classification",
        data["x_cal"],
        data["y_cal"],
        data["bins_cal"] if mondrian else None,
    )
    calibrator = wrapper.explainer.interval_learner

    with pytest.raises(error, match=match):
        wrapper.explainer.reinitialize(learner, **{k: data[v] for k, v in kwargs.items()})

    assert len(wrapper.explainer.y_cal) == len(data["y_cal"])
    assert wrapper.explainer.interval_learner is calibrator


def test_should_reject_appending_when_fast_mode_is_enabled() -> None:
    data = _reinitialize_data("classification")
    learner = _fitted_learner("classification", data)
    wrapper = _fast_calibrated(learner, data)
    before = _fast_weights(wrapper, data["x_test"][:5])

    with pytest.raises(ConfigurationError, match="not supported in fast mode"):
        wrapper.explainer.reinitialize(learner, data["x_new"], data["y_new"])

    assert len(wrapper.explainer.y_cal) == len(data["y_cal"])
    np.testing.assert_array_equal(_fast_weights(wrapper, data["x_test"][:5]), before)


def test_should_match_fresh_fast_calibration_when_reinitialize_swaps_learner_in_fast_mode() -> None:
    data = _reinitialize_data("classification")
    wrapper = _fast_calibrated(_fitted_learner("classification", data), data)
    replacement = LogisticRegression().fit(data["x_new"], data["y_new"])

    wrapper.explainer.reinitialize(replacement)

    np.testing.assert_array_equal(
        _fast_weights(wrapper, data["x_test"][:5]),
        _fast_weights(_fast_calibrated(replacement, data), data["x_test"][:5]),
    )


@pytest.mark.parametrize("mode", ["classification", "regression"])
def test_should_refresh_calibration_snapshot_when_reinitialize_appends(mode: str) -> None:
    data = _reinitialize_data(mode)
    learner = _fitted_learner(mode, data)
    wrapper = _calibrated(learner, mode, data["x_cal"], data["y_cal"])

    wrapper.explainer.reinitialize(learner, data["x_new"], data["y_new"])

    snapshot = wrapper.explainer.prediction_orchestrator.get_interval_calibration_features(
        fast=False
    )
    np.testing.assert_array_equal(snapshot, wrapper.explainer.x_cal)
