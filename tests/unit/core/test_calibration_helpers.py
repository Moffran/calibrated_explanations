"""Tests for calibration interval learner management functions.

Tests verify the behavior of interval learner initialization, updates, and
threshold assignment for both standard and fast explanation modes.
"""

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from calibrated_explanations.calibration import interval_learner as ch
from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer
from calibrated_explanations.core.exceptions import ConfigurationError


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


def test_fast_calibration_helper_delegates(monkeypatch):
    data = load_iris()
    x_train, x_cal, y_train, y_cal = train_test_split(
        data.data, data.target, test_size=0.2, random_state=7, stratify=data.target
    )
    clf = RandomForestClassifier(n_estimators=5, random_state=7, max_depth=2)
    clf.fit(x_train, y_train)

    captured_calls = []

    def fake_obtain(self, *, fast, metadata):  # noqa: D401 - short test stub
        captured_calls.append({"fast": fast, "metadata": dict(metadata)})
        return ["fast-cal-1", "fast-cal-2"], "tests.interval.fake"

    monkeypatch.setattr(CalibratedExplainer, "_obtain_interval_calibrator", fake_obtain)

    explainer = CalibratedExplainer(
        clf,
        x_cal,
        y_cal,
        mode="classification",
        seed=7,
        fast=True,
    )

    assert explainer.interval_learner == ["fast-cal-1", "fast-cal-2"]
    assert captured_calls and captured_calls[0]["fast"] is True
    assert captured_calls[0]["metadata"].get("operation") == "initialize_fast"

    ch.initialize_interval_learner_for_fast_explainer(explainer)

    assert captured_calls[-1]["fast"] is True
    assert captured_calls[-1]["metadata"].get("operation") == "initialize_fast"


class _BaseStubExplainer:
    """Utility stub exposing the minimal CalibratedExplainer surface for tests."""

    mode: str

    def __init__(self, *, mode: str):
        self.mode = mode
        self._CalibratedExplainer__initialized = False  # match real attribute name

    def is_fast(self) -> bool:  # pragma: no cover - concrete subclasses override
        raise NotImplementedError


def test_update_interval_learner__should_reject_fast_mode_updates():
    class FastExplainer(_BaseStubExplainer):
        def __init__(self):
            super().__init__(mode="classification")

        def is_fast(self) -> bool:
            return True

    explainer = FastExplainer()

    with pytest.raises(ConfigurationError):
        ch.update_interval_learner(explainer, xs=[1, 2, 3], ys=[1, 1, 2])


def test_update_interval_learner__should_raise_when_regression_interval_is_list():
    class RegressionListExplainer(_BaseStubExplainer):
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

    class RegressionExplainer(_BaseStubExplainer):
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
    assert explainer._CalibratedExplainer__initialized is True


def test_calibration_helpers_deprecation_and_delegate(monkeypatch):
    """Accessing names from the deprecated `calibration_helpers` module should
    emit a DeprecationWarning and delegate to the interval_learner implementation.

    The test avoids importing the real `interval_learner` implementation (which
    pulls in heavy runtime dependencies) by injecting a lightweight fake module
    into `sys.modules` before attribute access.
    """
    import warnings
    import sys
    import types

    from calibrated_explanations.core import calibration_helpers as ch_helpers

    # Create a fake calibration package + interval_learner submodule
    # Note: ADR-001 - calibration is extracted to top-level package
    fake_interval = types.ModuleType("calibrated_explanations.calibration.interval_learner")

    def fake_assign_threshold(explainer, t):
        return "ok"

    fake_interval.assign_threshold = fake_assign_threshold

    fake_pkg = types.ModuleType("calibrated_explanations.calibration")
    fake_pkg.interval_learner = fake_interval

    monkeypatch.setitem(sys.modules, "calibrated_explanations.calibration", fake_pkg)
    monkeypatch.setitem(
        sys.modules, "calibrated_explanations.calibration.interval_learner", fake_interval
    )

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        func = ch_helpers.assign_threshold

    assert any(
        issubclass(w.category, DeprecationWarning) for w in rec
    ), "expected DeprecationWarning"

    # Calling the delegated function should return the fake result
    res = func(object(), 0.5)
    assert res == "ok"


def test_calibration_helpers_unknown_attribute_raises():
    from calibrated_explanations.core import calibration_helpers as ch_helpers

    with pytest.raises(AttributeError):
        _ = ch_helpers.this_attribute_does_not_exist
