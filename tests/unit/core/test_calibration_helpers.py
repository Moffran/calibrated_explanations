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
from calibrated_explanations.utils.exceptions import ConfigurationError


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
