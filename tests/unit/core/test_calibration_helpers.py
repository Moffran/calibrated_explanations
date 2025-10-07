import numpy as np
from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer
from calibrated_explanations.core import calibration_helpers as ch
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


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
