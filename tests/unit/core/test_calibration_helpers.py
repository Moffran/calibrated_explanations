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
