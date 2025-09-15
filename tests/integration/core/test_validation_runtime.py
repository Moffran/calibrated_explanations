import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
from sklearn.ensemble import HistGradientBoostingClassifier

from calibrated_explanations.core.wrap_explainer import WrapCalibratedExplainer
from calibrated_explanations.core.exceptions import ValidationError


def test_calibrate_rejects_nans_in_calibration():
    # small clean proper training set
    X_prop = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [0.2, 0.8]])
    y_prop = np.array([0, 1, 1, 0])
    # calibration with a NaN in X
    X_cal = np.array([[0.1, np.nan], [0.9, 0.1]])
    y_cal = np.array([0, 1])

    w = WrapCalibratedExplainer(RandomForestClassifier(n_estimators=5, random_state=0))
    w.fit(X_prop, y_prop)

    with pytest.raises(ValidationError):
        w.calibrate(X_cal, y_cal)

    # calibration with a NaN in y
    X_cal2 = np.array([[0.1, 0.2], [0.9, 0.1]])
    y_cal2 = np.array([0, np.nan])
    with pytest.raises(ValidationError):
        w.calibrate(X_cal2, y_cal2)


def test_predict_allows_nans_in_X_test_for_nan_tolerant_model():
    # HistGradientBoostingClassifier supports NaNs in X
    rng = np.random.RandomState(0)
    X_prop = rng.rand(20, 3)
    y_prop = (X_prop[:, 0] > 0.5).astype(int)
    X_cal = rng.rand(10, 3)
    y_cal = (X_cal[:, 0] > 0.5).astype(int)

    w = WrapCalibratedExplainer(HistGradientBoostingClassifier(random_state=0))
    w.fit(X_prop, y_prop)
    w.calibrate(X_cal, y_cal)

    # Create test with NaNs
    X_test = rng.rand(5, 3)
    X_test[0, 1] = np.nan
    X_test[3, 2] = np.nan

    # Should not raise ValidationError at wrapper boundary; model can handle NaNs
    preds = w.predict(X_test)
    proba = w.predict_proba(X_test)

    assert preds.shape[0] == X_test.shape[0]
    # proba is (n, 2) for binary classification
    assert proba.shape[0] == X_test.shape[0]
