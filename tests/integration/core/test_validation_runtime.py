import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

from calibrated_explanations.core import WrapCalibratedExplainer, ValidationError


def test_calibrate_rejects_nans_in_calibration():
    # small clean proper training set
    x_prop = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [0.2, 0.8]])
    y_prop = np.array([0, 1, 1, 0])
    # calibration with a NaN in x
    x_cal = np.array([[0.1, np.nan], [0.9, 0.1]])
    y_cal = np.array([0, 1])

    w = WrapCalibratedExplainer(RandomForestClassifier(n_estimators=5, random_state=0))
    w.fit(x_prop, y_prop)

    with pytest.raises(ValidationError):
        w.calibrate(x_cal, y_cal)

    # calibration with a NaN in y
    x_cal2 = np.array([[0.1, 0.2], [0.9, 0.1]])
    y_cal2 = np.array([0, np.nan])
    with pytest.raises(ValidationError):
        w.calibrate(x_cal2, y_cal2)


# @pytest.mark.skipif(
#     sys.platform == "win32",
#     reason="HistGradientBoostingClassifier.fit hangs indefinitely on Windows; upstream sklearn bug",
# )
@pytest.mark.slow
def test_predict_allows_nans_in_x_test_for_nan_tolerant_model():
    # HistGradientBoostingClassifier supports NaNs in x
    rng = np.random.RandomState(0)
    x_prop = rng.rand(20, 3)
    y_prop = (x_prop[:, 0] > 0.5).astype(int)
    x_cal = rng.rand(10, 3)
    y_cal = (x_cal[:, 0] > 0.5).astype(int)

    w = WrapCalibratedExplainer(HistGradientBoostingClassifier(random_state=0))
    w.fit(x_prop, y_prop)
    w.calibrate(x_cal, y_cal)

    # Create test with NaNs
    x_test = rng.rand(5, 3)
    x_test[0, 1] = np.nan
    x_test[3, 2] = np.nan

    # Should not raise ValidationError at wrapper boundary; model can handle NaNs
    preds = w.predict(x_test)
    proba = w.predict_proba(x_test)

    assert preds.shape[0] == x_test.shape[0]
    # proba is (n, 2) for binary classification
    assert proba.shape[0] == x_test.shape[0]
