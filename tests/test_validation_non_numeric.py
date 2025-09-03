from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from calibrated_explanations.core.validation import validate_feature_matrix
from calibrated_explanations.core.exceptions import ValidationError, DataShapeError
from calibrated_explanations.core.wrap_explainer import WrapCalibratedExplainer
from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer


def test_validate_feature_matrix_dataframe_non_numeric_columns():
    df = pd.DataFrame({"num": [1.0, 2.0, 3.0], "txt": ["a", "b", "c"]})
    with pytest.raises(ValidationError) as exc:
        validate_feature_matrix(df, name="X")
    assert "non-numeric columns" in str(exc.value)


def test_validate_feature_matrix_numpy_object_dtype():
    X = np.array([[1, "a"], [2, "b"]], dtype=object)
    with pytest.raises(DataShapeError) as exc:
        validate_feature_matrix(X, name="X")
    assert "non-numeric dtype" in str(exc.value)


def test_wrap_fit_raises_on_non_numeric_dataframe():
    # Learner need not be fitted for WrapCalibratedExplainer.fit; validation happens before fit
    learner = LogisticRegression()
    wrapper = WrapCalibratedExplainer(learner)
    X = pd.DataFrame({"num": [1.0, 2.0, 3.0], "txt": ["a", "b", "c"]})
    y = np.array([0, 1, 0])
    with pytest.raises(ValidationError):
        wrapper.fit(X, y)


def test_calibrated_explainer_init_raises_on_non_numeric_dataframe():
    # CalibratedExplainer requires a fitted learner
    X_train = np.array([[0.0], [1.0], [2.0]])
    y_train = np.array([0, 1, 1])
    learner = LogisticRegression().fit(X_train, y_train)

    # Non-numeric calibration features should be rejected early with actionable error
    X_cal = pd.DataFrame({"num": [1.0, 2.0, 3.0], "cat": ["x", "y", "z"]})
    y_cal = np.array([0, 1, 0])
    with pytest.raises(ValidationError):
        CalibratedExplainer(learner, X_cal, y_cal, mode="classification")

