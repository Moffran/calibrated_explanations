import numpy as np
import pytest

from calibrated_explanations.core.validation import (
    infer_task,
    validate_fit_state,
    validate_inputs_matrix,
    validate_model,
)
from calibrated_explanations.core.exceptions import (
    ValidationError,
    DataShapeError,
    ModelNotSupportedError,
    NotFittedError,
)


class _DummyReg:
    def predict(self, X):  # noqa: D401
        return np.zeros((len(X),), dtype=float)


class _DummyCls(_DummyReg):
    def predict_proba(self, X):
        return np.tile([0.5, 0.5], (len(X), 1))


def test_infer_task_prefers_model_capabilities():
    assert infer_task(model=_DummyCls()) == "classification"
    assert infer_task(model=_DummyReg()) == "regression"
    # y-based inference fallback
    assert infer_task(y=np.array([0, 1, 0])) == "classification"
    assert infer_task(y=np.array([0.1, 0.2])) == "regression"


def test_validate_model_and_fit_state():
    class _NoPredict:
        pass

    with pytest.raises(ModelNotSupportedError):
        validate_model(_NoPredict())

    # fit state
    class _Obj:
        fitted = False

    with pytest.raises(NotFittedError):
        validate_fit_state(_Obj(), require=True)

    # no exception when fitted
    _Obj.fitted = True
    validate_fit_state(_Obj(), require=True)


def test_validate_inputs_matrix_shapes_and_finite():
    X = np.ones((3, 2), dtype=float)
    y = np.array([1.0, 2.0, 3.0])
    # happy path
    validate_inputs_matrix(X, y, require_y=True)
    # shape mismatch (y length)
    with pytest.raises(DataShapeError, match=r"Length of 'y' \("):
        validate_inputs_matrix(X, y[:2], require_y=True)
    # not 2D
    with pytest.raises(DataShapeError, match="Argument 'X' must be 2D"):
        validate_inputs_matrix(np.ones((3,)), y)
    # n_features mismatch
    with pytest.raises(DataShapeError, match="Argument 'X' must have 3 features"):
        validate_inputs_matrix(X, y, n_features=3)
    # NaN check
    X_bad = X.copy()
    X_bad[0, 0] = np.nan
    with pytest.raises(ValidationError):
        validate_inputs_matrix(X_bad, y)
    # allow NaN
    validate_inputs_matrix(X_bad, y, allow_nan=True)
