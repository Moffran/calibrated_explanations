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


class DummyReg:
    def predict(self, x):  # noqa: D401
        return np.zeros((len(x),), dtype=float)


class DummyCls(DummyReg):
    def predict_proba(self, x):
        return np.tile([0.5, 0.5], (len(x), 1))


def test_infer_task_prefers_model_capabilities():
    assert infer_task(model=DummyCls()) == "classification"
    assert infer_task(model=DummyReg()) == "regression"
    # y-based inference fallback
    assert infer_task(y=np.array([0, 1, 0])) == "classification"
    assert infer_task(y=np.array([0.1, 0.2])) == "regression"


def test_validate_model_and_fit_state():
    class NoPredict:
        pass

    with pytest.raises(ModelNotSupportedError):
        validate_model(NoPredict())

    # fit state
    class Obj:
        fitted = False

    with pytest.raises(NotFittedError):
        validate_fit_state(Obj(), require=True)

    # no exception when fitted
    Obj.fitted = True
    validate_fit_state(Obj(), require=True)


def test_validate_inputs_matrix_shapes_and_finite():
    x = np.ones((3, 2), dtype=float)
    y = np.array([1.0, 2.0, 3.0])
    # happy path
    validate_inputs_matrix(x, y, require_y=True)
    # shape mismatch (y length)
    with pytest.raises(DataShapeError, match=r"Length of 'y' \("):
        validate_inputs_matrix(x, y[:2], require_y=True)
    # not 2D
    with pytest.raises(DataShapeError, match="Argument 'x' must be 2D"):
        validate_inputs_matrix(np.ones((3,)), y)
    # n_features mismatch
    with pytest.raises(DataShapeError, match="Argument 'x' must have 3 features"):
        validate_inputs_matrix(x, y, n_features=3)
    # NaN check
    x_bad = x.copy()
    x_bad[0, 0] = np.nan
    with pytest.raises(ValidationError):
        validate_inputs_matrix(x_bad, y)
    # allow NaN
    validate_inputs_matrix(x_bad, y, allow_nan=True)
