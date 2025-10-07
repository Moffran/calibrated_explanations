import numpy as np
import pytest

from calibrated_explanations.core import validation


def test_infer_task_from_model_like():
    class M:
        def predict(self, x):
            return x

    class C(M):
        def predict_proba(self, x):
            return x

    assert validation.infer_task(model=M()) == "regression"
    assert validation.infer_task(model=C()) == "classification"


@pytest.mark.parametrize(
    "labels, expected",
    [
        (np.array([0, 1, 0]), "classification"),
        (np.array([1.2, 3.4, 5.6]), "regression"),
    ],
)
def test_infer_task_from_labels(labels, expected):
    assert validation.infer_task(y=labels) == expected


def test_validate_not_none_and_non_empty_errors():
    with pytest.raises(validation.ValidationError):
        validation.validate_not_none(None, "param")

    with pytest.raises(validation.ValidationError):
        validation.validate_non_empty([], "param")

    with pytest.raises(validation.ValidationError):
        validation.validate_inputs("", arg1="value")


def test_validate_type_errors():
    with pytest.raises(validation.DataShapeError):
        validation.validate_type("a", int, "param")


def test_validate_inputs_matrix_shape_checks():
    with pytest.raises(validation.DataShapeError):
        validation.validate_inputs_matrix([1, 2, 3])

    with pytest.raises(validation.ValidationError):
        validation.validate_inputs_matrix(np.zeros((2, 2)), require_y=True)

    with pytest.raises(validation.DataShapeError):
        validation.validate_inputs_matrix(np.zeros((2, 2)), y=np.arange(3))

    with pytest.raises(validation.DataShapeError):
        validation.validate_inputs_matrix(np.zeros((2, 2)), n_features=3)


def test_validate_inputs_matrix_finite_checks():
    x = np.array([[1.0, np.nan]])
    with pytest.raises(validation.ValidationError):
        validation.validate_inputs_matrix(x)

    y = np.array([1.0, np.nan])
    with pytest.raises(validation.ValidationError):
        validation.validate_inputs_matrix(np.ones((2, 1)), y=y)

    # Allowing NaN should silence the finiteness guard
    validation.validate_inputs_matrix(x, allow_nan=True)
    validation.validate_inputs_matrix(np.ones((2, 1)), y=y, allow_nan=True)


def test_validate_model_and_fit_state_errors():
    with pytest.raises(validation.ModelNotSupportedError):
        validation.validate_model(object())

    class Dummy:
        fitted = False

    with pytest.raises(validation.NotFittedError):
        validation.validate_fit_state(Dummy())

    # Non-required check should not raise even if attribute present
    validation.validate_fit_state(Dummy(), require=False)
