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


def test_validate_inputs_matrix_errors():
    with pytest.raises(validation.DataShapeError):
        validation.validate_inputs_matrix([1, 2, 3])
    with pytest.raises(validation.DataShapeError):
        validation.validate_inputs_matrix(np.zeros((2, 2)), y=np.arange(3))
