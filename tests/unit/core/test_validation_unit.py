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


def test_validate_model_and_fit_state_success():
    class Model:
        def predict(self, x):
            return x

    validation.validate_model(Model())

    class Fitted:
        fitted = True

    validation.validate_fit_state(Fitted())


def test_infer_task_defaults_to_regression_when_unknown():
    assert validation.infer_task() == "regression"


def test_validate_inputs_matrix_supports_frame_like_objects(monkeypatch):
    class FrameLike:
        def __init__(self, arr):
            self.values = arr
            self.shape = arr.shape

    class SeriesLike:
        def __init__(self, arr):
            self.values = arr

    x = FrameLike(np.arange(6).reshape(3, 2))
    y = SeriesLike(np.array([0, 1, 0]))

    recorded = {}

    def fake_infer_task(x_arg, y_arg, model):  # pragma: no cover - monkeypatched
        recorded["args"] = (x_arg, y_arg, model)
        return "classification"

    monkeypatch.setattr(validation, "infer_task", fake_infer_task)

    # No exception should be raised and the monkeypatched infer_task should run.
    validation.validate_inputs_matrix(x, y, task="auto")
    assert recorded["args"] == (x, y, None)


def test_validate_inputs_matrix_accepts_none_y_when_optional():
    validation.validate_inputs_matrix(np.zeros((2, 2)))


# ============================================================================
# ADR-002 Contract Tests: validate_inputs() Signature Compliance
# ============================================================================


def test_validate_inputs_adr002_signature_accepts_2d_array():
    """Verify validate_inputs() accepts x parameter (2D array)."""
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    validation.validate_inputs(x)  # Should not raise


def test_validate_inputs_adr002_signature_requires_2d_x():
    """Verify validate_inputs() rejects 1D x."""
    x = np.array([1.0, 2.0, 3.0])
    with pytest.raises(validation.DataShapeError) as exc_info:
        validation.validate_inputs(x)
    assert "2D" in str(exc_info.value)
    assert exc_info.value.details is not None
    assert exc_info.value.details.get("ndim") == 1


def test_validate_inputs_adr002_signature_with_y():
    """Verify validate_inputs() accepts optional y parameter."""
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([0, 1])
    validation.validate_inputs(x, y)  # Should not raise


def test_validate_inputs_adr002_signature_y_length_mismatch():
    """Verify validate_inputs() enforces y length matches x samples."""
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([0, 1, 2])  # Wrong length
    with pytest.raises(validation.DataShapeError) as exc_info:
        validation.validate_inputs(x, y)
    assert "Length of 'y'" in str(exc_info.value) or "does not match" in str(exc_info.value)
    assert exc_info.value.details is not None
    assert exc_info.value.details.get("y_length") == 3
    assert exc_info.value.details.get("x_samples") == 2


def test_validate_inputs_adr002_task_parameter():
    """Verify validate_inputs() accepts task parameter."""
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([0, 1])
    # All task types should be accepted
    validation.validate_inputs(x, y, task="auto")
    validation.validate_inputs(x, y, task="classification")
    validation.validate_inputs(x, y, task="regression")


def test_validate_inputs_adr002_allow_nan_parameter():
    """Verify validate_inputs() accepts allow_nan parameter."""
    x = np.array([[1.0, np.nan], [3.0, 4.0]])
    # Should raise when allow_nan=False (default)
    with pytest.raises(validation.ValidationError):
        validation.validate_inputs(x, allow_nan=False)
    # Should pass when allow_nan=True
    validation.validate_inputs(x, allow_nan=True)


def test_validate_inputs_adr002_require_y_parameter():
    """Verify validate_inputs() accepts require_y parameter."""
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    # Should raise when require_y=True and y=None
    with pytest.raises(validation.ValidationError) as exc_info:
        validation.validate_inputs(x, require_y=True)
    assert "y" in str(exc_info.value).lower()
    assert exc_info.value.details is not None
    # Should pass when require_y=False (default)
    validation.validate_inputs(x, require_y=False)


def test_validate_inputs_adr002_n_features_parameter():
    """Verify validate_inputs() accepts n_features parameter."""
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    # Should pass when n_features matches
    validation.validate_inputs(x, n_features=2)
    # Should raise when n_features mismatches
    with pytest.raises(validation.DataShapeError) as exc_info:
        validation.validate_inputs(x, n_features=3)
    assert exc_info.value.details is not None
    assert exc_info.value.details.get("expected_features") == 3
    assert exc_info.value.details.get("actual_features") == 2


def test_validate_inputs_adr002_class_labels_parameter():
    """Verify validate_inputs() accepts class_labels parameter."""
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([0, 1])
    labels = ["neg", "pos"]
    # Should pass with class_labels
    validation.validate_inputs(x, y, class_labels=labels)
    # Should raise if class_labels is None but passed explicitly
    # (This tests that the parameter is recognized and not just ignored)
    validation.validate_inputs(x, y, class_labels=None)


def test_validate_inputs_adr002_check_finite_parameter():
    """Verify validate_inputs() accepts check_finite parameter."""
    x = np.array([[1.0, np.inf], [3.0, 4.0]])
    # Should raise when check_finite=True and x contains inf
    with pytest.raises(validation.ValidationError) as exc_info:
        validation.validate_inputs(x, check_finite=True)
    assert exc_info.value.details is not None
    # Should pass when check_finite=False
    validation.validate_inputs(x, check_finite=False)


def test_validate_inputs_adr002_signature_details_payload():
    """Verify all exceptions include structured details payload."""
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([0, 1, 2])
    # DataShapeError should have details with diagnostic context
    with pytest.raises(validation.DataShapeError) as exc_info:
        validation.validate_inputs(x, y)
    assert exc_info.value.details is not None
    assert isinstance(exc_info.value.details, dict)
    assert "param" in exc_info.value.details


def test_validate_inputs_adr002_nan_in_y_with_details():
    """Verify validate_inputs() includes diagnostic details for NaN errors."""
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([0.0, np.nan])
    with pytest.raises(validation.ValidationError) as exc_info:
        validation.validate_inputs(x, y, allow_nan=False, check_finite=True)
    assert exc_info.value.details is not None
    assert exc_info.value.details.get("param") == "y"
    assert exc_info.value.details.get("check") == "finitude"


def test_validate_inputs_adr002_x_none_raises():
    """Verify validate_inputs() raises when x is None."""
    with pytest.raises(validation.ValidationError):
        validation.validate_inputs(None)


def test_validate_inputs_adr002_with_pandas_arrays():
    """Verify validate_inputs() handles pandas-like objects."""
    try:
        import pandas as pd

        x_df = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]])
        y_series = pd.Series([0, 1])
        # Should not raise; arrays are extracted via _as_2d_array
        validation.validate_inputs(x_df, y_series)
    except ImportError:
        pytest.skip("pandas not available")
