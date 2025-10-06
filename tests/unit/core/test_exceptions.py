from calibrated_explanations.core.exceptions import (
    CalibratedError,
    ConfigurationError,
    ConvergenceError,
    DataShapeError,
    ModelNotSupportedError,
    NotFittedError,
    SerializationError,
    ValidationError,
)


def test_exception_hierarchy_is_consistent():
    assert issubclass(ValidationError, CalibratedError)
    assert issubclass(DataShapeError, ValidationError)
    assert issubclass(ConfigurationError, CalibratedError)
    assert issubclass(ModelNotSupportedError, CalibratedError)
    assert issubclass(NotFittedError, CalibratedError)
    assert issubclass(ConvergenceError, CalibratedError)
    assert issubclass(SerializationError, CalibratedError)


def test_exceptions_carry_details_dict_and_repr():
    e = ValidationError("bad input", details={"code": "VAL_INPUT", "param": "x"})
    assert isinstance(e, Exception)
    assert e.details == {"code": "VAL_INPUT", "param": "x"}
    # repr should include class name and message, but not necessarily details
    r = repr(e)
    assert "ValidationError" in r
    assert "bad input" in r
