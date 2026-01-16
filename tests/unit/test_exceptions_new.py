"""Unit tests for exceptions module."""

from calibrated_explanations.utils.exceptions import CalibratedError, ValidationError


def test_calibrated_error_repr():
    """Test CalibratedError __repr__."""
    err = CalibratedError("test message")
    repr_str = repr(err)
    assert "CalibratedError" in repr_str
    assert "test message" in repr_str


def test_validation_error():
    """Test ValidationError."""
    err = ValidationError("validation failed")
    assert str(err) == "validation failed"
    assert isinstance(err, CalibratedError)
