"""Unit tests for ADR-002 validation helpers and exception utilities."""

import pytest

from calibrated_explanations.utils.exceptions import (
    ConfigurationError,
    ValidationError,
    explain_exception,
)
from calibrated_explanations.core.validation import validate


def test_explain_exception_with_calibrated_error():
    """explain_exception should format CalibratedError with details."""
    e = ValidationError(
        "Argument 'x' must not be empty",
        details={"param": "x", "requirement": "non-empty"},
    )

    result = explain_exception(e)
    assert "ValidationError" in result
    assert "Argument 'x' must not be empty" in result
    assert "Details:" in result
    assert "param" in result




def test_explain_exception_non_calibrated_error():
    """explain_exception should handle standard exceptions."""
    e = ValueError("Standard error")
    result = explain_exception(e)
    assert result == "Standard error"






def test_validate_helper_different_exception_types():
    """validate() should work with different CalibratedError subclasses."""
    # ConfigurationError
    with pytest.raises(ConfigurationError):
        validate(False, ConfigurationError, "Config error")

    # ValidationError
    with pytest.raises(ValidationError):
        validate(False, ValidationError, "Validation error")
