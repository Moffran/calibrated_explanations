"""Tests for ADR-002 parameter guardrails and validation."""

import pytest

from calibrated_explanations.api.params import validate_param_combination
from calibrated_explanations.utils.exceptions import ConfigurationError


def test_validate_param_combination_no_conflicts():
    """validate_param_combination should pass when no conflicts exist."""
    # Should not raise
    validate_param_combination({"threshold": 0.5})
    validate_param_combination({"confidence_level": 0.9})
    validate_param_combination({})


def test_validate_param_combination_detects_mutually_exclusive():
    """validate_param_combination should detect mutually exclusive parameters."""
    # Both threshold and confidence_level provided
    with pytest.raises(ConfigurationError, match="mutually exclusive"):
        validate_param_combination(
            {
                "threshold": 0.5,
                "confidence_level": 0.9,
            }
        )


def test_validate_param_combination_allows_one_of_exclusive_group():
    """validate_param_combination should allow choosing one from exclusive group."""
    # Only threshold
    validate_param_combination({"threshold": 0.5})

    # Only confidence_level
    validate_param_combination({"confidence_level": 0.9})


def test_validate_param_combination_details_include_conflict():
    """Exception details should include information about conflicting parameters."""
    try:
        validate_param_combination(
            {
                "threshold": 0.5,
                "confidence_level": 0.9,
            }
        )
    except ConfigurationError as e:
        assert e.details is not None
        assert "conflict" in e.details
        assert "provided" in e.details
        assert "threshold" in e.details["provided"]
        assert "confidence_level" in e.details["provided"]


def test_validate_param_combination_ignores_none_values():
    """validate_param_combination should ignore None parameter values."""
    # None values should not count as conflicting
    validate_param_combination(
        {
            "threshold": 0.5,
            "confidence_level": None,
        }
    )
