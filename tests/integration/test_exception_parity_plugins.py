"""Regression tests validating ADR-002 exception parity in plugin flows."""

import pytest

from calibrated_explanations.utils.exceptions import ValidationError
from calibrated_explanations.plugins.base import validate_plugin_meta


def test_validate_plugin_meta_missing_required_key_raises_validation_error():
    """Plugin metadata validation should raise ValidationError for missing keys."""
    incomplete_meta = {
        "schema_version": 1,
        "name": "test",
        "version": "1.0",
        # Missing 'provider'
    }

    with pytest.raises(ValidationError, match="missing required key"):
        validate_plugin_meta(incomplete_meta)

    # Verify exception carries message
    try:
        validate_plugin_meta(incomplete_meta)
    except ValidationError as e:
        assert "provider" in str(e)


def test_validate_plugin_meta_invalid_type_raises_validation_error():
    """Plugin metadata validation should raise ValidationError for wrong types."""
    bad_meta = {
        "schema_version": "not_an_int",  # Should be int
        "name": "test",
        "version": "1.0",
        "provider": "test",
    }

    with pytest.raises(ValidationError, match="must be"):
        validate_plugin_meta(bad_meta)


def test_validate_plugin_meta_non_dict_raises_validation_error():
    """Plugin metadata validation should raise ValidationError for non-dict input."""
    with pytest.raises(ValidationError, match="plugin_meta must be a dict"):
        validate_plugin_meta("not_a_dict")


def test_validate_plugin_meta_invalid_capabilities_raises_validation_error():
    """Plugin metadata validation should raise ValidationError for invalid capabilities."""
    bad_capabilities = {
        "schema_version": 1,
        "name": "test",
        "version": "1.0",
        "provider": "test",
        "capabilities": "not_a_sequence",  # Should be a sequence of strings
    }

    with pytest.raises(ValidationError, match="must be a sequence"):
        validate_plugin_meta(bad_capabilities)


def test_validate_plugin_meta_empty_capabilities_raises_validation_error():
    """Plugin metadata validation should raise ValidationError for empty capabilities."""
    empty_capabilities = {
        "schema_version": 1,
        "name": "test",
        "version": "1.0",
        "provider": "test",
        "capabilities": [],  # Empty not allowed
    }

    with pytest.raises(ValidationError, match="must not be empty"):
        validate_plugin_meta(empty_capabilities)


def test_validate_plugin_meta_trusted_not_boolean_raises_validation_error():
    """Plugin metadata validation should raise ValidationError for non-boolean trusted."""
    bad_trusted = {
        "schema_version": 1,
        "name": "test",
        "version": "1.0",
        "provider": "test",
        "capabilities": ["test:capability"],
        "trusted": "yes",  # Should be boolean
    }

    with pytest.raises(ValidationError, match="must be a boolean"):
        validate_plugin_meta(bad_trusted)
