from __future__ import annotations

import pytest

from calibrated_explanations.core import ValidationError
from calibrated_explanations.plugins.base import (
    validate_plugin_config,
    validate_plugin_config_schema,
    validate_plugin_meta,
)


def make_valid_meta() -> dict[str, object]:
    """Return a minimal but valid ``plugin_meta`` payload."""

    return {
        "schema_version": 1,
        "name": "tests.example",
        "version": "1.0",
        "provider": "tests",
        "capabilities": ["explain"],
    }


def test_validate_plugin_meta_rejects_non_dict():
    with pytest.raises(ValidationError, match="must be a dict"):
        validate_plugin_meta([])  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "bad_value, message",
    [
        ("explain", "sequence of strings"),
        (["explain", 1], "only string values"),
        (["", "explain"], "non-empty string values"),
        ([], "must not be empty"),
    ],
)
def test_capabilities_must_be_sequence_of_strings(bad_value, message):
    meta = make_valid_meta()
    meta["capabilities"] = bad_value

    with pytest.raises(ValidationError, match=message):
        validate_plugin_meta(meta)


def test_capabilities_required():
    meta = make_valid_meta()
    meta.pop("capabilities")

    with pytest.raises(ValidationError, match="missing required key: capabilities"):
        validate_plugin_meta(meta)


def test_checksum_type_is_validated():
    meta = make_valid_meta()
    meta["checksum"] = 123

    with pytest.raises(ValidationError, match="checksum"):
        validate_plugin_meta(meta)


def test_trusted_must_be_boolean():
    meta = make_valid_meta()
    meta["trusted"] = "yes"

    with pytest.raises(ValidationError, match="must be a boolean"):
        validate_plugin_meta(meta)


def test_validate_plugin_meta_accepts_provisional_config_schema():
    meta = make_valid_meta()
    meta["config_schema"] = {
        "version": 1,
        "keys": {
            "mode": {"type": "str", "default": "balanced", "choices": ["fast", "balanced"]},
            "api_token": {"type": "str", "sensitive": True, "required": False},
        },
    }

    validate_plugin_meta(meta)


def test_validate_plugin_config_schema_rejects_malformed_shape():
    with pytest.raises(ValidationError, match="type"):
        validate_plugin_config_schema(
            {
                "version": 1,
                "keys": {"mode": {"type": "unsupported"}},
            }
        )


def test_validate_plugin_config_applies_defaults_and_rejects_unknown_keys():
    schema = {
        "version": 1,
        "keys": {
            "mode": {"type": "str", "default": "balanced"},
            "labels": {"type": "list[str]", "default": ["a"]},
        },
    }

    resolved = validate_plugin_config(
        plugin_id="tests.example",
        config={"labels": ["x", "y"]},
        schema=schema,
    )

    assert resolved["mode"] == "balanced"
    assert resolved["labels"] == ("x", "y")
    with pytest.raises(ValidationError, match="unknown key"):
        validate_plugin_config(
            plugin_id="tests.example",
            config={"extra": True},
            schema=schema,
        )
