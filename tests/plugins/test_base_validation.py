from __future__ import annotations

import pytest

from calibrated_explanations.plugins.base import validate_plugin_meta


def _make_valid_meta() -> dict[str, object]:
    """Return a minimal but valid ``plugin_meta`` payload."""

    return {
        "schema_version": 1,
        "name": "tests.example",
        "version": "1.0",
        "provider": "tests",
        "capabilities": ["explain"],
    }


def test_validate_plugin_meta_rejects_non_dict():
    with pytest.raises(ValueError, match="must be a dict"):
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
    meta = _make_valid_meta()
    meta["capabilities"] = bad_value

    with pytest.raises(ValueError, match=message):
        validate_plugin_meta(meta)


@pytest.mark.parametrize(
    "key, value",
    [
        ("schema_version", "1"),
        ("name", ""),
        ("provider", 42),
    ],
)
def test_required_scalar_fields_are_validated(key, value):
    meta = _make_valid_meta()
    meta[key] = value

    with pytest.raises(ValueError, match="must be a non-empty"):
        validate_plugin_meta(meta)


def test_capabilities_required():
    meta = _make_valid_meta()
    meta.pop("capabilities")

    with pytest.raises(ValueError, match="missing required key: capabilities"):
        validate_plugin_meta(meta)


def test_checksum_type_is_validated():
    meta = _make_valid_meta()
    meta["checksum"] = 123

    with pytest.raises(ValueError, match="checksum"):
        validate_plugin_meta(meta)


def test_trusted_must_be_boolean():
    meta = _make_valid_meta()
    meta["trusted"] = "yes"

    with pytest.raises(ValueError, match="must be a boolean"):
        validate_plugin_meta(meta)


def test_trust_mapping_is_normalised():
    meta = _make_valid_meta()
    meta["trust"] = {"trusted": 1}

    validate_plugin_meta(meta)

    assert meta["trusted"] is True


def test_trust_scalar_is_normalised_to_boolean():
    meta = _make_valid_meta()
    meta["trust"] = 0

    validate_plugin_meta(meta)

    assert meta["trusted"] is False


def test_default_trust_value_is_false():
    meta = _make_valid_meta()

    validate_plugin_meta(meta)

    assert meta["trusted"] is False
