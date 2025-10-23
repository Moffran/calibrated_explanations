from __future__ import annotations

import warnings

import pytest

from calibrated_explanations.plugins.registry import (
    ensure_builtin_plugins,
    list_explanation_descriptors,
    mark_explanation_trusted,
    mark_explanation_untrusted,
    validate_explanation_metadata,
)


def _base_metadata() -> dict:
    return {
        "name": "test",
        "schema_version": 1,
        "capabilities": ["explain"],
        "modes": ("factual",),
        "tasks": ("classification",),
        "dependencies": (),
        "trust": False,
    }


def test_validate_allows_canonical_modes() -> None:
    meta = _base_metadata()

    normalised = validate_explanation_metadata(meta)

    assert normalised["modes"] == ("factual",)


def test_validate_alias_emits_warning_and_normalises() -> None:
    meta = _base_metadata()
    meta["modes"] = ("explanation:factual", "factual")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", category=DeprecationWarning)
        normalised = validate_explanation_metadata(meta)

    assert normalised["modes"] == ("factual",)
    assert any(isinstance(item.message, DeprecationWarning) for item in caught)


@pytest.mark.parametrize(
    "field,value,expected",
    (
        ("interval_dependency", "core.interval.legacy", ("core.interval.legacy",)),
        ("plot_dependency", ("legacy",), ("legacy",)),
        ("fallbacks", ("alt", "legacy"), ("alt", "legacy")),
        ("fallbacks", "core.explanation.factual", ("core.explanation.factual",)),
    ),
)
def test_dependency_fields_are_normalised(field: str, value, expected) -> None:
    meta = _base_metadata()
    meta[field] = value

    normalised = validate_explanation_metadata(meta)

    assert normalised[field] == expected


def test_tasks_field_required_and_validated() -> None:
    meta = _base_metadata()
    meta["tasks"] = ("classification", "regression")

    normalised = validate_explanation_metadata(meta)

    assert normalised["tasks"] == ("classification", "regression")

    meta_invalid = _base_metadata()
    meta_invalid["tasks"] = ("unknown",)

    with pytest.raises(ValueError):
        validate_explanation_metadata(meta_invalid)


def test_schema_version_future_rejected() -> None:
    meta = _base_metadata()
    meta["schema_version"] = 999

    with pytest.raises(ValueError) as exc:
        validate_explanation_metadata(meta)

    assert "unsupported schema_version" in str(exc.value)


def test_list_descriptors_respects_trust_state() -> None:
    ensure_builtin_plugins()
    descriptors = list_explanation_descriptors()
    assert any(d.identifier == "core.explanation.factual" for d in descriptors)

    mark_explanation_untrusted("core.explanation.factual")
    try:
        trusted_only = list_explanation_descriptors(trusted_only=True)
        assert all(d.identifier != "core.explanation.factual" for d in trusted_only)
    finally:
        mark_explanation_trusted("core.explanation.factual")


def test_validate_explanation_metadata_invalid_modes():
    meta = _base_metadata()
    meta["modes"] = ("invalid_mode",)

    with pytest.raises(ValueError, match="unsupported values"):
        validate_explanation_metadata(meta)


def test_validate_explanation_metadata_no_modes():
    meta = _base_metadata()
    del meta["modes"]

    with pytest.raises(ValueError, match="plugin_meta missing required key: modes"):
        validate_explanation_metadata(meta)


def test_validate_explanation_metadata_missing_trust():
    meta = _base_metadata()
    del meta["trust"]

    with pytest.raises(ValueError, match="missing required key: trust"):
        validate_explanation_metadata(meta)
