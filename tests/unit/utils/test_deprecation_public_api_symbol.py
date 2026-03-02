"""Tests for deprecate_public_api_symbol helper."""

from __future__ import annotations

import pytest

from calibrated_explanations.utils.deprecation import deprecate_public_api_symbol


def test_deprecate_public_api_symbol_formats_message_with_details():
    with pytest.warns(DeprecationWarning) as captured:
        deprecate_public_api_symbol(
            "LegacyThing",
            "from calibrated_explanations import LegacyThing",
            "from calibrated_explanations.modern import LegacyThing",
            removal_version="v9.9.9",
            extra_context="Use the modern import path.",
        )

    message = str(captured[0].message)
    assert "LegacyThing" in message
    assert "DEPRECATED" in message
    assert "RECOMMENDED" in message
    assert "v9.9.9" in message
    assert "Use the modern import path." in message


def test_deprecate_public_api_symbol_formats_message_without_details():
    with pytest.warns(DeprecationWarning) as captured:
        deprecate_public_api_symbol(
            "LegacyThing",
            "old.path",
            "new.path",
        )

    message = str(captured[0].message)
    assert "LegacyThing" in message
    assert "Details:" not in message
