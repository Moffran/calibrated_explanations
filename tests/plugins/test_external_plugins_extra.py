"""Tests for aggregated external plugin extras packaging."""
from __future__ import annotations

from importlib import metadata

import pytest


def test_external_plugins_extra_declares_expected_dependencies() -> None:
    """Ensure the aggregated ``external-plugins`` extra installs the curated stack."""
    try:
        requirements = metadata.requires("calibrated_explanations")
    except metadata.PackageNotFoundError as exc:  # pragma: no cover - safety guard for editable installs
        pytest.skip(f"calibrated_explanations distribution metadata unavailable: {exc}")

    assert requirements is not None

    expected = {
        "numpy>=1.24",
        "pandas>=2.0",
        "scikit-learn>=1.3",
    }
    extras = {
        requirement.split(";")[0].strip()
        for requirement in requirements
        if 'extra == "external-plugins"' in requirement
    }

    assert extras == expected
