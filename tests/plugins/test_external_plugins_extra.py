"""Tests for aggregated external plugin extras packaging."""

from __future__ import annotations

from importlib import metadata
from pathlib import Path

import pytest

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - fallback for older interpreters
    import tomli as tomllib  # type: ignore[no-redef]


def parse_external_plugins_from_pyproject() -> set[str]:
    project_root = Path(__file__).resolve().parents[2]
    pyproject_path = project_root / "pyproject.toml"
    if not pyproject_path.is_file():  # pragma: no cover - defensive branch for odd layouts
        pytest.skip(f"pyproject.toml not found at {pyproject_path}")

    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    extras = pyproject.get("project", {}).get("optional-dependencies", {}).get("external-plugins")
    if extras is None:  # pragma: no cover - ensures informative failures if config drifts
        pytest.fail("external-plugins extra not declared in pyproject.toml")
    return {dependency.strip() for dependency in extras}


def test_external_plugins_extra_declares_expected_dependencies() -> None:
    """Ensure the aggregated ``external-plugins`` extra installs the curated stack."""
    expected = {
        "numpy>=1.24",
        "pandas>=2.0",
        "scikit-learn>=1.3",
    }

    try:
        requirements = metadata.requires("calibrated_explanations")
    except metadata.PackageNotFoundError:
        extras = parse_external_plugins_from_pyproject()
    else:
        if requirements is None:
            extras = parse_external_plugins_from_pyproject()
        else:
            extras = {
                requirement.split(";")[0].strip()
                for requirement in requirements
                if 'extra == "external-plugins"' in requirement
            }
            if not extras:
                extras = parse_external_plugins_from_pyproject()

    assert extras == expected
