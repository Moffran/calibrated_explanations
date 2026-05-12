"""Project metadata governance tests."""

from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


def test_pyproject_development_status_classifier_matches_release_phase() -> None:
    """Ensure release-governed maturity metadata matches package release phase."""
    project_root = Path(__file__).resolve().parents[1]
    pyproject_path = project_root / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    project = pyproject.get("project", {})
    classifiers = project.get("classifiers", [])

    version = project.get("version", "")
    development_status_classifiers = [
        classifier for classifier in classifiers if classifier.startswith("Development Status ::")
    ]

    assert "Development Status :: 3 - Alpha" not in development_status_classifiers

    if version.startswith("0.") or "rc" in version:
        expected = "Development Status :: 4 - Beta"
    else:
        expected = "Development Status :: 5 - Production/Stable"

    assert development_status_classifiers == [expected]
