"""Tests for configuration helper utilities."""

import os
from typing import Any
import pytest

from calibrated_explanations.core.config_helpers import (
    coerce_string_tuple as _coerce_string_tuple,
    read_pyproject_section as _read_pyproject_section,
    split_csv as _split_csv,
)


@pytest.mark.parametrize(
    "input_value, expected",
    [
        (None, ()),
        ("", ()),
        ("   ", ()),
        ("a", ("a",)),
        ("a,b", ("a", "b")),
        (" a , b ", ("a", "b")),
        ("a, ,b", ("a", "b")),
        ("a,b,c", ("a", "b", "c")),
        (False, ()),
    ],
)
def test_split_csv(input_value, expected):
    """Test split_csv with various inputs."""
    assert _split_csv(input_value) == expected


@pytest.mark.parametrize(
    "input_value, expected",
    [
        (None, ()),
        ("", ()),
        ("value", ("value",)),
        (["a", "b"], ("a", "b")),
        (("a", "b"), ("a", "b")),
        (["a", "", "b", None], ("a", "b")),
        ([1, "a", 2.5], ("a",)),
    ],
)
def test_coerce_string_tuple(input_value, expected):
    """Test coerce_string_tuple with various inputs."""
    assert _coerce_string_tuple(input_value) == expected


def test_read_pyproject_section_handles_multiple_sources(
    monkeypatch: pytest.MonkeyPatch, tmp_path: "os.PathLike[str]"
) -> None:
    """Test that read_pyproject_section handles various edge cases and fallbacks."""
    module = __import__("calibrated_explanations.core.config_helpers", fromlist=["_tomllib"])
    monkeypatch.chdir(tmp_path)

    # No TOML reader available -> early fallback
    monkeypatch.setattr(module, "_tomllib", None)
    assert _read_pyproject_section(("tool",)) == {}

    class DummyToml:
        def __init__(self, payload: dict[str, Any]) -> None:
            self._payload = payload

        def load(self, _fh: Any) -> dict[str, Any]:
            return self._payload

    # File missing -> still fallback
    monkeypatch.setattr(module, "_tomllib", DummyToml({}))
    assert _read_pyproject_section(("tool", "missing")) == {}

    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text("[tool]\nname='demo'\n", encoding="utf-8")

    # Value present but not a mapping -> coerced to empty result
    monkeypatch.setattr(
        module,
        "_tomllib",
        DummyToml({"tool": {"calibrated_explanations": {"explanations": ["value"]}}}),
    )
    assert _read_pyproject_section(("tool", "calibrated_explanations", "explanations")) == {}

    # Proper mapping -> returned as dictionary copy
    monkeypatch.setattr(
        module,
        "_tomllib",
        DummyToml({"tool": {"calibrated_explanations": {"explanations": {"key": "value"}}}}),
    )
    assert _read_pyproject_section(("tool", "calibrated_explanations", "explanations")) == {
        "key": "value"
    }


def test_read_pyproject_section_integration(tmp_path, monkeypatch):
    """Integration test for reading a real pyproject.toml file."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        """
        [tool.calibrated_explanations.explanations]
        factual = "py.identifier"
        factual_fallbacks = ["fb.one", "", "fb.two"]
        """.strip(),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    result = _read_pyproject_section(("tool", "calibrated_explanations", "explanations"))

    assert result == {
        "factual": "py.identifier",
        "factual_fallbacks": ["fb.one", "", "fb.two"],
    }
