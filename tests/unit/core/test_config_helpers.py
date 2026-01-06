"""Tests for configuration helper utilities."""

import os
from typing import Any
import pytest

from calibrated_explanations.core.config_helpers import (
    coerce_string_tuple as coerce_string_tuple,
    read_pyproject_section as read_pyproject_section,
    split_csv as split_csv,
    write_pyproject_section as write_pyproject_section,
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
def testsplit_csv(input_value, expected):
    """Test split_csv with various inputs."""
    assert split_csv(input_value) == expected


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
def testcoerce_string_tuple(input_value, expected):
    """Test coerce_string_tuple with various inputs."""
    assert coerce_string_tuple(input_value) == expected


def testread_pyproject_section_handles_multiple_sources(
    monkeypatch: pytest.MonkeyPatch, tmp_path: "os.PathLike[str]"
) -> None:
    """Test that read_pyproject_section handles various edge cases and fallbacks."""
    module = __import__("calibrated_explanations.core.config_helpers", fromlist=["_tomllib"])
    monkeypatch.chdir(tmp_path)

    # No TOML reader available -> early fallback
    monkeypatch.setattr(module, "_tomllib", None)
    assert read_pyproject_section(("tool",)) == {}

    class DummyToml:
        def __init__(self, payload: dict[str, Any]) -> None:
            self.payload_data = payload

        def load(self, _fh: Any) -> dict[str, Any]:
            return self.payload_data

    # File missing -> still fallback
    monkeypatch.setattr(module, "_tomllib", DummyToml({}))
    assert read_pyproject_section(("tool", "missing")) == {}

    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text("[tool]\nname='demo'\n", encoding="utf-8")

    # Value present but not a mapping -> coerced to empty result
    monkeypatch.setattr(
        module,
        "_tomllib",
        DummyToml({"tool": {"calibrated_explanations": {"explanations": ["value"]}}}),
    )
    assert read_pyproject_section(("tool", "calibrated_explanations", "explanations")) == {}

    # Proper mapping -> returned as dictionary copy
    monkeypatch.setattr(
        module,
        "_tomllib",
        DummyToml({"tool": {"calibrated_explanations": {"explanations": {"key": "value"}}}}),
    )
    assert read_pyproject_section(("tool", "calibrated_explanations", "explanations")) == {
        "key": "value"
    }


def testread_pyproject_section_integration(tmp_path, monkeypatch):
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

    result = read_pyproject_section(("tool", "calibrated_explanations", "explanations"))

    assert result == {
        "factual": "py.identifier",
        "factual_fallbacks": ["fb.one", "", "fb.two"],
    }


def test_read_pyproject_section_handles_load_error(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Load errors should be swallowed as empty config."""
    module = __import__("calibrated_explanations.core.config_helpers", fromlist=["_tomllib", "_tomli_w"])

    class FailingToml:
        @staticmethod
        def load(_fh: Any) -> dict[str, Any]:
            raise ValueError("boom")

    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text("[tool]\nname='demo'\n", encoding="utf-8")
    monkeypatch.setattr(module, "_tomllib", FailingToml())

    assert read_pyproject_section(("tool",)) == {}


def test_write_pyproject_section_returns_false_when_missing_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Missing pyproject should short-circuit early."""
    module = __import__("calibrated_explanations.core.config_helpers", fromlist=["_tomllib", "_tomli_w"])

    class DummyTomliW:
        @staticmethod
        def dump(_data: dict[str, Any], fh) -> None:
            fh.write(b"noop")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module, "_tomllib", module._tomllib)
    monkeypatch.setattr(module, "_tomli_w", DummyTomliW)

    assert write_pyproject_section(("tool",), {"k": "v"}) is False


def test_write_pyproject_section_rejects_non_mapping_prefix(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Non-dict traversal should return False."""
    module = __import__("calibrated_explanations.core.config_helpers", fromlist=["_tomllib", "_tomli_w"])

    class DummyTomliW:
        @staticmethod
        def dump(_data: dict[str, Any], fh) -> None:
            fh.write(b"noop")

    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text("tool = 1\n", encoding="utf-8")
    monkeypatch.setattr(module, "_tomllib", module._tomllib)
    monkeypatch.setattr(module, "_tomli_w", DummyTomliW)

    assert write_pyproject_section(("tool", "calibrated_explanations"), {"k": "v"}) is False


def test_write_pyproject_section_rejects_non_mapping_leaf(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Non-dict parent values should not be mutated."""
    module = __import__("calibrated_explanations.core.config_helpers", fromlist=["_tomllib", "_tomli_w"])

    class DummyTomliW:
        @staticmethod
        def dump(_data: dict[str, Any], fh) -> None:
            fh.write(b"noop")

    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text(
        """
        [tool]
        calibrated_explanations = "not a mapping"
        """.strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "_tomllib", module._tomllib)
    monkeypatch.setattr(module, "_tomli_w", DummyTomliW)

    assert write_pyproject_section(("tool", "calibrated_explanations", "logging"), {"k": "v"}) is False


def test_write_pyproject_section_handles_writer_failure(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Writer exceptions should return False."""
    module = __import__("calibrated_explanations.core.config_helpers", fromlist=["_tomllib", "_tomli_w", "CalibratedError"])

    class FailingWriter:
        @staticmethod
        def dump(_data: dict[str, Any], _fh) -> None:
            raise module.CalibratedError("fail")

    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text("[tool]\nname='demo'\n", encoding="utf-8")
    monkeypatch.setattr(module, "_tomllib", module._tomllib)
    monkeypatch.setattr(module, "_tomli_w", FailingWriter)

    assert write_pyproject_section(("tool", "calibrated_explanations"), {"k": "v"}) is False


def test_read_pyproject_section_reraises_base_exception(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Non-Exception errors should bubble up."""
    module = __import__("calibrated_explanations.core.config_helpers", fromlist=["_tomllib", "_tomli_w"])

    class Exploder:
        @staticmethod
        def load(_fh: Any) -> dict[str, Any]:
            raise KeyboardInterrupt()

    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text("[tool]\nname='demo'\n", encoding="utf-8")
    monkeypatch.setattr(module, "_tomllib", Exploder())

    with pytest.raises(KeyboardInterrupt):
        read_pyproject_section(("tool",))


def test_write_pyproject_section_stops_on_non_mapping_cursor(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Traversal should fail when encountering a non-dict parent."""
    module = __import__("calibrated_explanations.core.config_helpers", fromlist=["_tomllib", "_tomli_w"])

    class DummyTomliW:
        @staticmethod
        def dump(_data: dict[str, Any], fh) -> None:
            fh.write(b"noop")

    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text("tool = 1\n", encoding="utf-8")
    monkeypatch.setattr(module, "_tomllib", module._tomllib)
    monkeypatch.setattr(module, "_tomli_w", DummyTomliW)

    assert write_pyproject_section(("tool", "calibrated_explanations", "logging"), {"k": "v"}) is False


def test_write_pyproject_section_requires_writer(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Should return False when toml writer is unavailable."""
    module = __import__("calibrated_explanations.core.config_helpers", fromlist=["_tomllib", "_tomli_w"])
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text("[tool]\n", encoding="utf-8")

    monkeypatch.setattr(module, "_tomli_w", None)
    monkeypatch.setattr(module, "_tomllib", module._tomllib)

    assert write_pyproject_section(("tool", "calibrated_explanations"), {"key": "value"}) is False


def test_write_pyproject_section_returns_false_on_load_error(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Should short-circuit when loading pyproject fails."""
    module = __import__("calibrated_explanations.core.config_helpers", fromlist=["_tomllib", "_tomli_w"])

    class DummyTomllib:
        @staticmethod
        def load(_fh: Any) -> dict[str, Any]:
            raise module.CalibratedError("boom")

    class DummyTomliW:
        @staticmethod
        def dump(_data: dict[str, Any], fh) -> None:
            fh.write(b"content")

    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text("[tool]\n", encoding="utf-8")
    monkeypatch.setattr(module, "_tomllib", DummyTomllib())
    monkeypatch.setattr(module, "_tomli_w", DummyTomliW())

    assert write_pyproject_section(("tool", "calibrated_explanations"), {"key": "value"}) is False


def test_write_pyproject_section_updates_nested_path(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Should update nested section and persist through writer stub."""
    module = __import__("calibrated_explanations.core.config_helpers", fromlist=["_tomllib", "_tomli_w"])

    class RecordingTomliW:
        dumped: dict[str, Any] | None = None

        @classmethod
        def dump(cls, data: dict[str, Any], fh) -> None:
            cls.dumped = data
            fh.write(b"written = true\n")

    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text("[tool]\nname='demo'\n", encoding="utf-8")
    monkeypatch.setattr(module, "_tomllib", module._tomllib)
    monkeypatch.setattr(module, "_tomli_w", RecordingTomliW)

    path = ("tool", "calibrated_explanations", "logging")
    payload = {"diagnostic_mode": True}

    assert write_pyproject_section(path, payload) is True
    assert RecordingTomliW.dumped is not None
    assert RecordingTomliW.dumped.get("tool", {}).get("calibrated_explanations", {}).get("logging") == payload
