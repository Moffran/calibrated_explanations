"""Tests for configuration helper utilities."""

import contextlib
import os
from typing import Any, Iterator
import pytest

from calibrated_explanations.core import config_helpers
from calibrated_explanations.core.config_helpers import (
    coerce_string_tuple as coerce_string_tuple,
    read_pyproject_section as read_pyproject_section,
    split_csv as split_csv,
    write_pyproject_section as write_pyproject_section,
)

ORIG_TOMLLIB, ORIG_TOMLI_W = config_helpers.get_toml_modules_for_testing()

# Compatibility shim for pytest.MonkeyPatch: some tests expect an
# `addfinalizer` method (like `request.addfinalizer`). Older/newer
# pytest versions may not provide this on the `MonkeyPatch` object;
# add a small shim that stores finalizers and runs them when the
# monkeypatch `undo()` method is called at test teardown.
if not hasattr(pytest.MonkeyPatch, "addfinalizer"):
    orig_undo = pytest.MonkeyPatch.undo

    def addfinalizer(self, func):
        if not hasattr(self, "ce_finalizers"):
            self.ce_finalizers = []
        self.ce_finalizers.append(func)

    def undo_with_finalizers(self):
        # run stored finalizers first (LIFO), then perform original undo
        if hasattr(self, "ce_finalizers"):
            for f in reversed(self.ce_finalizers):
                with contextlib.suppress(Exception):
                    f()
        return orig_undo(self)

    pytest.MonkeyPatch.addfinalizer = addfinalizer
    pytest.MonkeyPatch.undo = undo_with_finalizers


def reset_toml_modules() -> None:
    """Restore TOML reader/writer modules when needed."""

    config_helpers.set_toml_modules_for_testing(tomllib=ORIG_TOMLLIB, tomli_w=ORIG_TOMLI_W)


@pytest.fixture(autouse=True)
def ensure_toml_modules_restored() -> Iterator[None]:
    """Always restore the TOML modules before and after each test."""

    config_helpers.set_toml_modules_for_testing(tomllib=ORIG_TOMLLIB, tomli_w=ORIG_TOMLI_W)
    try:
        yield
    finally:
        config_helpers.set_toml_modules_for_testing(tomllib=ORIG_TOMLLIB, tomli_w=ORIG_TOMLI_W)


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
    monkeypatch.chdir(tmp_path)
    monkeypatch.addfinalizer(
        lambda: config_helpers.set_toml_modules_for_testing(
            tomllib=ORIG_TOMLLIB, tomli_w=ORIG_TOMLI_W
        )
    )

    # No TOML reader available -> early fallback
    config_helpers.set_toml_modules_for_testing(tomllib=None)
    assert read_pyproject_section(("tool",)) == {}

    class DummyToml:
        def __init__(self, payload: dict[str, Any]) -> None:
            self.payload_data = payload

        def load(self, _fh: Any) -> dict[str, Any]:
            return self.payload_data

    # File missing -> still fallback
    config_helpers.set_toml_modules_for_testing(tomllib=DummyToml({}))
    assert read_pyproject_section(("tool", "missing")) == {}

    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text("[tool]\nname='demo'\n", encoding="utf-8")

    # Value present but not a mapping -> coerced to empty result
    config_helpers.set_toml_modules_for_testing(
        tomllib=DummyToml({"tool": {"calibrated_explanations": {"explanations": ["value"]}}})
    )
    assert read_pyproject_section(("tool", "calibrated_explanations", "explanations")) == {}

    # Proper mapping -> returned as dictionary copy
    config_helpers.set_toml_modules_for_testing(
        tomllib=DummyToml({"tool": {"calibrated_explanations": {"explanations": {"key": "value"}}}})
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


def test_read_pyproject_section_handles_load_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Load errors should be swallowed as empty config."""

    class FailingToml:
        @staticmethod
        def load(_fh: Any) -> dict[str, Any]:
            raise ValueError("boom")

    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text("[tool]\nname='demo'\n", encoding="utf-8")
    reset_toml_modules()
    config_helpers.set_toml_modules_for_testing(tomllib=FailingToml())

    assert read_pyproject_section(("tool",)) == {}


def test_write_pyproject_section_returns_false_when_missing_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Missing pyproject should short-circuit early."""

    class DummyTomliW:
        @staticmethod
        def dump(_data: dict[str, Any], fh) -> None:
            fh.write(b"noop")

    monkeypatch.chdir(tmp_path)
    reset_toml_modules()
    config_helpers.set_toml_modules_for_testing(tomllib=ORIG_TOMLLIB, tomli_w=DummyTomliW)

    assert write_pyproject_section(("tool",), {"k": "v"}) is False


def test_write_pyproject_section_rejects_non_mapping_prefix(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Non-dict traversal should return False."""

    class DummyTomliW:
        @staticmethod
        def dump(_data: dict[str, Any], fh) -> None:
            fh.write(b"noop")

    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text("tool = 1\n", encoding="utf-8")
    reset_toml_modules()
    config_helpers.set_toml_modules_for_testing(tomllib=ORIG_TOMLLIB, tomli_w=DummyTomliW)

    assert write_pyproject_section(("tool", "calibrated_explanations"), {"k": "v"}) is False


def test_write_pyproject_section_rejects_non_mapping_leaf(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Non-dict parent values should not be mutated."""

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
    reset_toml_modules()
    config_helpers.set_toml_modules_for_testing(tomllib=ORIG_TOMLLIB, tomli_w=DummyTomliW)

    assert (
        write_pyproject_section(("tool", "calibrated_explanations", "logging"), {"k": "v"}) is False
    )


def test_write_pyproject_section_handles_writer_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Writer exceptions should return False."""

    class FailingWriter:
        @staticmethod
        def dump(_data: dict[str, Any], _fh) -> None:
            raise config_helpers.CalibratedError("fail")

    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text("[tool]\nname='demo'\n", encoding="utf-8")
    reset_toml_modules()
    config_helpers.set_toml_modules_for_testing(tomllib=ORIG_TOMLLIB, tomli_w=FailingWriter)

    assert write_pyproject_section(("tool", "calibrated_explanations"), {"k": "v"}) is False


def test_read_pyproject_section_reraises_base_exception(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Non-Exception errors should bubble up."""

    class Exploder:
        @staticmethod
        def load(_fh: Any) -> dict[str, Any]:
            raise KeyboardInterrupt()

    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text("[tool]\nname='demo'\n", encoding="utf-8")
    reset_toml_modules()
    config_helpers.set_toml_modules_for_testing(tomllib=Exploder())

    with pytest.raises(KeyboardInterrupt):
        read_pyproject_section(("tool",))


def test_write_pyproject_section_stops_on_non_mapping_cursor(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Traversal should fail when encountering a non-dict parent."""

    class DummyTomliW:
        @staticmethod
        def dump(_data: dict[str, Any], fh) -> None:
            fh.write(b"noop")

    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text("tool = 1\n", encoding="utf-8")
    reset_toml_modules()
    config_helpers.set_toml_modules_for_testing(tomllib=ORIG_TOMLLIB, tomli_w=DummyTomliW)

    assert (
        write_pyproject_section(("tool", "calibrated_explanations", "logging"), {"k": "v"}) is False
    )


def test_write_pyproject_section_requires_writer(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Should return False when toml writer is unavailable."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text("[tool]\n", encoding="utf-8")

    reset_toml_modules()
    config_helpers.set_toml_modules_for_testing(tomllib=ORIG_TOMLLIB, tomli_w=None)

    assert write_pyproject_section(("tool", "calibrated_explanations"), {"key": "value"}) is False


def test_write_pyproject_section_returns_false_on_load_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    """Should short-circuit when loading pyproject fails."""

    class DummyTomllib:
        @staticmethod
        def load(_fh: Any) -> dict[str, Any]:
            raise config_helpers.CalibratedError("boom")

    class DummyTomliW:
        @staticmethod
        def dump(_data: dict[str, Any], fh) -> None:
            fh.write(b"content")

    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text("[tool]\n", encoding="utf-8")
    reset_toml_modules()
    config_helpers.set_toml_modules_for_testing(tomllib=DummyTomllib(), tomli_w=DummyTomliW())

    assert write_pyproject_section(("tool", "calibrated_explanations"), {"key": "value"}) is False


def test_write_pyproject_section_updates_nested_path(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Should update nested section and persist through writer stub."""

    class RecordingTomliW:
        dumped: dict[str, Any] | None = None

        @classmethod
        def dump(cls, data: dict[str, Any], fh) -> None:
            cls.dumped = data
            fh.write(b"written = true\n")

    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text("[tool]\nname='demo'\n", encoding="utf-8")
    reset_toml_modules()
    config_helpers.set_toml_modules_for_testing(tomllib=ORIG_TOMLLIB, tomli_w=RecordingTomliW)

    path = ("tool", "calibrated_explanations", "logging")
    payload = {"diagnostic_mode": True}

    assert write_pyproject_section(path, payload) is True
    assert RecordingTomliW.dumped is not None
    assert (
        RecordingTomliW.dumped.get("tool", {}).get("calibrated_explanations", {}).get("logging")
        == payload
    )
