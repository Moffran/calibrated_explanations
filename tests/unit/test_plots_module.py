"""Tests for lightweight utilities in :mod:`calibrated_explanations.plotting`."""

from __future__ import annotations

import os
import pytest
from pathlib import Path
import warnings

# Suppress internal deprecation warning from viz.plots importing legacy plotting
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from calibrated_explanations.viz import plots as plotting


def test_read_plot_pyproject_handles_missing_file(tmp_path, monkeypatch):
    """When no ``pyproject.toml`` exists the helper should fall back to an empty dict."""

    monkeypatch.chdir(tmp_path)

    assert plotting._read_plot_pyproject() == {}


def test_read_plot_pyproject_extracts_nested_plot_settings(tmp_path, monkeypatch):
    """Ensure the helper returns the dedicated plotting settings section when present."""

    monkeypatch.chdir(tmp_path)

    (tmp_path / "pyproject.toml").write_text(
        """
        [tool.calibrated_explanations.plots]
        style = "toml-style"
        fallbacks = ["a", "b"]
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    assert plotting._read_plot_pyproject() == {"style": "toml-style", "fallbacks": ["a", "b"]}


@pytest.mark.parametrize(
    "value, expected",
    [
        ("first, second, third", ("first", "second", "third")),
        (["keep", "  also keep  ", 42, ""], ("keep", "also keep")),
        (None, ()),
        ("", ()),
        ("   ", ()),
        ("only_one", ("only_one",)),
        ("  spaced  ", ("spaced",)),
        ("a,b,c,", ("a", "b", "c")),
        (",a,b", ("a", "b")),
    ],
)
def test_split_csv_normalises_and_filters_values(value, expected):
    """``_split_csv`` should normalise whitespace and ignore non-string values."""

    assert plotting._split_csv(value) == expected


def test_should_read_plot_pyproject_with_malformed_toml(tmp_path, monkeypatch):
    """Reading malformed TOML should handle gracefully (e.g., empty or invalid)."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text("[invalid toml", encoding="utf-8")

    # Should return empty dict or raise handled exception, depending on implementation
    result = plotting._read_plot_pyproject()
    assert isinstance(result, dict)


def test_should_read_plot_pyproject_with_missing_section(tmp_path, monkeypatch):
    """Reading valid TOML without plots section should return empty dict."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text(
        """
        [build-system]
        requires = ["setuptools"]
        """,
        encoding="utf-8",
    )

    assert plotting._read_plot_pyproject() == {}


def test_should_read_plot_pyproject_with_partial_settings(tmp_path, monkeypatch):
    """Reading TOML with some plot settings should extract available keys."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text(
        """
        [tool.calibrated_explanations.plots]
        style = "minimal"
        """,
        encoding="utf-8",
    )

    result = plotting._read_plot_pyproject()
    assert result.get("style") == "minimal"
    assert "fallbacks" not in result or result["fallbacks"] is None


def test_should_handle_derive_threshold_labels_with_none():
    """Derive labels should handle None threshold gracefully."""
    pos_label, neg_label = plotting._derive_threshold_labels(None)
    assert pos_label == "Target within threshold"
    assert neg_label == "Outside threshold"


def test_should_handle_derive_threshold_labels_with_numeric():
    """Derive labels should handle numeric threshold."""
    pos_label, neg_label = plotting._derive_threshold_labels(0.5)
    assert pos_label == "Y < 0.50"
    assert neg_label == "Y >= 0.50"


def test_should_handle_derive_threshold_labels_with_sequence():
    """Derive labels should handle sequence thresholds."""
    pos_label, neg_label = plotting._derive_threshold_labels([0.2, 0.8])
    assert pos_label == "0.20 <= Y < 0.80"
    assert neg_label == "Outside interval"


def test_should_format_save_path_with_string():
    """_format_save_path should handle string paths."""
    result = plotting._format_save_path("/some/path", "file.png")
    expected = str(Path("/some/path") / "file.png")
    assert result == expected


def test_should_format_save_path_with_pathlib():
    """_format_save_path should handle pathlib.Path objects."""
    result = plotting._format_save_path(Path("/some/path"), "file.png")
    expected = str(Path("/some/path") / "file.png")
    assert result == expected


def test_should_format_save_path_with_empty_string():
    """_format_save_path should handle empty string."""
    result = plotting._format_save_path("", "file.png")
    assert result == "file.png"


def test_should_format_save_path_with_slash_suffix():
    """_format_save_path should handle paths with trailing slash."""
    # Note: This behavior depends on OS separator.
    # The implementation checks for / or \ explicitly.
    result = plotting._format_save_path("/some/path/", "file.png")
    assert result == "/some/path/file.png"
    
    # Test backslash on Windows if applicable, or just generic check
    if os.sep == "\\":
        result_win = plotting._format_save_path("C:\\some\\path\\", "file.png")
        assert result_win == "C:\\some\\path\\file.png"


def test_should_derive_threshold_labels_with_invalid_sequence():
    """Derive labels should handle invalid sequence thresholds gracefully."""
    # Pass invalid sequence (too short, not convertible to floats)
    pos_label, neg_label = plotting._derive_threshold_labels(["invalid"])
    assert pos_label == "Target within threshold"
    assert neg_label == "Outside threshold"
