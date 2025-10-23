"""Tests for lightweight utilities in :mod:`calibrated_explanations.plotting`."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

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
    ],
)
def test_split_csv_normalises_and_filters_values(value, expected):
    """``_split_csv`` should normalise whitespace and ignore non-string values."""

    assert plotting._split_csv(value) == expected


def test_resolve_plot_style_chain_prioritises_unique_sources(monkeypatch):
    """The style chain should respect priority order and deduplicate entries."""

    dummy_explainer = SimpleNamespace(
        _last_explanation_mode="probabilistic",
        _plot_plugin_fallbacks={"probabilistic": ("env-style", "plugin-style")},
    )

    monkeypatch.setenv("CE_PLOT_STYLE", "env-style")
    monkeypatch.setenv("CE_PLOT_STYLE_FALLBACKS", "fallback-one, fallback-two")
    monkeypatch.setattr(
        plotting,
        "_read_plot_pyproject",
        lambda: {"style": "toml-style", "fallbacks": ["fallback-two", "toml-extra"]},
    )

    chain = plotting._resolve_plot_style_chain(dummy_explainer, "explicit-style")

    assert chain == (
        "explicit-style",
        "env-style",
        "fallback-one",
        "fallback-two",
        "toml-style",
        "toml-extra",
        "plugin-style",
        "plot_spec.default",
        "legacy",
    )


def test_resolve_plot_style_chain_defaults_to_legacy(monkeypatch):
    """Without any hints the chain should gracefully fall back to the legacy backend."""

    monkeypatch.delenv("CE_PLOT_STYLE", raising=False)
    monkeypatch.delenv("CE_PLOT_STYLE_FALLBACKS", raising=False)
    monkeypatch.setattr(plotting, "_read_plot_pyproject", lambda: {})

    chain = plotting._resolve_plot_style_chain(SimpleNamespace(), None)

    assert chain == ("plot_spec.default", "legacy")
