from types import SimpleNamespace

import pytest

from src.calibrated_explanations import _plots


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    """Ensure environment variables are cleaned between tests."""
    monkeypatch.delenv("CE_PLOT_STYLE", raising=False)
    monkeypatch.delenv("CE_PLOT_STYLE_FALLBACKS", raising=False)


def test_read_plot_pyproject_extracts_nested_settings(tmp_path, monkeypatch):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        """
[tool.calibrated_explanations.plots]
style = "py-style"
fallbacks = ["py-fallback", "legacy"]
"""
    )

    monkeypatch.chdir(tmp_path)

    config = _plots._read_plot_pyproject()

    assert config["style"] == "py-style"
    assert config["fallbacks"] == ["py-fallback", "legacy"]


def test_read_plot_pyproject_missing_file_returns_empty_dict(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert _plots._read_plot_pyproject() == {}


def test_split_csv_handles_strings_sequences_and_invalid_values():
    assert _plots._split_csv(" a , b , , c ") == ("a", "b", "c")
    assert _plots._split_csv(["first", "", "second", 3]) == ("first", "second")
    assert _plots._split_csv(None) == ()
    assert _plots._split_csv(False) == ()


def test_resolve_plot_style_chain_orders_sources(monkeypatch):
    monkeypatch.setenv("CE_PLOT_STYLE", "env-style")
    monkeypatch.setenv("CE_PLOT_STYLE_FALLBACKS", "fallback-1, fallback-2")
    monkeypatch.setattr(
        _plots,
        "_read_plot_pyproject",
        lambda: {"style": "py-style", "fallbacks": ("py-fallback", "legacy")},
    )

    explainer = SimpleNamespace(
        _last_explanation_mode="prob", _plot_plugin_fallbacks={"prob": ("plugin", "legacy")}
    )

    chain = _plots._resolve_plot_style_chain(explainer, "explicit")

    assert chain == (
        "explicit",
        "env-style",
        "fallback-1",
        "fallback-2",
        "py-style",
        "py-fallback",
        "legacy",
        "plugin",
    )


def test_resolve_plot_style_chain_defaults_to_legacy(monkeypatch):
    monkeypatch.setattr(_plots, "_read_plot_pyproject", lambda: {})
    explainer = SimpleNamespace(_plot_plugin_fallbacks=SimpleNamespace())

    chain = _plots._resolve_plot_style_chain(explainer, None)

    assert chain == ("legacy",)


def test_require_matplotlib_raises_helpful_error(monkeypatch):
    monkeypatch.setattr(_plots, "plt", None)
    monkeypatch.setattr(_plots, "mcolors", None)
    monkeypatch.setattr(_plots, "_MATPLOTLIB_IMPORT_ERROR", RuntimeError("missing"))

    with pytest.raises(RuntimeError) as excinfo:
        _plots.__require_matplotlib()

    message = str(excinfo.value)
    assert "Plotting requires matplotlib" in message
    assert "missing" in message
