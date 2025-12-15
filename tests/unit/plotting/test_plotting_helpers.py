import configparser
from types import SimpleNamespace

import pytest

from calibrated_explanations import plotting
from calibrated_explanations.utils.exceptions import ConfigurationError
from calibrated_explanations.viz import coloring


class _DummyExplainer:
    def __init__(self):
        self._last_explanation_mode = "factual"
        self._plot_plugin_fallbacks = {"factual": ["fallback-mode"]}


def test_resolve_plot_style_chain_respects_order(monkeypatch):
    monkeypatch.setenv("CE_PLOT_STYLE", "env")
    monkeypatch.setenv("CE_PLOT_STYLE_FALLBACKS", "env-fallback,a2")
    monkeypatch.setattr(
        plotting, "_read_plot_pyproject", lambda: {"style": "py", "fallbacks": "py-fb"}
    )

    explainer = _DummyExplainer()
    chain = plotting._resolve_plot_style_chain(explainer, "explicit")

    assert chain[0] == "explicit"
    assert "env" in chain
    assert "env-fallback" in chain
    assert "py" in chain
    assert "plot_spec.default" in chain
    assert chain[-1] == "legacy"


def test_load_and_update_plot_config(tmp_path, monkeypatch):
    config_path = tmp_path / "plot_config.ini"
    monkeypatch.setattr(plotting, "_plot_config_path", lambda: config_path)

    base = plotting.load_plot_config()
    assert base["style"]["base"]

    plotting.update_plot_config({"style": {"base": "custom-style"}, "fonts": {"family": "Courier"}})
    read_config = configparser.ConfigParser()
    read_config.read(config_path)
    assert read_config["style"]["base"] == "custom-style"
    assert read_config["fonts"]["family"] == "Courier"


def test_require_matplotlib_raises_when_missing(monkeypatch):
    monkeypatch.setattr(plotting, "plt", None)
    monkeypatch.setattr(plotting, "mcolors", None)
    monkeypatch.setattr(plotting, "_MATPLOTLIB_IMPORT_ERROR", ImportError("boom"))
    with pytest.raises(ConfigurationError) as excinfo:
        plotting.__require_matplotlib()
    assert "matplotlib" in str(excinfo.value)


def test_setup_plot_style_applies_overrides(monkeypatch):
    dummy_style = SimpleNamespace()
    dummy_style.use = lambda _style: None
    dummy_plt = SimpleNamespace(style=dummy_style, rcParams={})
    monkeypatch.setattr(plotting, "__require_matplotlib", lambda: None)
    monkeypatch.setattr(plotting, "plt", dummy_plt)

    base_config = configparser.ConfigParser()
    base_config["style"] = {"base": "base-style"}
    base_config["fonts"] = {
        "family": "sans-serif",
        "sans_serif": "Arial",
        "axes_label_size": "12",
        "tick_label_size": "10",
        "legend_size": "10",
        "title_size": "14",
    }
    base_config["lines"] = {"width": "2"}
    base_config["grid"] = {"style": "--", "alpha": "0.5"}
    base_config["figure"] = {
        "dpi": "300",
        "save_dpi": "300",
        "facecolor": "white",
        "axes_facecolor": "white",
        "width": "10",
    }
    monkeypatch.setattr(plotting, "load_plot_config", lambda: base_config)

    overrides = {"style": {"base": "override"}, "fonts": {"family": "Courier"}}
    config = plotting.__setup_plot_style(style_override=overrides)

    assert config["style"]["base"] == "override"
    assert dummy_plt.rcParams["font.family"] == "Courier"


def test_color_brew_and_fill_color_behaviour():
    palette = coloring.color_brew(2)
    assert palette == [[86, 57, 229], [229, 71, 57]]
    assert coloring.get_fill_color({"predict": 0.75}) == "#ef8c83"
    assert coloring.get_fill_color({"predict": 0.25}) == "#9583ef"
    assert coloring.get_fill_color({"predict": "bad"}) == "#5639e5"
    assert coloring.get_fill_color({"predict": 0.85}, reduction=0.5) == "#f2a39c"
