import pytest
import importlib

pytest.importorskip("matplotlib")

# dynamic import after optional dependency check
_plots = importlib.import_module("calibrated_explanations.viz.plots")


def test_plot_proba_triangle_returns_fig():
    # Should return a matplotlib Figure and not raise
    fig = _plots._plot_proba_triangle()
    import matplotlib.figure as _mf

    assert isinstance(fig, _mf.Figure)
    # should contain at least one axis and some lines
    assert len(fig.axes) >= 1
    # close explicitly
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_setup_plot_style_accepts_override():
    # Provide a simple override and ensure it is applied to the returned config
    cfg = {"figure": {"width": "7"}}
    config = _plots.__setup_plot_style(style_override=cfg)
    assert config["figure"]["width"] == "7"
