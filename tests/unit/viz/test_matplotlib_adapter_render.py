from __future__ import annotations

from calibrated_explanations.viz import matplotlib_adapter
from calibrated_explanations.viz.plotspec import BarHPanelSpec, BarItem, IntervalHeaderSpec, PlotSpec


def test_render_noop_when_no_outputs_requested():
    assert matplotlib_adapter.render({}) is None


def test_render_dict_spec_builds_primitives():
    spec = {
        "plot_spec": {
            "kind": "global.scatter",
            "global_entries": {
                "proba": [0.2, 0.8],
                "uncertainty": [0.1, 0.3],
            },
        },
    }

    wrapper = matplotlib_adapter.render(spec, export_drawn_primitives=True)

    assert isinstance(wrapper, dict)
    assert wrapper["plot_spec"]["kind"] == "global.scatter"
    assert any(item["type"] == "scatter" for item in wrapper["primitives"])


def test_render_plot_spec_returns_primitives():
    header = IntervalHeaderSpec(pred=0.6, low=0.2, high=0.8)
    bars = [
        BarItem(label="f1", value=0.4, interval_low=0.1, interval_high=0.5, color_role="positive"),
        BarItem(label="f2", value=-0.3, interval_low=-0.5, interval_high=0.1, color_role="negative"),
    ]
    body = BarHPanelSpec(bars=bars)
    spec = PlotSpec(title="Demo", header=header, body=body)

    wrapper = matplotlib_adapter.render(spec, export_drawn_primitives=True)

    assert "header" in wrapper
    assert "solids" in wrapper
    assert wrapper["plot_spec"]["title"] == "Demo"
    assert len(wrapper["primitives"]) > 0
