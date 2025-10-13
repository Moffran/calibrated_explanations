"""Additional coverage-focused tests for the matplotlib PlotSpec adapter."""

from __future__ import annotations

import math

import pytest

from calibrated_explanations.viz.matplotlib_adapter import render
from calibrated_explanations.viz.plotspec import (
    BarHPanelSpec,
    BarItem,
    IntervalHeaderSpec,
    PlotSpec,
)


def test_render_dict_triangular_adds_triangle_and_quiver():
    spec = {
        "plot_spec": {
            "kind": "triangular",
            # Any truthy payload ensures the adapter adds a placeholder primitive.
            "triangular": {"edges": [0, 1, 2]},
        }
    }

    wrapper = render(spec, export_drawn_primitives=True)

    assert wrapper["triangle_background"]["type"] == "triangle_background"
    assert any(p["type"] == "quiver" for p in wrapper["primitives"])


def test_render_dict_global_multiclass_and_save_behavior():
    spec = {
        "plot_spec": {
            "kind": "global.binary",
            "global_entries": {
                "proba": [[0.1, 0.9], [0.8, 0.2]],
                "uncertainty": [[0.05, 0.95], [0.2, 0.8]],
            },
            "save_behavior": {"default_exts": ["png", "pdf"]},
        }
    }

    wrapper = render(spec, export_drawn_primitives=True)

    scatter = [p for p in wrapper["primitives"] if p["type"] == "scatter"]
    assert len(scatter) == 2
    ids = {p["id"] for p in wrapper["primitives"]}
    assert {"save.png", "save.pdf"}.issubset(ids)


def test_render_dict_global_handles_bad_values():
    class BadFloat:
        def __float__(self) -> float:
            raise TypeError("not convertible")

    spec = {
        "plot_spec": {
            "kind": "global.scatter",
            "global_entries": {"proba": [BadFloat()], "uncertainty": [0.1]},
        }
    }

    wrapper = render(spec, export_drawn_primitives=True)

    assert any(p["id"] == "global.scatter.summary" for p in wrapper["primitives"])


def test_render_body_only_height_handles_label_errors():
    class ExplodingLabel:
        def __init__(self) -> None:
            self._calls = 0

        def __str__(self) -> str:
            self._calls += 1
            if self._calls == 1:
                raise ValueError("boom")
            return "recovered"

    body = BarHPanelSpec(
        bars=[
            BarItem(label="line1\nline2", value=0.3, instance_value=1.0),
            BarItem(label=ExplodingLabel(), value=-0.2, instance_value=0.2),
        ],
        xlabel="Contribution",
        ylabel="Features",
    )
    spec = PlotSpec(title=None, header=None, body=body)

    fig = render(spec, return_fig=True)

    try:
        assert len(fig.axes) >= 1
        height = fig.get_size_inches()[1]
        assert math.isfinite(height) and height >= 3.0
    finally:
        # Close the figure explicitly because the adapter leaves it open when return_fig=True.
        fig.clf()


def test_render_dual_header_body_exports_expected_primitives():
    header = IntervalHeaderSpec(
        pred=0.6,
        low=0.25,
        high=0.9,
        xlim=(0.5, 0.5),
        xlabel="Probability",
        neg_label="no",
        pos_label="yes",
        dual=True,
    )
    bars = [
        # Default behaviour suppresses the solid when the interval straddles zero.
        BarItem(
            label="suppressed",
            value=0.62,
            interval_low=0.52,
            interval_high=0.82,
            instance_value=0.1,
        ),
        # Explicit flag keeps the solid even when the interval crosses zero.
        BarItem(
            label="solid",
            value=0.9,
            interval_low=0.4,
            interval_high=0.95,
            solid_on_interval_crosses_zero=False,
            instance_value=0.2,
        ),
        # Bar without interval exercises the simple fill branch.
        BarItem(label="no_interval", value=-0.15, instance_value=-0.05),
    ]
    body = BarHPanelSpec(bars=bars, xlabel="Logit", ylabel="Features")
    delattr(body, "solid_on_interval_crosses_zero")
    spec = PlotSpec(title="Dual", header=header, body=body)

    wrapper = render(spec, export_drawn_primitives=True)

    header_data = wrapper["header"]
    assert "negative" in header_data and "positive" in header_data
    assert header_data["negative"]["solid"][0] == 0.0
    assert any(solid["index"] == 2 for solid in wrapper.get("solids", []))
    overlays = wrapper.get("overlays", [])
    assert any(overlay["index"] == 1 and overlay["x0"] <= 0.0 for overlay in overlays)
    assert any(overlay["index"] == 1 and overlay["x1"] >= 0.0 for overlay in overlays)
    assert "base_interval" in wrapper and "body" in wrapper["base_interval"]


def test_render_regression_body_interval_behaviour(tmp_path):
    header = IntervalHeaderSpec(
        pred=0.4,
        low=0.2,
        high=0.7,
        xlabel="Score",
        ylabel="Target",
        dual=False,
        xlim=(0.2, 0.2),
    )
    bars = [
        BarItem(
            label="primary",
            value=0.5,
            interval_low=0.3,
            interval_high=0.8,
            instance_value="1.0",
        ),
        # Interval matches header range and should be skipped.
        BarItem(
            label="match_header",
            value=0.4,
            interval_low=0.2,
            interval_high=0.7,
        ),
        BarItem(
            label="negative_cross",
            value=-0.1,
            interval_low=-0.3,
            interval_high=0.2,
            solid_on_interval_crosses_zero=False,
            instance_value="2.0",
        ),
        BarItem(label="plain", value=0.05),
    ]
    body = BarHPanelSpec(
        bars=bars,
        xlabel="Contribution",
        ylabel="Features",
        solid_on_interval_crosses_zero=False,
    )
    spec = PlotSpec(title="Regression", header=header, body=body)

    out_path = tmp_path / "regression_plot.png"
    wrapper = render(
        spec,
        save_path=str(out_path),
        export_drawn_primitives=True,
    )

    assert out_path.exists()
    assert any(solid["index"] == 0 for solid in wrapper.get("solids", []))
    assert any(overlay["index"] == 2 for overlay in wrapper.get("overlays", []))
    assert "base_interval" in wrapper and "body" in wrapper["base_interval"]


def test_render_respects_figsize_and_show(monkeypatch):
    shown = {"called": False}

    def fake_show():
        shown["called"] = True

    monkeypatch.setattr("matplotlib.pyplot.show", fake_show)

    header = IntervalHeaderSpec(pred=0.5, low=0.2, high=0.8, dual=False)
    body = BarHPanelSpec(bars=[BarItem(label="only", value=0.0)])
    spec = PlotSpec(figure_size=(4, 6), header=header, body=body)

    fig = render(spec, show=True, return_fig=True)

    try:
        assert pytest.approx(fig.get_size_inches()[1], rel=1e-6) == 6
        assert shown["called"], "matplotlib.pyplot.show should be invoked"
    finally:
        fig.clf()


def test_render_handles_invalid_header_inputs(monkeypatch):
    class FlakyNumber:
        def __init__(self, value: float):
            self.value = value
            self.calls = 0

        def __float__(self) -> float:
            self.calls += 1
            if self.calls == 1:
                raise ValueError("temporary failure")
            return float(self.value)

        def __rsub__(self, other: float) -> float:
            return float(other) - float(self.value)

    header = IntervalHeaderSpec(
        pred=FlakyNumber(0.4),
        low=0.1,
        high=0.7,
        xlim=("bad", "value"),
        dual=True,
    )

    flaky_value = 0.2
    flaky_interval = 0.1

    class BodyProxy:
        def __init__(self):
            self.bars = [
                BarItem(
                    label="f0",
                    value=flaky_value,
                    interval_low=flaky_interval,
                    interval_high=0.2,
                )
            ]
            self.xlabel = "x"
            self.ylabel = "y"

    body = BodyProxy()

    monkeypatch.setattr("matplotlib.axes.Axes.get_ylim", lambda self: (1.0, 1.0))

    wrapper = render(PlotSpec(header=header, body=body), export_drawn_primitives=True)

    # After the flaky conversions the body primitives should still be exported.
    assert isinstance(wrapper.get("primitives"), list)


def test_render_logs_extent_errors_and_recovers():
    class FlakyFloat:
        def __init__(self, value: float, fail_on: int = 2):
            self.value = value
            self.calls = 0
            self.fail_on = fail_on

        def __float__(self) -> float:
            self.calls += 1
            if self.calls == self.fail_on:
                raise ValueError("bad value")
            return float(self.value)

    header = IntervalHeaderSpec(pred=0.3, low=0.1, high=0.5, dual=False)
    bars = [
        BarItem(
            label="problematic",
            value=FlakyFloat(0.05),
            interval_low=FlakyFloat(-0.02, fail_on=3),
            interval_high=FlakyFloat(0.02, fail_on=4),
            instance_value="same",
        ),
        BarItem(label="stable", value=0.0, interval_low=-0.1, interval_high=0.1),
    ]

    spec = PlotSpec(header=header, body=BarHPanelSpec(bars=bars))
    wrapper = render(spec, export_drawn_primitives=True)

    assert "solids" in wrapper or "overlays" in wrapper


def test_render_header_only_with_flaky_prediction():
    class FlakyNumber:
        def __init__(self, value: float):
            self.value = value
            self.calls = 0

        def __float__(self) -> float:
            self.calls += 1
            if self.calls == 1:
                raise ValueError("first call fails")
            return float(self.value)

        def __rsub__(self, other: float) -> float:
            return float(other) - float(self.value)

    header = IntervalHeaderSpec(pred=FlakyNumber(0.6), low=0.3, high=0.8, dual=True)
    wrapper = render(PlotSpec(header=header), export_drawn_primitives=True)

    assert "header" in wrapper


def test_render_body_without_solid_flags(monkeypatch):
    class SimpleBar:
        def __init__(self):
            self.label = "simple"
            self.value = 0.15
            self.interval_low = -0.05
            self.interval_high = 0.2
            self.instance_value = "inst"

    class SimpleBody:
        def __init__(self):
            self.bars = [SimpleBar()]
            self.xlabel = "X"
            self.ylabel = "Y"

    header = IntervalHeaderSpec(pred=0.4, low=0.1, high=0.7, dual=True)

    wrapper = render(PlotSpec(header=header, body=SimpleBody()), export_drawn_primitives=True)

    assert isinstance(wrapper.get("primitives"), list)


def test_render_accepts_non_dataclass_spec(monkeypatch):
    header = IntervalHeaderSpec(pred=0.5, low=0.2, high=0.7)
    body = BarHPanelSpec(bars=[BarItem(label="b", value=0.1)])

    class SpecProxy:
        def __init__(self):
            self.title = "proxy"
            self.figure_size = None
            self.header = header
            self.body = body

    result = render(SpecProxy(), export_drawn_primitives=True)

    assert result["plot_spec"]["title"] == "proxy"


def test_render_regression_header_invalid_xlim():
    header = IntervalHeaderSpec(pred=0.4, low=0.2, high=0.6, dual=False, xlim=("bad", None))
    wrapper = render(PlotSpec(header=header), export_drawn_primitives=True)
    assert "primitives" in wrapper


def test_render_body_flaky_header_prediction():
    class FlakyNumber:
        def __init__(self, value: float):
            self.value = value
            self.calls = 0

        def __float__(self) -> float:
            self.calls += 1
            if self.calls == 1:
                raise ValueError("float failed once")
            return float(self.value)

        def __rsub__(self, other: float) -> float:
            return float(other) - float(self.value)

    header = IntervalHeaderSpec(pred=FlakyNumber(0.6), low=0.2, high=0.8, dual=True)
    body = BarHPanelSpec(bars=[BarItem(label="f", value=0.1, interval_low=0.05, interval_high=0.2)])
    wrapper = render(PlotSpec(header=header, body=body), export_drawn_primitives=True)
    assert wrapper["header"]["positive"]["solid"]


def test_render_dual_twin_axis_equal_limits(monkeypatch):
    header = IntervalHeaderSpec(pred=0.5, low=0.2, high=0.8, dual=True)
    bars = [BarItem(label="f", value=0.2, instance_value="1", interval_low=0.1, interval_high=0.3)]
    body = BarHPanelSpec(bars=bars, xlabel="x")

    monkeypatch.setattr("matplotlib.axes.Axes.get_ylim", lambda self: (1.0, 1.0))

    wrapper = render(PlotSpec(header=header, body=body), export_drawn_primitives=True)
    assert "overlays" in wrapper


def test_render_regression_twin_axis_equal_limits(monkeypatch):
    header = IntervalHeaderSpec(pred=0.3, low=0.1, high=0.5, dual=False)
    bars = [
        BarItem(label="f", value=0.05, interval_low=-0.02, interval_high=0.08, instance_value="1"),
        BarItem(label="g", value=-0.03, instance_value="2"),
    ]
    body = BarHPanelSpec(bars=bars)

    monkeypatch.setattr("matplotlib.axes.Axes.get_ylim", lambda self: (2.0, 2.0))

    wrapper = render(PlotSpec(header=header, body=body), export_drawn_primitives=True)
    assert "solids" in wrapper or "overlays" in wrapper
