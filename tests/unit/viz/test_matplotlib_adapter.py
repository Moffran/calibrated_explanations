import sys
import types

import numpy as np
import pytest

from calibrated_explanations.viz import matplotlib_adapter as ma
from calibrated_explanations.viz.plotspec import (
    BarHPanelSpec,
    BarItem,
    IntervalHeaderSpec,
    PlotSpec,
)


CREATED_FIGURES = []


class _FakeSpine:
    def __init__(self):
        self.visible = True

    def set_visible(self, flag):
        self.visible = bool(flag)


class _BadLabel:
    def __str__(self):  # pragma: no cover - trivial
        raise ValueError("label failed")


class _FakeAxes:
    def __init__(self):
        self.calls = []
        self._ylim = (0.0, 0.0)
        self.spines = {key: _FakeSpine() for key in ("top", "right", "bottom", "left")}

    def fill_betweenx(self, *args, **kwargs):
        self.calls.append(("fill_betweenx", args, kwargs))

    def plot(self, *args, **kwargs):
        self.calls.append(("plot", args, kwargs))

    def set_xlim(self, *args, **kwargs):
        self.calls.append(("set_xlim", args, kwargs))

    def set_xlabel(self, *args, **kwargs):
        self.calls.append(("set_xlabel", args, kwargs))

    def set_xticks(self, *args, **kwargs):
        self.calls.append(("set_xticks", args, kwargs))

    def set_yticks(self, *args, **kwargs):
        self.calls.append(("set_yticks", args, kwargs))

    def set_yticklabels(self, *args, **kwargs):
        self.calls.append(("set_yticklabels", args, kwargs))

    def set_ylabel(self, *args, **kwargs):
        self.calls.append(("set_ylabel", args, kwargs))

    def set_ylim(self, *args, **kwargs):
        if args:
            self._ylim = args[0]
        self.calls.append(("set_ylim", args, kwargs))

    def get_ylim(self):
        return self._ylim

    def twinx(self):
        twin = _FakeAxes()
        self.calls.append(("twinx", (), {}))
        return twin


class _FakeGridSpec:
    def __init__(self, figure, nrows, ncols, height_ratios=None):
        self.figure = figure
        self.nrows = nrows
        self.ncols = ncols
        self.height_ratios = height_ratios

    def __getitem__(self, item):
        return item


class _FakeFigure:
    def __init__(self, figsize=None):
        self.figsize = figsize
        self.axes = []
        self.closed = False
        self.suptitle_args = None
        self.calls = []
        CREATED_FIGURES.append(self)

    def add_gridspec(self, nrows, ncols, height_ratios=None):
        return _FakeGridSpec(self, nrows, ncols, height_ratios)

    def add_subplot(self, spec):
        ax = _FakeAxes()
        self.axes.append(ax)
        return ax

    def suptitle(self, *args, **kwargs):
        self.suptitle_args = (args, kwargs)

    def tight_layout(self, *args, **kwargs):
        self.calls.append(("tight_layout", args, kwargs))

    def savefig(self, *args, **kwargs):
        self.calls.append(("savefig", args, kwargs))


@pytest.fixture(autouse=True)
def _patch_matplotlib(monkeypatch):
    CREATED_FIGURES.clear()
    fake_pyplot = types.ModuleType("pyplot")

    def figure(*, figsize=None):
        return _FakeFigure(figsize)

    def close(fig=None):
        if fig is not None:
            fig.closed = True

    def show(*args, **kwargs):
        fake_pyplot.last_show = (args, kwargs)

    fake_pyplot.figure = figure
    fake_pyplot.close = close
    fake_pyplot.show = show

    fake_matplotlib = types.ModuleType("matplotlib")
    fake_matplotlib.pyplot = fake_pyplot
    sys.modules["matplotlib"] = fake_matplotlib
    sys.modules["matplotlib.pyplot"] = fake_pyplot

    monkeypatch.setattr(ma, "_require_mpl", lambda: None)
    style = {
        "figure": {"width": 7.5},
        "colors": {
            "alpha": 0.35,
            "positive": "#1976d2",
            "negative": "#d32f2f",
            "regression": "#455a64",
        },
    }
    monkeypatch.setattr(ma, "_setup_style", lambda _: style)

    yield

    sys.modules.pop("matplotlib", None)
    sys.modules.pop("matplotlib.pyplot", None)


def test_render_noop_when_no_output_requested():
    spec = PlotSpec(title="unused")
    assert ma.render(spec) is None


def test_render_headless_skips_matplotlib(monkeypatch):
    """When nothing is requested the adapter must not attempt to import matplotlib."""

    spec = PlotSpec(title="skip")

    def _boom():  # pragma: no cover - should not execute
        raise AssertionError("_require_mpl should not be called")

    monkeypatch.setattr(ma, "_require_mpl", _boom)
    assert ma.render(spec) is None


def test_render_short_circuits_before_import(monkeypatch):
    """The adapter should not import matplotlib when nothing is rendered."""

    calls: list[bool] = []

    def marker():  # pragma: no cover - simple spy
        calls.append(True)

    monkeypatch.setattr(ma, "_require_mpl", marker)
    spec = PlotSpec(title="unused")
    assert ma.render(spec) is None
    assert not calls


def test_render_saves_before_show(monkeypatch):
    """Saving to disk must occur before the adapter requests a GUI show."""

    header = IntervalHeaderSpec(pred=0.5, low=0.2, high=0.8, dual=False)
    body = BarHPanelSpec(
        bars=[BarItem(label="f0", value=0.3, instance_value=0.1)],
        xlabel="Contribution",
        ylabel="Features",
    )
    spec = PlotSpec(title="order", header=header, body=body)

    events: list[str] = []

    fake_pyplot = sys.modules["matplotlib.pyplot"]
    original_show = fake_pyplot.show

    def spy_show(*args, **kwargs):  # pragma: no cover - spy helper
        events.append("show")
        return original_show(*args, **kwargs)

    original_savefig = _FakeFigure.savefig

    def spy_savefig(self, *args, **kwargs):  # pragma: no cover - spy helper
        events.append("save")
        return original_savefig(self, *args, **kwargs)

    monkeypatch.setattr(fake_pyplot, "show", spy_show)
    monkeypatch.setattr(_FakeFigure, "savefig", spy_savefig)

    ma.render(spec, show=True, save_path="out.png")

    assert events[0] == "save"
    assert events[1] == "show"


def test_auto_height_falls_back_when_label_conversion_fails():
    header = IntervalHeaderSpec(pred=0.5, low=0.2, high=0.8, dual=False)
    body = BarHPanelSpec(
        bars=[BarItem(label=_BadLabel(), value=0.1), BarItem(label="ok", value=-0.2)],
    )
    spec = PlotSpec(title="height", header=header, body=body)

    fig = ma.render(spec, return_fig=True)

    assert isinstance(fig, _FakeFigure)
    # Height should stay finite even after the label exception path executes
    assert fig.figsize[1] >= 3.0


def test_render_normalizes_dict_payloads():
    tri = {
        "plot_spec": {
            "kind": "triangular",
            "triangular": {"angles": np.array([0.0, 0.5, 1.0])},
        }
    }
    tri_result = ma.render(tri, export_drawn_primitives=True)
    assert tri_result["triangle_background"]["type"] == "triangle_background"
    assert any(p["type"] == "quiver" for p in tri_result["primitives"])

    global_payload = {
        "plot_spec": {
            "kind": "global_scatter",
            "global_entries": {
                "proba": [[0.1, 0.9], [0.25, 0.75]],
                "uncertainty": [[0.05, 0.15], [0.02, 0.12]],
            },
            "save_behavior": {"default_exts": ["png", "pdf"]},
        }
    }
    glob_result = ma.render(global_payload, export_drawn_primitives=True)
    scatter_ids = [p["id"] for p in glob_result["primitives"] if p["type"] == "scatter"]
    assert scatter_ids == ["global.scatter.0", "global.scatter.1"]
    save_ids = [p["id"] for p in glob_result["primitives"] if p["type"] == "save_fig"]
    assert save_ids == ["save.png", "save.pdf"]

    # Scalar probabilities hit the non-multiclass branch and still emit scatter primitives
    scalar_payload = {
        "plot_spec": {
            "kind": "global_scatter",
            "global_entries": {"proba": [0.2, 0.8], "uncertainty": [0.1, 0.05]},
        }
    }
    scalar_result = ma.render(scalar_payload, export_drawn_primitives=True)
    assert [p["coords"]["x"] for p in scalar_result["primitives"] if p["type"] == "scatter"] == [0.2, 0.8]

    # Trigger the defensive fallback path when casting to float fails
    bad_global = {
        "plot_spec": {
            "kind": "global_scatter",
            "global_entries": {"proba": ["bad"], "uncertainty": ["bad"]},
        }
    }
    bad_result = ma.render(bad_global, export_drawn_primitives=True)
    assert any(p["id"] == "global.scatter.summary" for p in bad_result["primitives"])


def test_dual_body_respects_item_level_solid_flags():
    header = IntervalHeaderSpec(pred=0.7, low=0.4, high=0.9, dual=True)
    body = types.SimpleNamespace(
        bars=[
            BarItem(
                label="unstable",
                value=1.1,
                interval_low=0.8,
                interval_high=1.3,
                solid_on_interval_crosses_zero=False,
            )
        ],
        xlabel=None,
        ylabel=None,
    )
    spec = PlotSpec(title="flaky", header=header, body=body)

    result = ma.render(spec, export_drawn_primitives=True)

    solids = result.get("solids", [])
    assert solids and solids[0]["index"] == 0
    assert solids[0]["x0"] == 0.0 and solids[0]["x1"] > 0.0


def test_dual_body_zero_extent_sets_padding():
    header = IntervalHeaderSpec(pred=0.5, low=0.4, high=0.6, dual=True)
    zero_body = types.SimpleNamespace(
        bars=[
            BarItem(label="zero_a", value=0.0, interval_low=0.0, interval_high=0.0),
            BarItem(label="zero_b", value=0.0, interval_low=0.0, interval_high=0.0),
        ],
        xlabel=None,
        ylabel=None,
    )
    spec = PlotSpec(title="dual-zero", header=header, body=zero_body)

    ma.render(spec, export_drawn_primitives=True)

    axis = CREATED_FIGURES[-1].axes[-1]
    xlim_calls = [call for call in axis.calls if call[0] == "set_xlim"]
    assert xlim_calls


class _SimpleBar:
    def __init__(self, label, value, low, high):
        self.label = label
        self.value = value
        self.interval_low = low
        self.interval_high = high
        self.instance_value = None


def test_dual_body_defaults_suppress_when_flags_missing():
    header = IntervalHeaderSpec(pred=0.4, low=0.2, high=0.6, dual=True)
    body = types.SimpleNamespace(
        bars=[_SimpleBar("no-flag", 0.3, -0.2, 0.5)],
        xlabel=None,
        ylabel=None,
    )
    spec = PlotSpec(title="dual-default", header=header, body=body)

    result = ma.render(spec, export_drawn_primitives=True)

    solids = result.get("solids", [])
    assert not solids  # solid suppressed because default flag is True


def test_render_dual_header_exports_primitives():
    header = IntervalHeaderSpec(
        pred=0.6,
        low=0.2,
        high=0.9,
        xlim=(0.5, 0.5),
        xlabel="probability",
        ylabel="target",
        dual=True,
    )
    body = BarHPanelSpec(
        bars=[
            BarItem(
                label="first\nline",
                value=0.65,
                interval_low=0.55,
                interval_high=0.75,
                instance_value=1.0,
            ),
            BarItem(
                label="second",
                value=0.92,
                interval_low=0.8,
                interval_high=0.94,
                solid_on_interval_crosses_zero=False,
                instance_value=0.5,
            ),
            BarItem(label="plain", value=-0.25),
        ],
        xlabel="Contribution",
        ylabel="Feature",
        solid_on_interval_crosses_zero=True,
    )
    spec = PlotSpec(title="Dual", header=header, body=body)
    result = ma.render(spec, export_drawn_primitives=True)

    assert isinstance(result["primitives"], list) and result["primitives"]
    assert set(result.keys()) >= {"solids", "overlays", "base_interval", "header"}
    header_pos = result["header"]["positive"]
    assert header_pos["solid"][0] == 0.0
    overlays = result["overlays"]
    assert any(o["color"] in ["#1976d2", "#d32f2f"] for o in overlays)
    # Ensure body primitives reflect contribution coordinates (span values around zero)
    assert any(s["x0"] < 0.0 for s in result["solids"])
    assert result["base_interval"]["body"]["x0"] < result["base_interval"]["body"]["x1"]

    # Rendering with explicit class labels exercises the alternate ytick branch
    labeled_header = IntervalHeaderSpec(
        pred=0.55,
        low=0.2,
        high=0.8,
        xlim=(0.0, 1.0),
        xlabel="probability",
        ylabel=None,
        dual=True,
        neg_label="no",
        pos_label="yes",
    )
    labeled_spec = PlotSpec(title="Dual labeled", header=labeled_header, body=body)
    labeled_result = ma.render(labeled_spec, export_drawn_primitives=True)
    assert "header" in labeled_result


def test_render_returns_figure_for_single_header():
    header = IntervalHeaderSpec(
        pred=10.0,
        low=8.0,
        high=12.0,
        xlabel="value",
        ylabel="prediction",
        dual=False,
    )
    spec = PlotSpec(title="Regression", figure_size=(4.0, 2.5), header=header, body=None)
    fig = ma.render(spec, return_fig=True, draw_intervals=False)
    assert isinstance(fig, _FakeFigure)
    assert not fig.closed


def test_render_regression_body_handles_intervals(tmp_path):
    header = IntervalHeaderSpec(
        pred=1.5,
        low=1.0,
        high=2.0,
        xlim=(2.0, 2.0),
        xlabel="score",
        ylabel="outcome",
        dual=False,
    )
    body = BarHPanelSpec(
        bars=[
            BarItem(
                label="alpha",
                value=1.7,
                interval_low=1.2,
                interval_high=1.9,
                instance_value=3.14,
            ),
            BarItem(
                label="beta",
                value=1.1,
                interval_low=0.8,
                interval_high=1.4,
            ),
            BarItem(
                label="gamma",
                value=0.0,
                interval_low=1.0,
                interval_high=2.0,
            ),
        ],
        xlabel="Contribution",
        ylabel="Feature",
        solid_on_interval_crosses_zero=False,
    )
    spec = PlotSpec(title="Regression body", header=header, body=body)
    save_path = tmp_path / "plot.png"
    result = ma.render(
        spec,
        export_drawn_primitives=True,
        save_path=str(save_path),
    )

    assert save_path.name == "plot.png"
    assert result.get("base_interval", {}).get("body") is None
    # Header-matching intervals are rendered alongside others
    labels = [p.get("index") for p in result.get("solids", [])]
    assert set(labels) >= {0, 1, 2}
    overlay_indices = {p.get("index") for p in result.get("overlays", [])}
    assert {0, 1, 2}.issubset(overlay_indices)

    # When solids are suppressed for cross-zero intervals, the dedicated branch emits split overlays
    cross_body = BarHPanelSpec(
        bars=[
            BarItem(
                label="alpha",
                value=1.7,
                interval_low=1.2,
                interval_high=1.9,
            ),
            BarItem(
                label="beta",
                value=1.1,
                interval_low=0.8,
                interval_high=1.4,
            ),
        ],
        solid_on_interval_crosses_zero=True,
    )
    cross_spec = PlotSpec(title="Regression body", header=header, body=cross_body)
    cross_result = ma.render(cross_spec, export_drawn_primitives=True, show=True)
    body_primitives = [item for item in cross_result["primitives"] if item.get("axis_id") == "body"]
    assert body_primitives
    cross_overlays = cross_result.get("overlays", [])
    # New adapter draws regression overlays directly in value space; just
    # ensure overlays are present and well-ordered, not necessarily crossing 0.
    assert cross_overlays and all(o.get("x0") <= o.get("x1") for o in cross_overlays)


def test_regression_body_respects_item_solid_flags_and_extent_padding():
    header = IntervalHeaderSpec(pred=0.0, low=-0.1, high=0.1, dual=False)
    zero_body = types.SimpleNamespace(
        bars=[
            BarItem(label="zero_a", value=0.0, interval_low=0.0, interval_high=0.0),
            BarItem(label="zero_b", value=0.0, interval_low=0.0, interval_high=0.0),
        ],
        xlabel=None,
        ylabel=None,
    )
    zero_spec = PlotSpec(title="padding", header=header, body=zero_body)

    ma.render(zero_spec, export_drawn_primitives=True)

    skip_body = types.SimpleNamespace(
        bars=[
            BarItem(label="baseline", value=0.2, interval_low=-0.1, interval_high=0.1),
            BarItem(
                label="skip",
                value=0.5,
                interval_low=-0.1,
                interval_high=0.1,
                solid_on_interval_crosses_zero=False,
            ),
        ],
        xlabel=None,
        ylabel=None,
    )
    skip_spec = PlotSpec(title="skip", header=header, body=skip_body)

    skip_result = ma.render(skip_spec, export_drawn_primitives=True)

    solids = skip_result.get("solids", [])
    assert any(item.get("index") == 1 for item in solids)


def test_regression_colors_without_uncertainty_match_legacy_palette():
    header = IntervalHeaderSpec(pred=0.0, low=-1.0, high=1.0, dual=False)
    body = BarHPanelSpec(
        bars=[
            BarItem(label="positive", value=0.4),
            BarItem(label="negative", value=-0.6),
        ],
        xlabel="Contribution",
        ylabel="Feature",
    )
    spec = PlotSpec(title="regression-colors", header=header, body=body)

    result = ma.render(spec, export_drawn_primitives=True)

    solids = {item["index"]: item for item in result.get("solids", [])}
    assert 0 in solids and 1 in solids
    assert solids[0]["color"] == "b"
    assert solids[1]["color"] == "r"
    assert result.get("base_interval", {}).get("body") is None


def test_regression_interval_colors_match_legacy_palette():
    header = IntervalHeaderSpec(pred=0.0, low=-0.5, high=0.5, dual=False)
    body = BarHPanelSpec(
        bars=[
            BarItem(
                label="positive",
                value=0.5,
                interval_low=0.2,
                interval_high=0.7,
                solid_on_interval_crosses_zero=False,
            ),
            BarItem(
                label="negative",
                value=-0.3,
                interval_low=-0.6,
                interval_high=-0.1,
                solid_on_interval_crosses_zero=False,
            ),
        ],
        xlabel="Contribution",
        ylabel="Feature",
        solid_on_interval_crosses_zero=False,
    )
    spec = PlotSpec(title="regression-interval-colors", header=header, body=body)

    result = ma.render(spec, export_drawn_primitives=True)

    solids = {item["index"]: item for item in result.get("solids", [])}
    assert 0 in solids and 1 in solids
    assert solids[0]["color"] == "b"
    assert solids[1]["color"] == "r"

    overlays = {item["index"]: item for item in result.get("overlays", [])}
    assert overlays.get(0, {}).get("color") == "b"
    assert overlays.get(1, {}).get("color") == "r"

    assert result.get("base_interval", {}).get("body") is None


def test_export_guard_raises_on_probability_coordinate_mismatch():
    header = IntervalHeaderSpec(pred=1.3, low=1.1, high=1.4, dual=True)
    body = BarHPanelSpec(bars=[BarItem(label="bad", value=1.2, interval_low=1.1, interval_high=1.3)])
    spec = PlotSpec(title="assert", header=header, body=body)

    with pytest.raises(AssertionError):
        ma.render(spec, export_drawn_primitives=True)


class _NamespaceSpec:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def test_render_handles_non_dataclass_spec_payload():
    header = IntervalHeaderSpec(pred=0.2, low=0.1, high=0.3, dual=False)
    body = BarHPanelSpec(bars=[BarItem(label="ns", value=0.1)])
    spec = _NamespaceSpec(title="ns", header=header, body=body, figure_size=(4, 3))

    result = ma.render(spec, export_drawn_primitives=True)

    assert isinstance(result["plot_spec"], dict)


def test_render_closes_when_only_show_requested():
    header = IntervalHeaderSpec(pred=0.1, low=0.0, high=0.2, dual=False)
    body = BarHPanelSpec(bars=[BarItem(label="show", value=0.2)])
    spec = PlotSpec(title="show", header=header, body=body)

    ma.render(spec, show=True)

    assert CREATED_FIGURES[-1].closed is True
