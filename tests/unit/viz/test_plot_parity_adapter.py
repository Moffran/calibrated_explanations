import os

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib

matplotlib.use("Agg", force=True)
import pytest

from calibrated_explanations.viz import matplotlib_adapter as mpl_adapter
from calibrated_explanations.viz import (
    REGRESSION_BAR_COLOR,
    REGRESSION_BASE_COLOR,
    build_global_plotspec,
)

from tests.unit.viz.test_plot_parity_fixtures import (
    factual_probabilistic_conjunction_multiline,
    factual_probabilistic_no_uncertainty,
    factual_probabilistic_zero_crossing,
    factual_probabilistic_multiclass,
    factual_regression_interval,
    alternative_probabilistic_cross_05,
    alternative_probabilistic_both_below_05,
    alternative_probabilistic_feature_cross_05,
    global_probabilistic_multiclass,
    triangular_probabilistic,
)


REG_BAR_COLOR = REGRESSION_BAR_COLOR
REG_BASE_COLOR = REGRESSION_BASE_COLOR
pytestmark = pytest.mark.viz


def test_factual_probabilistic_exports_header_and_contribution_semantics():
    spec = factual_probabilistic_zero_crossing()
    assert spec.header is not None
    spec.header.neg_caption = "P(y=negative)"
    spec.header.pos_caption = "P(y=positive)"

    primitives = mpl_adapter.render(spec, export_drawn_primitives=True)

    assert primitives["plot_spec"]["header"]["neg_caption"] == "P(y=negative)"
    assert primitives["plot_spec"]["header"]["pos_caption"] == "P(y=positive)"
    assert set(primitives["header"]) == {"negative", "positive"}

    base_interval = primitives["base_interval"]["body"]
    assert base_interval["x0"] < 0.0 < base_interval["x1"]

    split_overlays = [item for item in primitives["overlays"] if item["index"] == 0]
    assert len(split_overlays) == 2
    assert any(item["x0"] < 0.0 and item["x1"] == 0.0 for item in split_overlays)
    assert any(item["x0"] == 0.0 and item["x1"] > 0.0 for item in split_overlays)


def test_factual_regression_exports_legacy_sign_colors_and_axis_meaning():
    spec = factual_regression_interval()
    primitives = mpl_adapter.render(spec, export_drawn_primitives=True)

    assert primitives["plot_spec"]["header"]["xlabel"] == "Prediction interval"
    assert primitives["plot_spec"]["body"]["xlabel"] == "Feature weights"

    solids = {item["index"]: item for item in primitives["solids"]}
    assert solids[0]["color"] == "b"
    assert solids[1]["color"] == "r"


def test_alternative_probabilistic_cross_primitives():
    spec = alternative_probabilistic_cross_05()
    assert spec.header is None
    primitives = mpl_adapter.render(spec, export_drawn_primitives=True)
    assert not primitives.get("header")
    overlays = primitives.get("overlays", [])
    assert isinstance(overlays, list) and len(overlays) >= 1
    indices = {item.get("index") for item in overlays}
    # Base interval should be present (index -1) alongside feature overlays (>=0)
    assert -1 in indices
    assert any(idx is not None and idx >= 0 for idx in indices)
    assert primitives["base_interval"]["body"]["x0"] == pytest.approx(0.45)
    assert primitives["base_interval"]["body"]["x1"] == pytest.approx(0.65)
    assert spec.body is not None
    assert spec.body.xlim == (0.0, 1.0)


def test_triangular_plotspec_exports_legacy_direction_and_axis_semantics():
    spec = triangular_probabilistic()

    primitives = mpl_adapter.render(spec, export_drawn_primitives=True)

    assert primitives["triangle_background"]["type"] == "triangle_background"
    quiver = next(item for item in primitives["primitives"] if item["type"] == "quiver")
    assert quiver["coords"]["x"] == pytest.approx(0.7)
    assert quiver["coords"]["y"] == pytest.approx(0.05)
    assert quiver["coords"]["dx"] == pytest.approx(-0.1)
    assert quiver["coords"]["dy"] == pytest.approx(-0.03)

    axes = next(item for item in primitives["primitives"] if item["id"] == "triangle.axes")
    assert axes["style"]["xlabel"] == "Probability"
    assert axes["style"]["ylabel"] == "Uncertainty"


def test_global_plotspec_exports_class_conditioning_and_threshold_labels():
    classified = build_global_plotspec(
        title="global-threshold",
        proba=[0.2, 0.8],
        predict=None,
        low=[0.1, 0.7],
        high=[0.3, 0.9],
        uncertainty=[0.2, 0.2],
        y_test=[20.0, 5.0],
        threshold=10.0,
        class_labels=None,
        is_regularized=True,
    )

    thresholded = mpl_adapter.render(classified, export_drawn_primitives=True)
    axes = next(item for item in thresholded["primitives"] if item["id"] == "global.axes")
    assert axes["style"]["xlabel"] == "Probability of Y < 10.0"

    multiclass = build_global_plotspec(
        title="global-classes",
        proba=[[0.1, 0.9], [0.8, 0.2]],
        predict=None,
        low=[[0.0, 0.8], [0.7, 0.1]],
        high=[[0.2, 1.0], [0.9, 0.3]],
        uncertainty=[[0.1, 0.2], [0.1, 0.2]],
        y_test=[1, 0],
        class_labels={0: "no", 1: "yes"},
        is_regularized=True,
    )

    class_conditioned = mpl_adapter.render(multiclass, export_drawn_primitives=True)
    labels = {
        item["style"]["label"]
        for item in class_conditioned["primitives"]
        if item["type"] == "scatter"
    }
    assert labels == {"Y = no", "Y = yes"}


def test_factual_probabilistic_multiclass_neg_caption_uses_not_equal_notation():
    """Neg caption carries '!=' when explicit caption is supplied by caller."""
    spec = factual_probabilistic_multiclass()
    assert spec.header is not None
    assert spec.header.neg_caption == "P(y!=class2)"
    assert spec.header.pos_caption == "P(y=class2)"
    primitives = mpl_adapter.render(spec, export_drawn_primitives=True)
    assert primitives["plot_spec"]["header"]["neg_caption"] == "P(y!=class2)"
    assert "negative" in primitives["header"]
    assert "positive" in primitives["header"]


def test_factual_probabilistic_feature_order_preserved():
    """feature_order attribute reflects the order of bars in the body."""
    spec = factual_probabilistic_no_uncertainty()
    assert spec.feature_order is not None
    assert len(spec.feature_order) == len(spec.body.bars)
    assert list(spec.feature_order) == list(range(len(spec.body.bars)))


def test_conjunction_multiline_labels_expand_rendered_height():
    """Multiline conjunction labels expand rendered figure height for review legibility."""
    spec = factual_probabilistic_conjunction_multiline()

    figure = mpl_adapter.render(spec, show=False, return_fig=True)

    try:
        height = float(figure.get_size_inches()[1])
        assert height == pytest.approx(6.0)
    finally:
        matplotlib.pyplot.close(figure.number)


def test_alternative_probabilistic_both_below_05_single_segment_per_bar():
    """Bars entirely below 0.5 produce a single segment (no 0.5-split)."""
    spec = alternative_probabilistic_both_below_05()
    for bar in spec.body.bars:
        assert bar.segments is not None, "segments must be populated"
        assert (
            len(bar.segments) == 1
        ), f"bar '{bar.label}' expected 1 segment, got {len(bar.segments)}"
    # base_segments for predict=0.25, low=0.15, high=0.35 — all below 0.5 → 1 segment
    assert len(spec.body.base_segments) == 1


def test_alternative_probabilistic_feature_cross_05_two_segments_different_colors():
    """A feature bar whose interval crosses 0.5 must produce two segments with distinct colors."""
    spec = alternative_probabilistic_feature_cross_05()
    # bar index 0: predict=0.5, low=0.3, high=0.7 — crosses 0.5
    cross_bar = spec.body.bars[0]
    assert cross_bar.segments is not None
    assert len(cross_bar.segments) == 2, "cross-0.5 bar must split into two segments"
    left_color, right_color = cross_bar.segments[0].color, cross_bar.segments[1].color
    assert (
        left_color != right_color
    ), "left and right segments must have distinct colors at 0.5-split"
    # segment boundary must be at 0.5
    assert cross_bar.segments[0].high == pytest.approx(0.5)
    assert cross_bar.segments[1].low == pytest.approx(0.5)
    # bar index 1: low=0.75, high=0.85 — entirely above 0.5 → single segment
    above_bar = spec.body.bars[1]
    assert len(above_bar.segments) == 1


def test_alternative_regression_base_line_and_interval_semantics():
    """Alternative regression body carries base_lines (prediction marker) and base_segments."""
    from tests.unit.viz.test_plot_parity_fixtures import alternative_regression_interval

    spec = alternative_regression_interval()
    assert spec.body is not None
    assert spec.body.base_segments is not None and len(spec.body.base_segments) > 0
    assert spec.body.base_lines is not None and len(spec.body.base_lines) > 0
    # base_line x-value should be the prediction center
    base_x = spec.body.base_lines[0][0]
    assert base_x == pytest.approx(1.2), f"base_line x expected 1.2 (predict), got {base_x}"


def test_global_probabilistic_multiclass_saved(tmp_path):
    spec = global_probabilistic_multiclass()
    # builder returns canonical dataclass; save behavior is configured on the
    # dataclass and converted at serializer boundary.
    assert spec.global_entries is not None
    spec.save_behavior.path = str(tmp_path)
    assert spec.save_behavior is not None
    assert tuple(spec.save_behavior.default_exts or ()) == ("svg", "png")


def test_adapter_returns_normalized_and_legacy_for_plotspec():
    """Ensure adapter returns normalized and legacy keys for a PlotSpec dataclass."""
    # use factual_probabilistic_no_uncertainty (returns a PlotSpec dataclass)
    spec = factual_probabilistic_no_uncertainty()
    wrapper = mpl_adapter.render(spec, export_drawn_primitives=True)
    # wrapper must include top-level plot_spec and primitives list
    assert isinstance(wrapper, dict)
    assert "plot_spec" in wrapper and "primitives" in wrapper
    # legacy keys like solids/overlays/header should also be present (may be empty)
    assert any(k in wrapper for k in ("solids", "overlays", "header", "base_interval"))


def testplot_triangular_delegates_to_adapter(monkeypatch, tmp_path):
    """Ensure `plot_triangular` delegates to builder+adapter and handles save_ext."""
    from calibrated_explanations.viz import plots as _plots

    calls = []

    def fake_render(spec, *, show=False, save_path=None, **kwargs):
        calls.append({"spec": spec, "show": show, "save_path": save_path})
        return {}

    monkeypatch.setattr("calibrated_explanations.viz.matplotlib_adapter.render", fake_render)

    # prepare simple numeric arrays for triangular plot
    proba = [0.2]
    uncertainty = [0.1]
    rule_proba = [0.3]
    rule_uncertainty = [0.05]

    # call with show=False and no save_ext -> should no-op and not call adapter.render
    _plots.plot_triangular(
        None,
        proba,
        uncertainty,
        rule_proba,
        rule_uncertainty,
        1,
        "t",
        None,
        False,
        save_ext=None,
        use_legacy=False,
    )
    assert len(calls) == 0

    # Reset and call with save_ext to trigger adapter.save behavior
    calls.clear()
    _plots.plot_triangular(
        None,
        proba,
        uncertainty,
        rule_proba,
        rule_uncertainty,
        1,
        "t",
        str(tmp_path) + "/",
        False,
        save_ext=["png"],
        use_legacy=False,
    )  # noqa: E501
    # adapter.render should be invoked for initial render + each save ext
    assert len(calls) >= 2
