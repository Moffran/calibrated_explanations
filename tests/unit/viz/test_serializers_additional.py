from calibrated_explanations.viz.serializers import plotspec_to_dict
from calibrated_explanations.viz.plotspec import PlotSpec
from calibrated_explanations.viz.plotspec import BarHPanelSpec, BarItem, IntervalHeaderSpec


def test_plotspec_to_dict_feature_order_and_style():
    header = IntervalHeaderSpec(pred=0.5, low=0.2, high=0.8)
    body = BarHPanelSpec(bars=[BarItem(label="f0", value=0.1)])
    ps = PlotSpec(title="t", header=header, body=body)

    d = plotspec_to_dict(ps)

    # feature_order should be converted to integers (even if empty/default)
    assert isinstance(d.get("plot_spec", {}).get("feature_order", []), list)
    # default style key exists when not provided
    assert "style" in d.get("plot_spec", {})


def test_plotspec_to_dict_serializes_plotspec():
    # ensure function serializes a PlotSpec-like object
    header = IntervalHeaderSpec(pred=0.5, low=0.2, high=0.8)
    body = BarHPanelSpec(bars=[BarItem(label="f1", value=0.2)])
    ps = PlotSpec(title="raw", header=header, body=body)

    out = plotspec_to_dict(ps)

    # title is placed at top-level in the envelope
    assert out.get("title") == "raw"
