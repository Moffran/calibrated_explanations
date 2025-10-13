import dataclasses

from calibrated_explanations.viz.plotspec import (
    BarHPanelSpec,
    BarItem,
    IntervalHeaderSpec,
    PlotSpec,
    __all__ as plotspec_all,
)


def test_interval_header_spec_defaults_and_overrides():
    header = IntervalHeaderSpec(
        pred=0.75,
        low=0.25,
        high=0.9,
        xlim=(-1.0, 1.0),
        xlabel="Predicted",
        ylabel="Density",
        dual=False,
        neg_label="class 0",
        pos_label="class 1",
        uncertainty_color="#cccccc",
        uncertainty_alpha=0.42,
    )

    assert dataclasses.asdict(header)["pred"] == 0.75
    assert header.low == 0.25
    assert header.high == 0.9
    assert header.dual is False
    assert header.neg_label == "class 0"
    assert header.pos_label == "class 1"
    assert header.xlim == (-1.0, 1.0)
    assert header.xlabel == "Predicted"
    assert header.ylabel == "Density"
    assert header.uncertainty_color == "#cccccc"
    assert header.uncertainty_alpha == 0.42


def test_bar_item_and_panel_defaults_are_preserved():
    bar = BarItem(
        label="Feature A",
        value=-0.6,
        interval_low=-0.8,
        interval_high=-0.2,
        color_role="negative",
        instance_value=3.14,
        solid_on_interval_crosses_zero=False,
    )

    assert dataclasses.asdict(bar) == {
        "label": "Feature A",
        "value": -0.6,
        "interval_low": -0.8,
        "interval_high": -0.2,
        "color_role": "negative",
        "instance_value": 3.14,
        "solid_on_interval_crosses_zero": False,
    }

    panel = BarHPanelSpec(bars=[bar], xlabel="Contribution", ylabel="Feature")

    assert list(panel.bars) == [bar]
    assert panel.xlabel == "Contribution"
    assert panel.ylabel == "Feature"
    assert panel.solid_on_interval_crosses_zero is True


def test_plotspec_collects_header_and_body():
    header = IntervalHeaderSpec(pred=0.1, low=-0.2, high=0.4)
    bars = [BarItem(label="f0", value=0.3)]
    body = BarHPanelSpec(bars=bars)

    spec = PlotSpec(title="Example", figure_size=(4.0, 3.0), header=header, body=body)

    assert spec.title == "Example"
    assert spec.figure_size == (4.0, 3.0)
    assert spec.header is header
    assert spec.body is body
    # dataclasses support equality which should include nested specs
    assert spec == PlotSpec(title="Example", figure_size=(4.0, 3.0), header=header, body=body)


def test_plotspec_module_exports_are_consistent():
    expected = {"PlotSpec", "IntervalHeaderSpec", "BarHPanelSpec", "BarItem"}
    assert set(plotspec_all) == expected

    module_attrs = {name: globals()[name] for name in plotspec_all}
    for name, value in module_attrs.items():
        assert value.__name__ == name
