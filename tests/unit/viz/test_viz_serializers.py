import pytest

from calibrated_explanations.viz.serializers import (
    plotspec_to_dict,
    plotspec_from_dict,
    validate_plotspec,
)
from calibrated_explanations.viz.plotspec import (
    PlotSpec,
    IntervalHeaderSpec,
    BarHPanelSpec,
    BarItem,
)


def test_plotspec_roundtrip_and_validate():
    header = IntervalHeaderSpec(pred=0.3, low=0.1, high=0.9)
    bars = [BarItem(label="a", value=0.1), BarItem(label="b", value=0.2)]
    body = BarHPanelSpec(bars=bars)
    spec = PlotSpec(title="t", header=header, body=body)
    d = plotspec_to_dict(spec)
    assert d["plotspec_version"]
    s2 = plotspec_from_dict(d)
    assert s2.title == "t"


def test_validate_rejects_bad_payload():
    with pytest.raises(ValueError):
        validate_plotspec({})
    with pytest.raises(ValueError):
        validate_plotspec({"plotspec_version": "1.0.0", "body": {"bars": "notalist"}})


def test_interval_header_spec_optional_fields():
    header = IntervalHeaderSpec(
        pred=0.4,
        low=0.2,
        high=0.9,
        xlim=(0.0, 1.0),
        xlabel="prediction",
        ylabel="probability",
        dual=False,
        neg_label="negative",
        pos_label="positive",
        uncertainty_color="#ccc",
        uncertainty_alpha=0.75,
    )
    assert header.dual is False
    assert header.xlabel == "prediction"
    assert header.neg_label == "negative"
    assert header.uncertainty_alpha == 0.75


def test_bar_item_and_panel_configuration():
    items = [
        BarItem(
            label="feature",
            value=0.35,
            interval_low=-0.05,
            interval_high=0.4,
            color_role="positive",
            instance_value=3.14,
            solid_on_interval_crosses_zero=False,
        ),
        BarItem(label="baseline", value=-0.1),
    ]
    panel = BarHPanelSpec(
        bars=items,
        xlabel="Contribution",
        ylabel="Feature",
        solid_on_interval_crosses_zero=False,
    )
    assert panel.bars[0].color_role == "positive"
    assert panel.solid_on_interval_crosses_zero is False


def test_plotspec_all_fields():
    header = IntervalHeaderSpec(pred=0.5, low=0.25, high=0.75)
    panel = BarHPanelSpec(bars=[BarItem(label="feat", value=0.2)])
    spec = PlotSpec(
        title="Example",
        figure_size=(6.0, 4.0),
        header=header,
        body=panel,
    )
    assert spec.figure_size == (6.0, 4.0)
    assert spec.body.bars[0].label == "feat"
