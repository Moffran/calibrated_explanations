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
