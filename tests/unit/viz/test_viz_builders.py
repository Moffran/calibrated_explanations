import numpy as np
import pytest

from calibrated_explanations.viz.builders import (
    build_regression_bars_spec,
    build_probabilistic_bars_spec,
)


def make_fw(n):
    rng = np.random.default_rng(0)
    vals = rng.normal(0, 0.2, size=n)
    low = vals - rng.uniform(0.05, 0.15, size=n)
    high = vals + rng.uniform(0.05, 0.25, size=n)
    return {"predict": vals, "low": low, "high": high}


def test_build_regression_numeric_path_and_labels():
    vals = np.array([0.1, -0.2, 0.3])
    spec = build_regression_bars_spec(
        title="t",
        predict={"predict": 0.5},
        feature_weights=vals,
        features_to_plot=[0, 1, 2],
        column_names=["a", "b", "c"],
        instance=[10, 20, 30],
        y_minmax=(0.0, 1.0),
        interval=False,
    )
    bars = spec.body.bars
    assert [b.label for b in bars] == ["a", "b", "c"]
    assert [b.instance_value for b in bars] == [10, 20, 30]


def test_build_regression_interval_dict_and_header_xlim():
    fw = make_fw(4)
    spec = build_regression_bars_spec(
        title=None,
        predict={"predict": 0.2, "low": 0.1, "high": 0.3},
        feature_weights=fw,
        features_to_plot=[0, 1, 2, 3],
        column_names=None,
        instance=None,
        y_minmax=(0.0, 1.0),
        interval=True,
    )
    # header xlim should be set when y_minmax provided
    assert spec.header.xlim is not None
    # bars should have interval_low/high set
    for b in spec.body.bars:
        assert b.interval_low is not None and b.interval_high is not None


def test_sort_by_value_and_abs_ordering():
    vals = np.array([0.05, -0.9, 0.5, -0.1])
    spec_val = build_regression_bars_spec(
        title=None,
        predict={"predict": 0.0},
        feature_weights=vals,
        features_to_plot=[0, 1, 2, 3],
        column_names=["f0", "f1", "f2", "f3"],
        instance=None,
        y_minmax=None,
        interval=False,
        sort_by="value",
        ascending=True,
    )
    vals_sorted = [b.value for b in spec_val.body.bars]
    assert vals_sorted == sorted(vals_sorted)

    spec_abs = build_regression_bars_spec(
        title=None,
        predict={"predict": 0.0},
        feature_weights=vals,
        features_to_plot=[0, 1, 2, 3],
        column_names=None,
        instance=None,
        y_minmax=None,
        interval=False,
        sort_by="abs",
        ascending=False,
    )
    mags = [abs(b.value) for b in spec_abs.body.bars]
    assert mags == sorted(mags, reverse=True)


def test_sort_by_width_and_label():
    fw = make_fw(4)
    # width sorting descending
    spec_width = build_regression_bars_spec(
        title=None,
        predict={"predict": 0.0},
        feature_weights=fw,
        features_to_plot=[0, 1, 2, 3],
        column_names=["z", "y", "x", "w"],
        instance=None,
        y_minmax=None,
        interval=True,
        sort_by="width",
        ascending=False,
    )
    widths = [abs(b.interval_high - b.interval_low) for b in spec_width.body.bars]
    assert widths == sorted(widths, reverse=True)

    # label sorting ascending
    spec_label = build_regression_bars_spec(
        title=None,
        predict={"predict": 0.0},
        feature_weights=fw,
        features_to_plot=[2, 0, 1],
        column_names=["b", "a", "c"],
        instance=None,
        y_minmax=None,
        interval=True,
        sort_by="label",
        ascending=True,
    )
    labels = [b.label for b in spec_label.body.bars]
    assert labels == sorted(labels)


def test_probabilistic_builder_color_role_and_dual_header():
    fw = make_fw(3)
    spec = build_probabilistic_bars_spec(
        title="p",
        predict={"predict": 0.6, "low": 0.5, "high": 0.7},
        feature_weights=fw,
        features_to_plot=[0, 1, 2],
        column_names=["a", "b", "c"],
        instance=[1, 2, 3],
        y_minmax=None,
        interval=True,
    )
    # header should be dual for probabilistic builder
    assert getattr(spec.header, "dual", False) is True
    # color role should be 'positive' for positive values and 'negative' otherwise
    roles = [b.color_role for b in spec.body.bars]
    assert all(r in ("positive", "negative") for r in roles)


def test_probabilistic_builder_unit_interval_with_custom_minmax():
    fw = make_fw(2)
    spec = build_probabilistic_bars_spec(
        title="prob",
        predict={"predict": 0.6, "low": 0.45, "high": 0.7},
        feature_weights=fw,
        features_to_plot=[0, 1],
        column_names=["a", "b"],
        instance=[1, 2],
        y_minmax=(10.0, 25.0),
        interval=True,
    )
    assert spec.header is not None
    assert spec.header.xlim == (0.0, 1.0)
    assert spec.header.low == pytest.approx(0.45)
    assert spec.header.high == pytest.approx(0.7)
    assert spec.header.pred == pytest.approx(0.6)
