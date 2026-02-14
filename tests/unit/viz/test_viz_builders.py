import numpy as np

from calibrated_explanations.viz import (
    build_regression_bars_spec,
    is_valid_probability_values,
)


def make_fw(n):
    rng = np.random.default_rng(0)
    vals = rng.normal(0, 0.2, size=n)
    low = vals - rng.uniform(0.05, 0.15, size=n)
    high = vals + rng.uniform(0.05, 0.25, size=n)
    return {"predict": vals, "low": low, "high": high}


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


# Tests for is_valid_probability_values public API


def test_is_valid_probability_values_should_reject_non_numeric_strings():
    """Verify is_valid_probability_values rejects strings that can't convert to float."""
    assert not is_valid_probability_values("abc")
    assert not is_valid_probability_values("1.0.0")
    assert not is_valid_probability_values("not_a_number")


def test_is_valid_probability_values_should_reject_empty_input():
    """Verify is_valid_probability_values requires at least one value."""
    assert not is_valid_probability_values()
