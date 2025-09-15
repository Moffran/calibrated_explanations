import os
import tempfile

import numpy as np
import pytest

from calibrated_explanations.viz import build_regression_bars_spec, matplotlib_adapter


pytest.importorskip("matplotlib")
pytestmark = pytest.mark.viz


def test_plotspec_regression_render_smoke():
    rng = np.random.default_rng(0)
    nfeat = 5
    # Fake inputs similar to _plot_regression
    predict = {"predict": 0.5, "low": 0.2, "high": 0.8}
    fw = {
        "predict": rng.normal(0, 0.2, size=nfeat),
        "low": rng.normal(-0.1, 0.1, size=nfeat),
        "high": rng.normal(0.1, 0.1, size=nfeat),
    }
    feats = list(range(nfeat))
    cols = [f"f{i}" for i in range(nfeat)]
    instance = rng.normal(size=nfeat)
    spec = build_regression_bars_spec(
        title="MVP",
        predict=predict,
        feature_weights=fw,
        features_to_plot=feats,
        column_names=cols,
        instance=instance,
        y_minmax=(0.0, 1.0),
        interval=True,
    )
    # Render to a temp file; should not raise
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "mvp.png")
        matplotlib_adapter.render(spec, show=False, save_path=out)
        assert os.path.exists(out)


def test_plotspec_sorting_abs_desc():
    nfeat = 6
    predict = {"predict": 0.4, "low": 0.1, "high": 0.7}
    vals = np.array([0.2, -0.9, 0.5, -0.1, 0.7, -0.6])
    fw = {"predict": vals, "low": vals - 0.1, "high": vals + 0.1}
    feats = list(range(nfeat))
    cols = [f"f{i}" for i in range(nfeat)]
    spec = build_regression_bars_spec(
        title=None,
        predict=predict,
        feature_weights=fw,
        features_to_plot=feats,
        column_names=cols,
        instance=None,
        y_minmax=(0.0, 1.0),
        interval=True,
        sort_by="abs",
        ascending=False,
    )
    bars = spec.body.bars  # type: ignore[union-attr]
    magnitudes = [abs(b.value) for b in bars]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_plotspec_sorting_width_and_interval_equivalence():
    rng = np.random.default_rng(2)
    nfeat = 4
    predict = {"predict": 0.5, "low": 0.3, "high": 0.7}
    vals = rng.normal(0, 0.2, size=nfeat)
    low = vals - rng.uniform(0.05, 0.15, size=nfeat)
    high = vals + rng.uniform(0.05, 0.25, size=nfeat)
    fw = {"predict": vals, "low": low, "high": high}
    feats = list(range(nfeat))
    cols = [f"f{i}" for i in range(nfeat)]

    spec_interval = build_regression_bars_spec(
        title=None,
        predict=predict,
        feature_weights=fw,
        features_to_plot=feats,
        column_names=cols,
        instance=None,
        y_minmax=(0.0, 1.0),
        interval=True,
        sort_by="interval",
        ascending=False,
    )
    spec_width = build_regression_bars_spec(
        title=None,
        predict=predict,
        feature_weights=fw,
        features_to_plot=feats,
        column_names=cols,
        instance=None,
        y_minmax=(0.0, 1.0),
        interval=True,
        sort_by="width",
        ascending=False,
    )
    labels_interval = [b.label for b in spec_interval.body.bars]  # type: ignore[union-attr]
    labels_width = [b.label for b in spec_width.body.bars]  # type: ignore[union-attr]
    assert labels_interval == labels_width


def test_plotspec_sorting_abs_means_distance_from_zero():
    # Ensure 'abs' sorts by |value - 0| (distance from zero)
    predict = {"predict": 0.0}
    vals = np.array([-0.05, 0.2, -0.3, 0.1])
    fw = {"predict": vals, "low": vals - 0.01, "high": vals + 0.01}
    feats = list(range(len(vals)))
    cols = [f"f{i}" for i in feats]
    spec = build_regression_bars_spec(
        title=None,
        predict=predict,
        feature_weights=fw,
        features_to_plot=feats,
        column_names=cols,
        instance=None,
        y_minmax=None,
        interval=True,
        sort_by="abs",
        ascending=False,
    )
    bars = spec.body.bars  # type: ignore[union-attr]
    distances = [abs(b.value - 0.0) for b in bars]
    assert distances == sorted(distances, reverse=True)


def test_plotspec_sorting_width_descending_changes_order():
    # Construct intervals with distinct widths to verify sorting effect
    predict = {"predict": 0.0}
    vals = np.array([0.1, -0.2, 0.05, 0.3])
    # widths: 0.30, 0.10, 0.20, 0.40
    low = np.array([-0.1, -0.15, 0.0, 0.1])
    high = np.array([0.2, -0.05, 0.2, 0.5])
    fw = {"predict": vals, "low": low, "high": high}
    feats = list(range(len(vals)))
    cols = [f"f{i}" for i in feats]
    spec = build_regression_bars_spec(
        title=None,
        predict=predict,
        feature_weights=fw,
        features_to_plot=feats,
        column_names=cols,
        instance=None,
        y_minmax=None,
        interval=True,
        sort_by="width",
        ascending=False,
    )
    bars = spec.body.bars  # type: ignore[union-attr]
    widths_sorted = [abs(b.interval_high - b.interval_low) for b in bars]  # type: ignore[operator]
    assert widths_sorted == sorted(widths_sorted, reverse=True)


def test_plotspec_sorting_value_ascending():
    predict = {"predict": 0.0}
    vals = np.array([0.3, -0.1, 0.2, -0.4])
    fw = {"predict": vals, "low": vals - 0.05, "high": vals + 0.05}
    feats = list(range(len(vals)))
    cols = [f"f{i}" for i in feats]
    spec = build_regression_bars_spec(
        title=None,
        predict=predict,
        feature_weights=fw,
        features_to_plot=feats,
        column_names=cols,
        instance=None,
        y_minmax=None,
        interval=True,
        sort_by="value",
        ascending=True,
    )
    bars = spec.body.bars  # type: ignore[union-attr]
    values_sorted = [b.value for b in bars]
    assert values_sorted == sorted(values_sorted)


def test_plotspec_sorting_label_ascending():
    predict = {"predict": 0.0}
    vals = np.array([0.1, 0.1, 0.1])
    fw = {"predict": vals, "low": vals - 0.01, "high": vals + 0.01}
    feats = [2, 0, 1]
    cols = ["b_label", "a_label", "c_label"]
    spec = build_regression_bars_spec(
        title=None,
        predict=predict,
        feature_weights=fw,
        features_to_plot=feats,
        column_names=cols,
        instance=None,
        y_minmax=None,
        interval=True,
        sort_by="label",
        ascending=True,
    )
    bars = spec.body.bars  # type: ignore[union-attr]
    labels_sorted = [b.label for b in bars]
    assert labels_sorted == sorted(labels_sorted)
