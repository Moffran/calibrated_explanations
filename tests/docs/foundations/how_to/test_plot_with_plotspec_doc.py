from __future__ import annotations

import pytest


def test_plot_with_plotspec_snippet(tmp_path):
    pytest.importorskip("calibrated_explanations.viz")
    from calibrated_explanations.viz import (  # type: ignore[import-untyped]
        build_regression_bars_spec,
        matplotlib_adapter,
    )

    predict = {"predict": 3.2, "low": 2.7, "high": 3.8}
    feature_weights = {
        "predict": [0.8, -0.4, 0.2],
        "low": [0.5, -0.7, 0.0],
        "high": [1.1, -0.1, 0.4],
    }
    features_to_plot = [0, 1, 2]
    column_names = ["feat_a", "feat_b", "feat_c"]
    instance = [1.2, 0.3, -0.1]

    spec = build_regression_bars_spec(
        title="Regression feature contributions",
        predict=predict,
        feature_weights=feature_weights,
        features_to_plot=features_to_plot,
        column_names=column_names,
        instance=instance,
        y_minmax=(2.0, 4.0),
        interval=True,
        sort_by="abs",  # one of: none|value|abs|width|label
        ascending=False,
    )

    output = tmp_path / "plotspec_example.png"
    matplotlib_adapter.render(spec, show=False, save_path=str(output))
    assert output.exists()
