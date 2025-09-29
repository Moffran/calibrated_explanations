from calibrated_explanations.viz.builders import build_regression_bars_spec
from calibrated_explanations.viz.matplotlib_adapter import render


def _make_regression_interval_weights():
    # Predict with an interval around p=100. We'll use feature weight intervals
    # that cross zero after mapping into contribution coords when header.pred is subtracted.
    predict = {"predict": 100.0, "low": 90.0, "high": 110.0}
    # One feature that has a small positive effect but interval crossing the header
    # in raw values: predict weight is +5, interval low=95, high=105 -> body coords: -5..+5 crosses zero
    feature_weights = {
        "predict": [105.0],
        "low": [95.0],
        "high": [115.0],
    }
    return predict, feature_weights


def test_regression_exports_base_interval_and_suppresses_solid_by_default():
    predict, feature_weights = _make_regression_interval_weights()
    spec = build_regression_bars_spec(
        title="test",
        predict=predict,
        feature_weights=feature_weights,
        features_to_plot=[0],
        column_names=["f0"],
        instance=None,
        y_minmax=(0.0, 200.0),
        interval=True,
    )
    primitives = render(spec, export_drawn_primitives=True)
    # Base interval should be present in primitives under 'base_interval' -> 'body'
    base = primitives.get("base_interval", {}).get("body")
    assert base is not None, f"Expected base interval primitive, got: {primitives}"
    # By default (legacy), the solid for the bar should be suppressed when interval crosses zero
    solids = primitives.get("solids", [])
    overlays = primitives.get("overlays", [])
    assert all(s.get("index", 0) != 0 for s in solids), f"Unexpected solid for index 0: {solids}"
    assert any(o.get("index", -1) == 0 for o in overlays), "Expected overlay for index 0"


def test_regression_parity_draws_solid_when_flag_false():
    predict, feature_weights = _make_regression_interval_weights()
    spec = build_regression_bars_spec(
        title="test",
        predict=predict,
        feature_weights=feature_weights,
        features_to_plot=[0],
        column_names=["f0"],
        instance=None,
        y_minmax=(0.0, 200.0),
        interval=True,
        legacy_solid_behavior=False,
    )
    primitives = render(spec, export_drawn_primitives=True)
    solids = primitives.get("solids", [])
    overlays = primitives.get("overlays", [])
    assert len(solids) >= 1, f"Expected at least one solid in parity mode, got {solids}"
    assert len(overlays) >= 1, "Expected overlay present"
