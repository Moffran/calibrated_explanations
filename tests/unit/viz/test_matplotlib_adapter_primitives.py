from calibrated_explanations.viz.builders import (
    build_probabilistic_bars_spec,
)
from calibrated_explanations.viz.matplotlib_adapter import render


def _make_interval_weights():
    # Create a simple predict dict and feature_weights with one rule crossing zero
    predict = {"predict": 0.5, "low": 0.4, "high": 0.6}
    # One rule that maps to an interval crossing the header prediction
    feature_weights = {
        "predict": [0.6],
        "low": [0.45],
        "high": [0.55],
    }
    return predict, feature_weights


def test_default_suppresses_solid_for_crossing_interval():
    predict, feature_weights = _make_interval_weights()
    spec = build_probabilistic_bars_spec(
        title="test",
        predict=predict,
        feature_weights=feature_weights,
        features_to_plot=[0],
        column_names=["f0"],
        instance=None,
        y_minmax=None,
        interval=True,
    )
    primitives = render(spec, export_drawn_primitives=True)
    # Default legacy behaviour: when an interval crosses zero the solid is
    # suppressed and split overlays are drawn instead. Confirm overlays used
    # for index 0 and no solid remains for that index.
    solids = primitives.get("solids", [])
    overlays = primitives.get("overlays", [])
    assert any(
        o.get("index", -1) == 0 for o in overlays
    ), f"Expected overlay for index 0, got overlays={overlays}"
    assert all(s.get("index", -1) != 0 for s in solids), f"Unexpected solid for index 0: {solids}"


def test_parity_shows_solid_when_flag_false():
    predict, feature_weights = _make_interval_weights()
    spec = build_probabilistic_bars_spec(
        title="test",
        predict=predict,
        feature_weights=feature_weights,
        features_to_plot=[0],
        column_names=["f0"],
        instance=None,
        y_minmax=None,
        interval=True,
        legacy_solid_behavior=False,
    )
    primitives = render(spec, export_drawn_primitives=True)
    solids = primitives.get("solids", [])
    overlays = primitives.get("overlays", [])
    # Parity mode should draw a solid and overlay for the crossing interval
    assert len(solids) >= 1, f"Expected at least one solid, got {solids}"
    assert len(overlays) >= 1, "Expected overlay present"
