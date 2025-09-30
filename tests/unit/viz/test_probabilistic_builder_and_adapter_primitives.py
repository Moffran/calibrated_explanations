import json
import os
import pytest

from calibrated_explanations.viz.builders import build_factual_probabilistic_plotspec_dict
from calibrated_explanations.viz.matplotlib_adapter import render as mpl_render
from calibrated_explanations.viz.builders import build_probabilistic_bars_spec

HERE = os.path.dirname(__file__)
SCHEMA_DIR = os.path.abspath(os.path.join(HERE, "../../..", "improvement_docs", "plot_spec"))


def test_builder_outputs_valid_shape():
    # Minimal synthetic inputs matching legacy function shape
    predict = {"predict": 0.7, "low": 0.6, "high": 0.8}
    feature_weights = {"predict": [0.1, 0.3], "low": [0.05, 0.2], "high": [0.15, 0.4]}
    features_to_plot = [0, 1]
    title = "test"
    spec_dict = build_factual_probabilistic_plotspec_dict(
        title=title,
        predict=predict,
        feature_weights=feature_weights,
        features_to_plot=features_to_plot,
        column_names=["f0", "f1"],
        rule_labels=None,
        instance=["a", "b"],
        y_minmax=(0.0, 1.0),
        interval=True,
        sort_by=None,
        ascending=False,
    )
    assert isinstance(spec_dict, dict)
    ps = spec_dict.get("plot_spec")
    assert ps is not None
    assert ps.get("kind") in ("factual_probabilistic", "factual_regression")
    assert "feature_entries" in ps
    assert len(ps["feature_entries"]) == 2


def test_builder_validates_against_schema_if_available():
    try:
        import jsonschema  # type: ignore
    except Exception:
        pytest.skip("jsonschema not available; skip strict validation")
    schema_path = os.path.join(SCHEMA_DIR, "plotspec_schema.json")
    with open(schema_path, "r", encoding="utf-8") as _f:
        plotspec_schema = json.load(_f)
    spec_dict = build_factual_probabilistic_plotspec_dict(
        title=None,
        predict={"predict": 0.6, "low": 0.5, "high": 0.7},
        feature_weights={"predict": [0.2], "low": [0.1], "high": [0.3]},
        features_to_plot=[0],
        column_names=["f0"],
        rule_labels=None,
        instance=[0],
        y_minmax=(0.0, 1.0),
        interval=True,
    )
    jsonschema.validate(instance=spec_dict["plot_spec"], schema=plotspec_schema)


def test_adapter_returns_primitives_for_simple_case():
    # Build a PlotSpec dataclass directly and ask the matplotlib adapter to
    # export drawn primitives for it. This is a small golden test ensuring the
    # adapter hooks into the PlotSpec builder path and returns a primitives dict.
    spec = build_probabilistic_bars_spec(
        title="golden",
        predict={"predict": 0.6, "low": 0.5, "high": 0.7},
        feature_weights={"predict": [0.2], "low": [0.1], "high": [0.3]},
        features_to_plot=[0],
        column_names=["f0"],
        rule_labels=None,
        instance=[0],
        y_minmax=(0.0, 1.0),
        interval=True,
    )
    result = mpl_render(spec, show=False, save_path=None, export_drawn_primitives=True)
    assert isinstance(result, dict)
    # Adapter may return a normalized wrapper (with 'primitives') or a
    # legacy-style top-level dict (solids/overlays/header). Accept both for
    # backward compatibility while ensuring at least one semantic primitive
    # is discoverable.
    if "primitives" in result:
        primitives = result["primitives"]
        assert isinstance(primitives, list)
        assert any(isinstance(p, dict) and "semantic" in p for p in primitives)
    else:
        # Fallback: construct a minimal semantic view from legacy keys
        has_any = bool(
            result.get("solids")
            or result.get("overlays")
            or result.get("base_interval")
            or result.get("header")
        )
        assert has_any, "Adapter returned no primitives in either normalized or legacy form"
