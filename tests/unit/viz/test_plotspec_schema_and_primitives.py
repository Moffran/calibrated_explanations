import importlib.util
import json
import os
import pytest

HERE = os.path.dirname(__file__)
SCHEMA_DIR = os.path.abspath(os.path.join(HERE, "../../..", "improvement_docs", "plot_spec"))


def load_schema(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def test_plotspec_and_primitives_schemas_exist():
    plotspec = os.path.join(SCHEMA_DIR, "plotspec_schema.json")
    primitives = os.path.join(SCHEMA_DIR, "primitives_schema.json")
    assert os.path.exists(plotspec), "plotspec_schema.json must exist"
    assert os.path.exists(primitives), "primitives_schema.json must exist"


def test_schema_declares_interval_and_save_requirements():
    plotspec = load_schema(os.path.join(SCHEMA_DIR, "plotspec_schema.json"))
    required = set(plotspec.get("required", ()))
    assert {"header", "body", "feature_order"}.issubset(required)
    feature_entries = (
        plotspec.get("properties", {})
        .get("feature_entries", {})
        .get("items", {})
        .get("properties", {})
    )
    assert "low" in feature_entries and "high" in feature_entries
    save_behavior = plotspec.get("properties", {}).get("save_behavior", {}).get("properties", {})
    if save_behavior:  # schema may allow omitting save_behavior block entirely
        assert "default_exts" in save_behavior


def test_plotspec_schema_declares_feature_entry_requirements():
    schema = load_schema(os.path.join(SCHEMA_DIR, "plotspec_schema.json"))
    props = schema.get("properties", {})
    assert "feature_entries" in props
    feature_entries = props["feature_entries"]
    items = feature_entries.get("items", {})
    required = set(items.get("required", []))
    assert {"index", "name", "weight"}.issubset(required)


def test_plotspec_schema_defines_feature_contract():
    schema = load_schema(os.path.join(SCHEMA_DIR, "plotspec_schema.json"))

    assert "feature_order" in schema["required"]
    feature_order = schema["properties"]["feature_order"]
    assert feature_order["type"] == "array"
    assert feature_order["items"]["type"] == "integer"

    feature_entries = schema["properties"]["feature_entries"]
    entry_schema = feature_entries["items"]
    assert set(entry_schema["required"]) == {"index", "name", "weight"}
    low_type = entry_schema["properties"]["low"]["type"]
    high_type = entry_schema["properties"]["high"]["type"]
    assert "number" in low_type
    assert "number" in high_type


@pytest.mark.skipif(
    importlib.util.find_spec("jsonschema") is None,
    reason="jsonschema not installed in this environment; run locally with jsonschema to validate schemas",
)
def test_example_plotspec_validates():
    # This test is skipped by default; enable locally if jsonschema is available.
    import jsonschema

    plotspec_schema = load_schema(os.path.join(SCHEMA_DIR, "plotspec_schema.json"))
    primitives_schema = load_schema(os.path.join(SCHEMA_DIR, "primitives_schema.json"))

    # Minimal factual_probabilistic example
    example = {
        "plot_spec": {
            "kind": "factual_probabilistic",
            "mode": "classification",
            "header": {"dual": True},
            "body": {"bars_count": 1},
            "style": "regular",
            "uncertainty": True,
            "feature_order": [0],
            "feature_entries": [
                {"index": 0, "name": "f0", "weight": 0.35, "low": 0.2, "high": 0.5}
            ],
        },
        "primitives": [
            {
                "id": "p1",
                "axis_id": "header.pos",
                "type": "fill_between",
                "coords": {"x": [0, 1], "y1": [0.2, 0.2], "y2": [0.5, 0.5]},
                "style": {"color": "#ff0000", "alpha": 0.2},
                "semantic": "probability_fill",
            }
        ],
    }

    # Validate plot_spec and full export
    jsonschema.validate(instance=example["plot_spec"], schema=plotspec_schema)
    jsonschema.validate(instance=example, schema=primitives_schema)
