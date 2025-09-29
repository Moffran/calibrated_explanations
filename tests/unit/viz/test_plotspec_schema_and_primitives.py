import json
import os
import pytest

HERE = os.path.dirname(__file__)
SCHEMA_DIR = os.path.abspath(os.path.join(HERE, "../../..", "improvement_docs", "plot_spec"))


def _load(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def test_plotspec_and_primitives_schemas_exist():
    plotspec = os.path.join(SCHEMA_DIR, "plotspec_schema.json")
    primitives = os.path.join(SCHEMA_DIR, "primitives_schema.json")
    assert os.path.exists(plotspec), "plotspec_schema.json must exist"
    assert os.path.exists(primitives), "primitives_schema.json must exist"


@pytest.mark.skipif(
    True,
    reason="jsonschema not installed in this environment; run locally with jsonschema to validate schemas",
)
def test_example_plotspec_validates():
    # This test is skipped by default; enable locally if jsonschema is available.
    import jsonschema

    plotspec_schema = _load(os.path.join(SCHEMA_DIR, "plotspec_schema.json"))
    primitives_schema = _load(os.path.join(SCHEMA_DIR, "primitives_schema.json"))

    # Minimal factual_probabilistic example
    example = {
        "plot_spec": {
            "kind": "factual_probabilistic",
            "mode": "classification",
            "header": "dual",
            "body": "weight",
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
