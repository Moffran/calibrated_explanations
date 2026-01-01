from __future__ import annotations

import json
from pathlib import Path

import pytest

from calibrated_explanations.schema import validation as schema_validation


def test_golden_fixture_validates_with_jsonschema():
    """Golden payload validates with real jsonschema when available."""
    jsonschema = pytest.importorskip("jsonschema")

    # The canonical golden fixture lives in the integration tests folder
    p = Path(__file__).resolve().parents[2] / "integration" / "core" / "golden_explanation_v1.json"
    obj = json.loads(p.read_text(encoding="utf-8"))

    # This will raise jsonschema.ValidationError on failure
    jsonschema.validate(instance=obj, schema=schema_validation._schema_json())


def test_missing_required_fields_rejected_by_jsonschema():
    """A payload missing required top-level fields should be rejected by jsonschema."""
    jsonschema = pytest.importorskip("jsonschema")

    bad_payload = {"task": "classification"}  # missing required keys like 'rules'

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=bad_payload, schema=schema_validation._schema_json())
