import json
from pathlib import Path

from calibrated_explanations.serialization import from_json, validate_payload


def test_golden_fixture_roundtrip_and_validate():
    p = Path(__file__).with_name("golden_explanation_v1.json")
    obj = json.loads(p.read_text(encoding="utf-8"))
    # validate_payload is a no-op when jsonschema is not installed
    validate_payload(obj)
    exp = from_json(obj)
    assert exp.task == "classification"
    assert exp.index == 0
    assert len(exp.rules) == 2
