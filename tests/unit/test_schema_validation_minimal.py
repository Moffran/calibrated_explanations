import pytest
import importlib

from calibrated_explanations.schema import validation as schema_validation
from calibrated_explanations.utils.exceptions import ValidationError


def test_validate_payload_minimal_checks(monkeypatch):
    monkeypatch.setattr(schema_validation, "jsonschema", None)

    with pytest.raises(ValidationError, match="Missing required payload key"):
        schema_validation.validate_payload({})

    with pytest.raises(ValidationError, match="Field 'task' must be a string"):
        schema_validation.validate_payload(
            {
                "task": 123,
                "index": 0,
                "explanation_type": "factual",
                "prediction": {},
                "rules": [],
            }
        )

    with pytest.raises(ValidationError, match="Field 'prediction' must be an object"):
        schema_validation.validate_payload(
            {
                "task": "ok",
                "index": 0,
                "explanation_type": "factual",
                "prediction": "bad",
                "rules": [],
            }
        )

    with pytest.raises(ValidationError, match="prediction missing required key"):
        schema_validation.validate_payload(
            {
                "task": "ok",
                "index": 0,
                "explanation_type": "factual",
                "prediction": {"predict": 1},
                "rules": [],
            }
        )

    with pytest.raises(ValidationError, match="Field 'rules' must be an array"):
        schema_validation.validate_payload(
            {
                "task": "ok",
                "index": 0,
                "explanation_type": "factual",
                "prediction": {},
                "rules": "bad",
            }
        )

    with pytest.raises(ValidationError, match="missing required key"):
        schema_validation.validate_payload(
            {
                "task": "ok",
                "index": 0,
                "explanation_type": "factual",
                "prediction": {},
                "rules": [{"feature": 0}],
            }
        )

    with pytest.raises(ValidationError, match="feature must be integer or list"):
        schema_validation.validate_payload(
            {
                "task": "ok",
                "index": 0,
                "explanation_type": "factual",
                "prediction": {},
                "rules": [
                    {
                        "feature": "bad",
                        "rule": "r",
                        "rule_weight": {},
                        "rule_prediction": {},
                    }
                ],
            }
        )


def test_schema_validation_handles_missing_jsonschema_import(monkeypatch):
    real_import = __import__

    def guarded_import(name, *args, **kwargs):
        if name == "jsonschema":
            raise ImportError("simulated missing jsonschema")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded_import)
    reloaded = importlib.reload(schema_validation)
    assert reloaded.jsonschema is None

    with pytest.raises(ValidationError, match="feature list must contain only integers"):
        schema_validation.validate_payload(
            {
                "task": "ok",
                "index": 0,
                "explanation_type": "factual",
                "prediction": {},
                "rules": [
                    {
                        "feature": ["bad"],
                        "rule": "r",
                        "rule_weight": {},
                        "rule_prediction": {},
                    }
                ],
            }
        )
