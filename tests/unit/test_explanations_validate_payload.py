import pytest
from unittest.mock import MagicMock

import calibrated_explanations.serialization as serialization
from calibrated_explanations.utils.exceptions import ValidationError


class DummyExplainer:
    pass


def test_should_raise_validationerror_when_payload_invalid(monkeypatch):
    # Create a mock Explanation object
    exp = MagicMock()
    exp.task = "classification"
    exp.index = 0
    exp.explanation_type = "factual"
    exp.prediction = {"predict": 0.5, "low": 0.4, "high": 0.6}
    exp.rules = []
    exp.provenance = None
    exp.metadata = None

    def bad_validator(payload):
        raise Exception("schema failure")

    monkeypatch.setattr(serialization, "validate_payload", bad_validator)

    with pytest.raises(ValidationError):
        serialization.to_json(exp)
