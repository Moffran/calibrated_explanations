"""Parity checks for invariant failures across validator and bridge entry points."""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest

from calibrated_explanations.plugins.builtins import LegacyPredictBridge
from calibrated_explanations.plugins.explanations import (
    ExplanationBatch,
    validate_explanation_batch,
)
from calibrated_explanations.explanations.explanations import CalibratedExplanations
from calibrated_explanations.explanations.explanation import CalibratedExplanation
from calibrated_explanations.utils.exceptions import ValidationError


class DummyContainer(CalibratedExplanations):
    """Concrete container type for protocol validation tests."""


class DummyExplanation(CalibratedExplanation):
    """Concrete explanation type for protocol validation tests."""


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (
            {"predict": np.array([0.5]), "low": np.array([0.6]), "high": np.array([0.4])},
            "Interval invariant violated",
        ),
        (
            {"predict": np.array([0.9]), "low": np.array([0.4]), "high": np.array([0.6])},
            "Prediction invariant violated",
        ),
    ],
)
def test_should_raise_validationerror_with_consistent_shape_when_invariants_fail(
    payload: dict[str, np.ndarray],
    expected: str,
) -> None:
    """Equivalent invalid payloads should fail consistently in both entry points."""
    batch = ExplanationBatch(
        container_cls=DummyContainer,
        explanation_cls=DummyExplanation,
        instances=[{"prediction": payload}],
        collection_metadata={"task": "regression", "mode": "test"},
    )

    with pytest.raises(ValidationError, match=expected):
        validate_explanation_batch(batch, expected_task="regression", expected_mode="test")

    mock_explainer = Mock()
    mock_explainer.predict.return_value = (
        payload["predict"],
        (payload["low"], payload["high"]),
    )
    bridge = LegacyPredictBridge(mock_explainer)

    with pytest.raises(ValidationError, match=expected):
        bridge.predict("X", mode="regression", task="regression")
