from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import numpy as np

from calibrated_explanations.plugins import builtins
from calibrated_explanations.utils.exceptions import NotFittedError
from calibrated_explanations.explanations.explanations import CalibratedExplanations
from calibrated_explanations.explanations.explanation import FactualExplanation


def test_derive_threshold_labels_scalar():
    pos, neg = builtins._derive_threshold_labels(0.5)
    assert pos == "Y < 0.50"
    assert neg == "Y ≥ 0.50"


def test_derive_threshold_labels_interval():
    pos, neg = builtins._derive_threshold_labels((0.2, 0.8))
    assert pos == "0.20 <= Y < 0.80"
    assert neg == "Outside interval"


def test_derive_threshold_labels_invalid():
    pos, neg = builtins._derive_threshold_labels("invalid")
    assert pos == "Target within threshold"
    assert neg == "Outside threshold"


def test_collection_to_batch_empty():
    collection = MagicMock(spec=CalibratedExplanations)
    collection.explanations = []
    collection.mode = "factual"

    batch = builtins._collection_to_batch(collection)
    assert batch.collection_metadata["mode"] == "factual"
    assert batch.explanation_cls == FactualExplanation
    assert len(batch.instances) == 0


def test_collection_to_batch_with_items():
    collection = MagicMock(spec=CalibratedExplanations)
    exp1 = MagicMock(spec=FactualExplanation)
    collection.explanations = [exp1]
    collection.mode = "factual"

    batch = builtins._collection_to_batch(collection)
    assert batch.explanation_cls is type(exp1)
    assert len(batch.instances) == 1
    assert batch.instances[0]["explanation"] == exp1


def test_legacy_interval_calibrator_plugin_create_regression():
    plugin = builtins.LegacyIntervalCalibratorPlugin()
    context = MagicMock()
    context.metadata = {"task": "regression", "explainer": MagicMock()}
    context.calibration_splits = [(np.array([]), np.array([]))]

    # We need to mock IntervalRegressor import or ensure it's available
    with patch(
        "calibrated_explanations.calibration.interval_regressor.IntervalRegressor"
    ) as mock_regressor:
        plugin.create(context)
        mock_regressor.assert_called_once()


def test_legacy_interval_calibrator_plugin_create_missing_explainer():
    plugin = builtins.LegacyIntervalCalibratorPlugin()
    context = MagicMock()
    context.metadata = {"task": "regression"}  # Missing explainer
    context.calibration_splits = [(np.array([]), np.array([]))]

    with pytest.raises(NotFittedError, match="Legacy interval context missing 'explainer' handle"):
        plugin.create(context)
