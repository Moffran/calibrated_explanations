from __future__ import annotations


import pytest

from tests.helpers.doc_utils import run_quickstart_classification


def test_quickstart_classification_snippet_output(enable_fallbacks, capsys):
    context = run_quickstart_classification()
    # assert "Prediction" in captured # Removed print in helper
    assert len(context.factual) == 5
    assert len(context.alternatives) == 2


def test_quickstart_classification_metadata(enable_fallbacks):
    context = run_quickstart_classification()
    batch = context.factual
    assert len(batch) == 5
    telemetry = getattr(batch, "telemetry", {})
    assert "interval_source" in telemetry
    assert telemetry.get("mode") == "factual"
    first_instance = batch[0]
    prediction = first_instance.prediction
    uncertainty = telemetry.get("uncertainty", {})
    assert uncertainty.get("representation")
    assert uncertainty.get("calibrated_value") == pytest.approx(
        prediction.get("predict", prediction.get("calibrated_value"))
    )
    alternatives = context.alternatives
    assert len(alternatives) == 2
