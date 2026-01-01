from __future__ import annotations


from tests.helpers.doc_utils import run_quickstart_regression


def test_quickstart_regression_snippet_output(enable_fallbacks, capsys):
    context = run_quickstart_regression()
    # assert "Prediction interval" in captured # Removed print
    # assert "Calibrated probability" in captured # Removed print
    assert len(context.factual) == 3
    assert len(context.alternatives) == 2


def test_quickstart_regression_metadata(enable_fallbacks):
    context = run_quickstart_regression()
    batch = context.explainer.explore_alternatives(context.X_test[:3], threshold=150)
    telemetry = getattr(batch, "telemetry", {})
    assert telemetry.get("task") == "regression"
    uncertainty = telemetry.get("uncertainty", {})
    assert uncertainty.get("representation")
    assert 0.0 <= context.probabilistic[0, 1] <= 1.0
    assert len(context.probabilistic_interval) == 2
