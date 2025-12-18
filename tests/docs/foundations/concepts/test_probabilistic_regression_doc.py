from __future__ import annotations

from tests.helpers.doc_utils import (
    run_quickstart_regression,
)


def test_probabilistic_regression_snippets(enable_fallbacks):
    context = run_quickstart_regression()
    probabilities, probability_interval = context.explainer.predict_proba(
        context.X_test[:1],
        threshold=150,
        uq_interval=True,
    )
    low, high = probability_interval
    print(f"Calibrated probability: {probabilities[0, 1]:.3f}")
    print(f"Interval bounds: {low[0]:.3f} – {high[0]:.3f}")

    alternatives = context.explainer.explore_alternatives(
        context.X_test[:2],
        threshold=150,
    )
    assert alternatives
