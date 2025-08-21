"""Phase 1A prediction helper delegators.

This module introduces *thin* wrapper functions around existing private
methods of ``CalibratedExplainer``. It is an intermediate, mechanical step
that allows future extractions without touching behavior now. Tests will
exercise these wrappers to lock in semantics before moving logic bodies.
"""

from __future__ import annotations

from typing import Any

# NOTE: We intentionally avoid importing CalibratedExplainer for type-only usage to
# prevent cyclical import complexity during the gradual split.


def validate_and_prepare_input(explainer: Any, X_test):  # pragma: no cover - thin wrapper
    return explainer._validate_and_prepare_input(X_test)  # noqa: SLF001


def initialize_explanation(
    explainer: Any,
    X_test,
    low_high_percentiles,
    threshold,
    bins,
    features_to_ignore,
):  # pragma: no cover - thin wrapper
    return explainer._initialize_explanation(
        X_test, low_high_percentiles, threshold, bins, features_to_ignore
    )


def explain_predict_step(
    explainer: Any,
    X_test,
    threshold,
    low_high_percentiles,
    bins,
    features_to_ignore,
):  # pragma: no cover - thin wrapper
    return explainer._explain_predict_step(
        X_test, threshold, low_high_percentiles, bins, features_to_ignore
    )


__all__ = [
    "validate_and_prepare_input",
    "initialize_explanation",
    "explain_predict_step",
]
