from __future__ import annotations

import numpy as np
import pytest

from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer
from calibrated_explanations.core.wrap_explainer import WrapCalibratedExplainer
from calibrated_explanations.explanations.explanations import CalibratedExplanations


def test_should_raise_attribute_error_for_removed_task21_core_wrapper_entrypoints() -> None:
    """Task 5A v0.11.2 removals must fail closed for core and wrapper LIME/SHAP hooks."""
    explainer = object.__new__(CalibratedExplainer)
    wrapper = object.__new__(WrapCalibratedExplainer)

    removed_calls = [
        ("preload_lime", lambda: explainer.preload_lime()),
        ("preload_shap", lambda: explainer.preload_shap()),
        ("explain_lime", lambda: explainer.explain_lime(np.array([[0.0, 1.0]]))),
        ("explain_shap", lambda: explainer.explain_shap(np.array([[0.0, 1.0]]))),
        ("is_lime_enabled", lambda: explainer.is_lime_enabled()),
        ("is_shap_enabled", lambda: explainer.is_shap_enabled()),
        ("explain_lime", lambda: wrapper.explain_lime(np.array([[0.0, 1.0]]))),
        ("explain_shap", lambda: wrapper.explain_shap(np.array([[0.0, 1.0]]))),
    ]

    for symbol, callback in removed_calls:
        with pytest.raises(AttributeError, match=symbol):
            callback()


def test_should_raise_attribute_error_for_removed_collection_as_lime_as_shap() -> None:
    """Collection LIME/SHAP adapters were removed in v0.11.3."""
    collection = object.__new__(CalibratedExplanations)

    for symbol in ("as_lime", "as_shap"):
        with pytest.raises(AttributeError, match=symbol):
            getattr(collection, symbol)()
