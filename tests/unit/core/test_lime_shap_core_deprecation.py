from __future__ import annotations

import numpy as np
import pytest

from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer
from calibrated_explanations.core.wrap_explainer import WrapCalibratedExplainer
from calibrated_explanations.explanations.explanations import CalibratedExplanations


def test_task21_lime_shap_entrypoints_raise_with_ce_deprecations_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CE_DEPRECATIONS", "error")

    explainer = object.__new__(CalibratedExplainer)
    wrapper = object.__new__(WrapCalibratedExplainer)
    collection = object.__new__(CalibratedExplanations)

    checks = [
        ("preload_lime is deprecated", lambda: explainer.preload_lime()),
        ("preload_shap is deprecated", lambda: explainer.preload_shap()),
        ("explain_lime is deprecated", lambda: explainer.explain_lime(np.array([[0.0, 1.0]]))),
        ("explain_shap is deprecated", lambda: explainer.explain_shap(np.array([[0.0, 1.0]]))),
        ("is_lime_enabled is deprecated", lambda: explainer.is_lime_enabled()),
        ("is_shap_enabled is deprecated", lambda: explainer.is_shap_enabled()),
        (
            "WrapCalibratedExplainer.explain_lime is deprecated",
            lambda: wrapper.explain_lime(np.array([[0.0, 1.0]])),
        ),
        (
            "WrapCalibratedExplainer.explain_shap is deprecated",
            lambda: wrapper.explain_shap(np.array([[0.0, 1.0]])),
        ),
        ("CalibratedExplanations.as_lime is deprecated", lambda: collection.as_lime()),
        ("CalibratedExplanations.as_shap is deprecated", lambda: collection.as_shap()),
    ]

    for match, callback in checks:
        with pytest.raises(DeprecationWarning, match=match):
            callback()
