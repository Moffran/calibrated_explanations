import warnings

import pytest

from calibrated_explanations.explanations.models import Explanation, FeatureRule
from calibrated_explanations.plugins.plots import PlotRenderContext
from calibrated_explanations.serialization import to_json
from calibrated_explanations.utils.exceptions import ValidationError


def test_explanations_facade_invalid_attr_raises():
    import calibrated_explanations.explanations as explanations

    with pytest.raises(AttributeError):
        getattr(explanations, "NotARealExplanation")


def test_viz_facade_invalid_attr_raises():
    import calibrated_explanations.viz as viz

    with pytest.raises(AttributeError):
        getattr(viz, "NotAPlot")


def test_plot_render_context_getstate_returns_dict():
    context = PlotRenderContext(
        explanation=None,
        instance_metadata={},
        style="default",
        intent={},
        show=False,
        path=None,
        save_ext=None,
        options={},
    )

    state = context.__getstate__()

    assert isinstance(state, dict)
    assert state["style"] == "default"


def test_serialization_invariant_low_greater_than_high():
    explanation = Explanation(
        task="classification",
        index=0,
        explanation_type="factual",
        prediction={"predict": 0.5, "low": 0.9, "high": 0.1},
        rules=[
            FeatureRule(
                feature=0,
                rule="x > 0",
                rule_weight={"predict": 0.1},
                rule_prediction={"predict": 0.1, "low": 0.1, "high": 0.1},
            )
        ],
    )

    with pytest.raises(ValidationError, match="low > high"):
        to_json(explanation)


def test_reject_policy_deprecated_attr_access_warns():
    from calibrated_explanations.core.reject import policy as reject_policy
    from calibrated_explanations.explanations.reject import RejectPolicy

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert reject_policy.PREDICT_AND_FLAG is RejectPolicy.FLAG

    assert any(
        issubclass(item.category, DeprecationWarning) for item in caught
    ), "Expected DeprecationWarning for deprecated reject policy name."

    with pytest.raises(AttributeError):
        getattr(reject_policy, "NOT_A_POLICY")
