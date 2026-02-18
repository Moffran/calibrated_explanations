import pytest

from calibrated_explanations.explanations.models import Explanation, FeatureRule
from calibrated_explanations.plugins.plots import PlotRenderContext
from calibrated_explanations.serialization import to_json
from calibrated_explanations.utils.exceptions import ValidationError


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
        prediction={"predict": [0.2, 0.8], "low": [0.1, 0.9], "high": [0.3, 0.7]},
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
