"""Targeted unit tests to close small coverage gaps in utility and plugin modules."""

from __future__ import annotations

import pickle


# ---------------------------------------------------------------------------
# explanations/explanations.py — _jsonify callable case (line 116)
# Direct function call _jsonify(...) is NOT a private-member attribute access.
# ---------------------------------------------------------------------------


def test_jsonify_callable_returns_string():
    from calibrated_explanations.explanations.explanations import jsonify_value

    result = jsonify_value(lambda x: x)
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# explanations/explanations.py — ExportedMultiClassExplanationCollection.__getstate__
# (line 102): covered when pickling the frozen dataclass.
# ---------------------------------------------------------------------------


def test_exported_multiclass_collection_pickle_roundtrip():
    from calibrated_explanations.explanations.explanations import (
        ExportedMultiClassExplanationCollection,
    )

    obj = ExportedMultiClassExplanationCollection(
        metadata={"size": 1},
        explanations_by_instance=[{0: None}],
    )
    data = pickle.dumps(obj)
    restored = pickle.loads(data)
    assert restored.metadata == {"size": 1}


# ---------------------------------------------------------------------------
# utils/exceptions.py — explain_exception without details (arc 104->106)
# ---------------------------------------------------------------------------


def test_explain_exception_without_details():
    from calibrated_explanations.utils.exceptions import ValidationError, explain_exception

    err = ValidationError("no details here")  # details=None by default
    text = explain_exception(err)
    assert "ValidationError" in text
    assert "no details here" in text
    assert "Details" not in text  # details block skipped


# ---------------------------------------------------------------------------
# utils/int_utils.py — collect_ints with non-integer string (arc 47->49)
# ---------------------------------------------------------------------------


def test_collect_ints_non_integer_string():
    from calibrated_explanations.utils.int_utils import collect_ints

    result = collect_ints("not_an_integer")
    assert result == []


# ---------------------------------------------------------------------------
# plugins/plots.py — PlotRenderContext __setstate__ (lines 61-64)
# ---------------------------------------------------------------------------


def test_plot_render_context_pickle_roundtrip():
    from calibrated_explanations.plugins.plots import PlotRenderContext

    ctx = PlotRenderContext(
        explanation=None,
        instance_metadata={},
        style="regular",
        intent={},
        show=False,
        path=None,
        save_ext=None,
        options={},
        plugin_config={"alpha": 1.0},
    )
    data = pickle.dumps(ctx)
    restored = pickle.loads(data)
    assert restored.style == ctx.style
    assert restored.show == ctx.show


# ---------------------------------------------------------------------------
# explanations/_conjunctions.py — ConjunctionState.__init__ without is_conjunctive
# (line 40): triggers when initial_rules dict lacks "is_conjunctive".
# ---------------------------------------------------------------------------


def test_conjunction_state_adds_is_conjunctive_when_missing():
    from calibrated_explanations.explanations._conjunctions import ConjunctionState

    rules = {
        "base_predict": [0.5],
        "base_predict_low": [0.4],
        "base_predict_high": [0.6],
        "predict": [0.5],
        "predict_low": [0.4],
        "predict_high": [0.6],
        "weight": [0.1],
        "weight_low": [0.0],
        "weight_high": [0.2],
        "value": ["v"],
        "rule": ["r1"],
        "feature": [0],
        "sampled_values": [None],
        "feature_value": [1.0],
        "classes": [],
        # NOTE: no "is_conjunctive" key → triggers line 40
    }
    cs = ConjunctionState(initial_rules=rules)
    assert cs.state["is_conjunctive"] == [False]
