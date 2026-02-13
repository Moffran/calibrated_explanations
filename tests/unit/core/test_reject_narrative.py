"""
Unit tests for rejection handling in NarrativeGenerator.
"""

from unittest.mock import patch
import pytest
from calibrated_explanations.core.narrative_generator import NarrativeGenerator

# Use a minimal template structure for testing
MOCK_TEMPLATES = {
    "reject_indicators": {
        "ambiguity": {
            "advanced": "Ambiguous prediction. Predicted set size: {set_size}. Confidence: {confidence}."
        },
        "outlier": {"beginner": "This looks like an outlier.", "advanced": "Outlier detected."},
    },
    "narrative_templates": {
        "classification": {
            "factual": {"advanced": "Base narrative.", "beginner": "Simple base narrative."}
        }
    },
}


class FakeRejectContext:
    def __init__(self, rejected=True, reject_type="ambiguity", set_size=None, confidence=None):
        self.rejected = rejected
        self.reject_type = reject_type
        self.prediction_set_size = set_size
        self.confidence = confidence
        self.prediction_set = [0, 1] if set_size else []


class FakeExplanation:
    def __init__(self, rules=None, reject_context=None):
        self.rules_data = rules or {}
        if reject_context:
            self.reject_context = reject_context

    def get_rules(self):
        # Return minimal rules to satisfy generate_narrative
        default = {
            "base_predict": 0.5,
            "base_predict_low": 0.4,
            "base_predict_high": 0.6,
            "classes": 1,
            "rules": [],
        }
        return {**default, **self.rules_data}

    def get_class_labels(self):
        return {0: "A", 1: "B"}


@pytest.fixture
def generator():
    with patch(
        "calibrated_explanations.core.narrative_generator.load_template_file",
        return_value=MOCK_TEMPLATES,
    ):
        gen = NarrativeGenerator("dummy_path.yaml")
        return gen


def test_narrative_prepends_rejection_text_ambiguity(generator):
    context = FakeRejectContext(rejected=True, reject_type="ambiguity", set_size=3, confidence=0.95)
    explanation = FakeExplanation(reject_context=context)

    narrative = generator.generate_narrative(
        explanation,
        problem_type="classification",
        explanation_type="factual",
        expertise_level="advanced",
    )

    assert "Ambiguous prediction." in narrative
    assert "Predicted set size: 3" in narrative
    assert "Confidence: 0.95" in narrative
    assert "Base narrative." in narrative


def test_narrative_prepends_rejection_text_outlier(generator):
    context = FakeRejectContext(rejected=True, reject_type="outlier")
    explanation = FakeExplanation(reject_context=context)

    narrative = generator.generate_narrative(
        explanation,
        problem_type="classification",
        explanation_type="factual",
        expertise_level="advanced",
    )

    assert "Outlier detected." in narrative
    assert "Base narrative." in narrative




def test_narrative_handles_missing_reject_context(generator):
    # Explanation without reject_context attribute
    explanation = FakeExplanation()

    narrative = generator.generate_narrative(
        explanation,
        problem_type="classification",
        explanation_type="factual",
        expertise_level="advanced",
    )

    assert "Ambiguous prediction" not in narrative
    assert "Base narrative." in narrative


def test_narrative_handles_formatting_errors_gracefully(generator):
    # Force a formatting error by passing missing keys for template
    # Template expects set_size and confidence, but we provide None
    # (Though fmt_float handles None, let's see)

    # Actually, the template is formatted using .format().
    # If the context object lacks attributes, it might raise AttributeError or KeyError if dict.
    # The code does getattr(rc, "prediction_set", None) etc. so it passes None.
    # formatting "Predicted set size: {set_size}" with set_size=None becomes "Predicted set size: None".
    # This is valid python formatting.

    # Let's try to break it with a template expecting a key that we don't supply in the code?
    # The code extracts specific keys: prediction_set, set_size, confidence.
    # If the template requires something else, it will fail.

    broken_templates = {
        "reject_indicators": {"ambiguity": {"advanced": "Sort of ambiguous: {missing_key}"}},
        "narrative_templates": MOCK_TEMPLATES["narrative_templates"],
    }

    with patch(
        "calibrated_explanations.core.narrative_generator.load_template_file",
        return_value=broken_templates,
    ):
        gen = NarrativeGenerator("dummy_path.yaml")
        context = FakeRejectContext(rejected=True, reject_type="ambiguity")
        explanation = FakeExplanation(reject_context=context)

        # Should catch exception and proceed with unformatted text (best-effort)
        narrative = gen.generate_narrative(
            explanation,
            problem_type="classification",
            explanation_type="factual",
            expertise_level="advanced",
        )

        # Verify it didn't crash and fell back to raw template
        assert "Sort of ambiguous" in narrative
        assert "{missing_key}" in narrative
        assert "Base narrative." in narrative
