import pytest
import numpy as np
from calibrated_explanations.core.narrative_generator import (
    to_py,
    clean_condition,
    crosses_zero,
    has_wide_prediction_interval,
    NarrativeGenerator,
    load_template_file,
)
from calibrated_explanations.utils.exceptions import ValidationError
from unittest.mock import MagicMock


def test_to_py_variants():
    assert to_py(np.bool_(True)) is True
    assert to_py("string") == "string"
    assert to_py(np.array([1, 2])).__class__ is list


def test_clean_condition_variants():
    assert clean_condition("", "feat") == ""
    assert clean_condition("rule", None) == "rule"
    assert clean_condition("rule", "") == "rule"
    # Regex failure fallback (using invalid regex char in feat_name if possible,
    # but re.escape handles it. Let's mock re.sub to fail)
    import re

    original_sub = re.sub
    try:

        def mock_sub(*args, **kwargs):
            raise Exception("regex fail")

        re.sub = mock_sub
        assert clean_condition("feat rule", "feat") == "feat rule"
    finally:
        re.sub = original_sub


def test_clean_condition_should_reraise_non_exception_baseexception(monkeypatch):
    import re

    def raise_keyboard_interrupt(*_args, **_kwargs):
        raise KeyboardInterrupt("stop")

    monkeypatch.setattr(re, "sub", raise_keyboard_interrupt)
    with pytest.raises(KeyboardInterrupt):
        clean_condition("feature > 1", "feature")


def test_crosses_zero_fallback():
    assert crosses_zero({"weight_low": "abc", "weight_high": 1}) is False


def test_crosses_zero_should_handle_array_like_intervals():
    assert crosses_zero({"weight_low": [-0.1, 0.2], "weight_high": [0.1, 0.3]}) is True
    assert crosses_zero({"weight_low": [0.1, 0.2], "weight_high": [0.3, 0.4]}) is False


def test_crosses_zero_should_handle_mismatched_interval_lengths():
    assert crosses_zero({"weight_low": [-0.2, 0.3], "weight_high": [0.1]}) is True


def test_has_wide_prediction_interval_handles_invalid_values() -> None:
    assert has_wide_prediction_interval({"predict_low": "bad", "predict_high": 1.0}) is False


def test_load_template_file_error_paths(tmp_path, monkeypatch):
    from calibrated_explanations.core import narrative_generator as ng_mod
    from calibrated_explanations.utils.exceptions import SerializationError

    with pytest.raises(SerializationError, match="Template file not found"):
        load_template_file(str(tmp_path / "missing.json"))

    unsupported = tmp_path / "template.txt"
    unsupported.write_text("x", encoding="utf-8")
    with pytest.raises(SerializationError, match="Unsupported template file format"):
        load_template_file(str(unsupported))

    yaml_path = tmp_path / "template.yaml"
    yaml_path.write_text("a: [unterminated", encoding="utf-8")
    if getattr(ng_mod, "yaml", None) is not None:
        with pytest.raises(SerializationError, match="Failed to parse YAML template"):
            load_template_file(str(yaml_path))
    else:
        with pytest.raises(SerializationError, match="YAML support requires pyyaml"):
            load_template_file(str(yaml_path))


def test_generate_narrative_should_format_interval_threshold_context() -> None:
    gen = NarrativeGenerator()
    gen.templates = {
        "narrative_templates": {
            "probabilistic_regression": {
                "factual": {
                    "beginner": "Prediction: {calibrated_pred}; Condition: {threshold_condition}",
                }
            }
        }
    }
    exp = MagicMock()
    exp.get_rules.return_value = {
        "base_predict": [0.42],
        "base_predict_low": [0.21],
        "base_predict_high": [0.63],
        "rule": [],
    }
    out = gen.generate_narrative(
        exp,
        "probabilistic_regression",
        explanation_type="factual",
        expertise_level="beginner",
        threshold=(0.1, 0.9),
    )
    assert "0.100 < target <= 0.900" in out


def test_serialize_rules_should_extract_feature_name_fallbacks() -> None:
    gen = NarrativeGenerator()
    rules = {
        "rule": ["", "city in inland", "income <= 2.0"],
        "feature": [None, "not-an-int", 99],
        "predict": [0.1, 0.2, 0.3],
    }
    serialized = gen.serialize_rules(rules, feature_names=["f0"])
    assert serialized[0]["feature_name"] == ""
    assert serialized[1]["feature_name"] == "city"
    assert serialized[2]["feature_name"] == "income"


def test_expand_template_should_cover_conjunctive_raw_and_fallback_paths() -> None:
    gen = NarrativeGenerator()
    template = "Rule: {feature_name} {condition} then {predict} [{predict_low}, {predict_high}]"
    pos_features = [
        {
            "feature_name": "x",
            "rule": "rawsegment",
            "value": None,
            "weight": 0.3,
            "predict": 0.6,
            "predict_low": None,
            "predict_high": 0.9,
            "is_conjunctive": True,
        },
        {
            "feature_name": None,
            "rule": "",
            "value": None,
            "weight": 0.2,
            "predict": 0.55,
            "predict_low": 0.40,
            "predict_high": 0.70,
            "is_conjunctive": True,
        },
    ]
    out = gen.expand_template(
        template,
        pos_features,
        neg_features=[],
        uncertain_features=[],
        context={"pred_interval_lower": "N/A", "pred_interval_upper": "N/A"},
        level="advanced",
        problem_type="probabilistic_regression",
        explanation_type="alternative",
        conjunction_separator=" AND ",
        align_weights=False,
    )
    assert "rawsegment" in out


def test_narrative_generator_validation():
    gen = NarrativeGenerator()
    # Templates not loaded
    with pytest.raises(ValidationError, match="Templates not loaded"):
        gen.generate_narrative(MagicMock(), "regression")

    # No get_rules method
    gen.templates = {"narrative_templates": {}}
    with pytest.raises(ValidationError, match="Explanation has no get_rules method"):
        gen.generate_narrative(object(), "regression")


def test_generate_narrative_label_fallbacks():
    gen = NarrativeGenerator()
    gen.templates = {
        "narrative_templates": {
            "binary_classification": {"factual": {"beginner": "Template {label} {calibrated_pred}"}}
        }
    }

    # No get_class_labels
    mock_exp = MagicMock()
    mock_exp.get_rules.return_value = {"classes": 1, "base_predict": [0.8]}
    del mock_exp.get_class_labels
    res = gen.generate_narrative(mock_exp, "binary_classification", expertise_level="beginner")
    assert "Template 1" in res

    # Fallback to infer from prediction
    mock_exp2 = MagicMock()
    mock_exp2.get_rules.return_value = {"base_predict": [0.8]}
    res2 = gen.generate_narrative(mock_exp2, "binary_classification", expertise_level="beginner")
    assert "Template 1" in res2

    mock_exp3 = MagicMock()
    mock_exp3.get_rules.return_value = {"base_predict": [0.2]}
    res3 = gen.generate_narrative(mock_exp3, "binary_classification", expertise_level="beginner")
    assert "Template 0" in res3


def test_generate_narrative_template_not_found():
    gen = NarrativeGenerator()
    gen.templates = {"narrative_templates": {}}
    mock_exp = MagicMock()
    mock_exp.get_rules.return_value = {}
    res = gen.generate_narrative(mock_exp, "regression")
    assert "Template not found" in res


def test_expand_template_feat_name_fallback():
    gen = NarrativeGenerator()
    template = "{feature_name}"
    # feature_name is None, fallback to rule.split()[0]
    pos_features = [{"feature_name": None, "rule": "f1 > 0", "weight": 1.0}]
    res = gen.expand_template(template, pos_features, [], [], {}, "beginner")
    assert "f1" in res

    # rule is empty
    pos_features2 = [{"feature_name": None, "rule": "", "weight": 1.0}]
    res2 = gen.expand_template(template, pos_features2, [], [], {}, "beginner")
    assert res2 == ""


def test_generate_narrative_should_not_crash_for_alternative_conjunctive_features():
    from calibrated_explanations.explanations.explanation import AlternativeExplanation

    class AltStub(AlternativeExplanation):
        def __init__(self):
            pass

        def get_explainer(self):
            return MagicMock(feature_names=["f1", "median_income"])

        def get_rules(self):
            return {
                "base_predict": [0.5],
                "base_predict_low": [0.4],
                "base_predict_high": [0.6],
                "rule": ["f1 <= -121.83 & \nmedian_income > 4.92"],
                "value": ["-122.17\n7.62"],
                "feature": [[0, 1]],
                "weight": [0.1],
                "weight_low": [0.0],
                "weight_high": [0.2],
                "predict": [0.55],
                "predict_low": [0.45],
                "predict_high": [0.65],
            }

    gen = NarrativeGenerator()
    gen.templates = {
        "narrative_templates": {
            "probabilistic_regression": {
                "alternative": {
                    "beginner": "* {feature_name} ({feature_actual_value}) {condition}",
                }
            }
        }
    }

    exp = AltStub()
    exp.prediction = {"predict": 0.5, "low": 0.4, "high": 0.6}

    res = gen.generate_narrative(
        exp,
        "probabilistic_regression",
        explanation_type="alternative",
        expertise_level="beginner",
        conjunction_separator=" AND ",
        align_weights=False,
    )

    assert "Error generating narrative" not in res
    assert "(f1 (-122.17) <= -121.83 AND median_income (7.62) > 4.92)" in res


def test_expand_template_should_tag_uncertain_for_alternatives_when_interval_covers_point_five():
    gen = NarrativeGenerator()
    template = "\n".join(
        [
            "Alternatives to increase:",
            "- If {feature_name} {condition} then Calibrated Probability {predict} [{predict_low}, {predict_high}]",
            "Alternatives to decrease:",
            "- If {feature_name} {condition} then Calibrated Probability {predict} [{predict_low}, {predict_high}]",
        ]
    )

    pos_features = [
        {
            "feature_name": "total_bedrooms",
            "rule": "total_bedrooms < 9.00",
            "value": "8",
            "weight": 0.01,
            "predict": 0.45,
            "predict_low": 0.42,
            "predict_high": 0.48,
            "is_conjunctive": False,
        },
        {
            "feature_name": "housing_median_age",
            "rule": "housing_median_age < 13.50 & \nocean_proximity = INLAND & \nmedian_income > 3.23",
            "value": "10\nINLAND\n4.0",
            "weight": 0.02,
            "predict": 0.46,
            "predict_low": 0.38,  # covers 0.5 -> uncertain
            "predict_high": 0.50,
            "is_conjunctive": True,
        },
    ]

    neg_features = [
        {
            "feature_name": "median_income",
            "rule": "median_income > 2.0",
            "value": "3.0",
            "weight": -0.02,
            "predict": 0.30,
            "predict_low": 0.25,
            "predict_high": 0.35,
            "is_conjunctive": False,
        },
        {
            "feature_name": "f2",
            "rule": "f2 > 1.0",
            "value": "2.0",
            "weight": -0.02,
            "predict": 0.49,
            "predict_low": 0.48,
            "predict_high": 0.52,  # covers 0.5 -> uncertain
            "is_conjunctive": False,
        },
    ]

    res = gen.expand_template(
        template,
        pos_features,
        neg_features,
        [],
        context={},
        level="advanced",
        problem_type="probabilistic_regression",
        explanation_type="alternative",
        base_predict=None,
        conjunction_separator=" AND ",
        align_weights=False,
    )

    lines = [ln.strip() for ln in res.splitlines() if ln.strip().startswith("-")]

    total_bedrooms_line = next(ln for ln in lines if "total_bedrooms < 9.00" in ln)
    assert "⚠️ direction uncertain" not in total_bedrooms_line
    assert "⚠️ uncertain" not in total_bedrooms_line

    housing_line = next(ln for ln in lines if "housing_median_age" in ln)
    assert "⚠️ direction uncertain" not in housing_line
    assert "⚠️ uncertain" in housing_line


def test_generate_narrative_should_not_split_uncertainty_for_regression():
    gen = NarrativeGenerator()
    # Mock template with explicit pos/neg sections but NO uncertain section
    template = "\n".join(
        [
            "Increase:",
            "- {feature_name} inc",
            "Decrease:",
            "- {feature_name} dec",
        ]
    )
    # Patch load_templates to return this
    gen.templates = {"narrative_templates": {"regression": {"alternative": {"advanced": template}}}}

    # Features with WIDE intervals (should be uncertain if using standard logic)
    # In regression, wide means huge numbers, which are standard.
    # Width = 150000. threshold=0.20. 150000 > 0.20 -> Uncertain.

    # Mock explanation
    # Use spec to ensure it doesn't have private attributes automatically
    mock_exp = MagicMock(spec=["get_rules"])
    mock_exp.get_rules.return_value = {
        "rule": ["f1 > 0", "f2 < 0"],
        "feature_name": ["f1", "f2"],
        "predict": [200000, 198000],
        "predict_low": [100000, 100000],
        "predict_high": [250000, 250000],
        "base_predict": [200000],
        "weight": [1000, -1000],
    }

    res = gen.generate_narrative(
        mock_exp,
        problem_type="regression",
        explanation_type="alternative",
        expertise_level="advanced",
    )

    # If splitting DISABLED (fix applied), Decrease section should be populated
    assert "Decrease:" in res
    assert "f2 dec" in res

    assert "f1 inc" in res
