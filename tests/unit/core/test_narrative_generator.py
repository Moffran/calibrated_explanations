import pytest
import numpy as np
from calibrated_explanations.core.narrative_generator import (
    load_template_file,
    to_py,
    _first_or_none,
    clean_condition,
    crosses_zero,
    has_wide_prediction_interval,
    NarrativeGenerator,
)
from calibrated_explanations.utils.exceptions import SerializationError, ValidationError
from unittest.mock import MagicMock


def test_load_template_file_errors(tmp_path):
    # File not found
    with pytest.raises(SerializationError, match="Template file not found"):
        load_template_file(str(tmp_path / "nonexistent.json"))

    # Unsupported format
    unsupported = tmp_path / "test.txt"
    unsupported.write_text("hello")
    with pytest.raises(SerializationError, match="Unsupported template file format"):
        load_template_file(str(unsupported))

    # Invalid JSON
    invalid_json = tmp_path / "test.json"
    invalid_json.write_text("{invalid")
    with pytest.raises(SerializationError, match="Failed to parse JSON template"):
        load_template_file(str(invalid_json))


def test_to_py_variants():
    assert to_py(np.bool_(True)) is True
    assert to_py("string") == "string"


def test_first_or_none_variants():
    assert _first_or_none(None) is None
    assert _first_or_none([]) is None
    assert _first_or_none([1, 2]) == 1
    assert _first_or_none(5) == 5


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


def test_crosses_zero_fallback():
    assert crosses_zero({"weight_low": "abc", "weight_high": 1}) is False


def test_has_wide_prediction_interval_fallback():
    assert has_wide_prediction_interval({"predict_low": "abc", "predict_high": 1}) is False


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


def test_serialize_rules_variants():
    gen = NarrativeGenerator()
    # Invalid feature index
    rules_dict = {"rule": ["feat > 5"], "feature": [10]}
    res = gen._serialize_rules(rules_dict, feature_names=["a", "b"])
    assert res[0]["feature_name"] == "feat"  # extracted from rule


def test_expand_template_caution_logic():
    gen = NarrativeGenerator()
    template = "Template {feature_name}"
    pos_features = [{"feature_name": "f1", "rule": "f1 > 0", "weight": 1.0}]
    context = {
        "pred_interval_lower": "0.1",
        "pred_interval_upper": "0.5",  # width 0.4 > 0.2
    }
    # Beginner level caution
    res = gen._expand_template(
        template, pos_features, [], [], context, "beginner", "binary_classification"
    )
    assert "⚠️ Use caution: uncertainty is high." in res

    # Advanced level caution
    res2 = gen._expand_template(
        template, pos_features, [], [], context, "advanced", "binary_classification"
    )
    assert "calibrated probability interval is wide (0.400)" in res2


def test_expand_template_feat_name_fallback():
    gen = NarrativeGenerator()
    template = "{feature_name}"
    # feature_name is None, fallback to rule.split()[0]
    pos_features = [{"feature_name": None, "rule": "f1 > 0", "weight": 1.0}]
    res = gen._expand_template(template, pos_features, [], [], {}, "beginner")
    assert "f1" in res

    # rule is empty
    pos_features2 = [{"feature_name": None, "rule": "", "weight": 1.0}]
    res2 = gen._expand_template(template, pos_features2, [], [], {}, "beginner")
    assert res2 == ""
