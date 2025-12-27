"""Tests for the narrative plot plugin."""

from types import SimpleNamespace

import pytest
from sklearn.datasets import load_diabetes, load_iris
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

from calibrated_explanations import CalibratedExplainer
from calibrated_explanations.viz import NarrativePlotPlugin
from calibrated_explanations.utils.exceptions import SerializationError, ValidationError
from calibrated_explanations.core.narrative_generator import NarrativeGenerator, load_template_file


# Fixtures for test data and models
@pytest.fixture
def iris_data():
    """Load and split iris dataset."""
    x, y = load_iris(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    return x_train, x_test, y_train, y_test


@pytest.fixture
def diabetes_data():
    """Load and split diabetes dataset."""
    x, y = load_diabetes(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    return x_train, x_test, y_train, y_test


@pytest.fixture
def classification_explainer(iris_data):
    """Create a classification explainer."""
    x_train, x_test, y_train, y_test = iris_data
    # Use binary classification (classes 0 and 1 only)
    mask = y_train < 2
    x_train_binary = x_train[mask]
    y_train_binary = y_train[mask]

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(x_train_binary, y_train_binary)

    explainer = CalibratedExplainer(
        model, x_train_binary, y_train_binary, feature_names=[f"feature_{i}" for i in range(4)]
    )
    return explainer


@pytest.fixture
def regression_explainer(diabetes_data):
    """Create a regression explainer."""
    x_train, x_test, y_train, y_test = diabetes_data

    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(x_train, y_train)

    explainer = CalibratedExplainer(
        model,
        x_train,
        y_train,
        mode="regression",
        feature_names=[f"feature_{i}" for i in range(10)],
    )
    return explainer


# Test basic functionality
def test_narrative_plugin_initialization():
    """Test that the plugin can be initialized."""
    plugin = NarrativePlotPlugin()
    assert plugin is not None
    assert plugin.default_template is not None


def test_narrative_plugin_with_custom_template():
    """Test plugin initialization with custom template path."""
    plugin = NarrativePlotPlugin(template_path="custom_template.yaml")
    assert plugin.template_path == "custom_template.yaml"


# Test with classification
def test_narrative_plugin_classification_factual_beginner(classification_explainer, iris_data):
    """Test narrative generation for binary classification with beginner level."""
    _, x_test, _, _ = iris_data
    x_test_binary = x_test[iris_data[3] < 2][:3]  # Get 3 test instances

    explanations = classification_explainer.explain_factual(x_test_binary)

    plugin = NarrativePlotPlugin()
    result = plugin.plot(explanations, expertise_level="beginner", output="dict")

    assert isinstance(result, list)
    assert len(result) == 3
    assert "instance_index" in result[0]
    assert "factual_explanation_beginner" in result[0]
    assert "problem_type" in result[0]
    assert result[0]["problem_type"] == "binary_classification"


def test_narrative_plugin_classification_all_levels(classification_explainer, iris_data):
    """Test narrative generation with all expertise levels."""
    _, x_test, _, _ = iris_data
    x_test_binary = x_test[iris_data[3] < 2][:2]

    explanations = classification_explainer.explain_factual(x_test_binary)

    plugin = NarrativePlotPlugin()
    result = plugin.plot(
        explanations, expertise_level=("beginner", "intermediate", "advanced"), output="dict"
    )

    assert len(result) == 2
    assert "factual_explanation_beginner" in result[0]
    assert "factual_explanation_intermediate" in result[0]
    assert "factual_explanation_advanced" in result[0]

    # Check that narratives are different for different levels
    assert result[0]["factual_explanation_beginner"] != result[0]["factual_explanation_advanced"]


# Test with regression
def test_narrative_plugin_regression_factual(regression_explainer, diabetes_data):
    """Test narrative generation for regression."""
    _, x_test, _, _ = diabetes_data

    explanations = regression_explainer.explain_factual(x_test[:3])

    plugin = NarrativePlotPlugin()
    result = plugin.plot(explanations, expertise_level="intermediate", output="dict")

    assert len(result) == 3
    assert result[0]["problem_type"] == "regression"
    assert "factual_explanation_intermediate" in result[0]


def test_narrative_plugin_probabilistic_regression(regression_explainer, diabetes_data):
    """Test narrative generation for probabilistic (thresholded) regression."""
    _, x_test, _, _ = diabetes_data

    # Use a threshold to make it probabilistic regression
    threshold = 150.0
    explanations = regression_explainer.explain_factual(x_test[:2], threshold=threshold)

    plugin = NarrativePlotPlugin()
    result = plugin.plot(explanations, expertise_level="advanced", output="dict")

    assert len(result) == 2
    assert result[0]["problem_type"] == "probabilistic_regression"


# Test alternative explanations
def test_narrative_plugin_alternative_explanations(regression_explainer, diabetes_data):
    """Test narrative generation for alternative explanations."""
    _, x_test, _, _ = diabetes_data

    explanations = regression_explainer.explore_alternatives(x_test[:2])

    plugin = NarrativePlotPlugin()
    result = plugin.plot(explanations, expertise_level="beginner", output="dict")

    assert len(result) == 2
    # Should have alternative_explanation instead of factual_explanation
    assert "alternative_explanation_beginner" in result[0]
    assert "factual_explanation_beginner" not in result[0]


# Test output formats
def test_narrative_plugin_dataframe_output(classification_explainer, iris_data):
    """Test dataframe output format."""
    pytest.importorskip("pandas")

    _, x_test, _, _ = iris_data
    x_test_binary = x_test[iris_data[3] < 2][:2]

    explanations = classification_explainer.explain_factual(x_test_binary)

    plugin = NarrativePlotPlugin()
    result = plugin.plot(explanations, expertise_level="beginner", output="dataframe")

    import pandas as pd

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert "instance_index" in result.columns
    assert "factual_explanation_beginner" in result.columns


def test_narrative_plugin_text_output(classification_explainer, iris_data):
    """Test text output format."""
    _, x_test, _, _ = iris_data
    x_test_binary = x_test[iris_data[3] < 2][:2]

    explanations = classification_explainer.explain_factual(x_test_binary)

    plugin = NarrativePlotPlugin()
    result = plugin.plot(explanations, expertise_level="beginner", output="text")

    assert isinstance(result, str)
    assert "Instance 0" in result
    assert "Instance 1" in result


def test_narrative_generator_requires_templates():
    generator = NarrativeGenerator()
    explanation = SimpleNamespace(
        get_rules=lambda: {"rule": [], "base_predict": [0.5]},
        get_class_labels=lambda: {"0": "zero"},
    )

    with pytest.raises(ValidationError):
        generator.generate_narrative(
            explanation,
            problem_type="binary_classification",
            expertise_level="beginner",
        )


def test_narrative_generator_emits_caution_and_uncertainty_tags():
    generator = NarrativeGenerator()
    generator.templates = {
        "narrative_templates": {
            "binary_classification": {
                "factual": {
                    "advanced": "\n".join(
                        [
                            "Lead: {label} {calibrated_pred}",
                            "POS {feature_name}: {feature_weight} [{feature_weight_low}, {feature_weight_high}]",
                            "NEG {feature_name}: {feature_weight} [{feature_weight_low}, {feature_weight_high}]",
                            "UNC {feature_name}: {feature_weight}",
                        ]
                    )
                }
            }
        }
    }
    rules_dict = {
        "rule": ["age >= 30", "income <= 5", "savings >= 1"],
        "feature": [0, 1, 2],
        "weight": [0.5, -0.4, 0.2],
        "weight_low": [0.3, -0.5, -0.1],
        "weight_high": [0.7, -0.2, 0.6],
        "value": [42, 3, 2],
        "predict": [0.65, 0.25, 0.85],
        "predict_low": [0.60, 0.20, 0.05],
        "predict_high": [0.70, 0.35, 0.90],
        "base_predict": [0.8],
        "base_predict_low": [0.1],
        "base_predict_high": [0.95],
        "classes": 1,
    }
    explanation = SimpleNamespace(
        get_rules=lambda: rules_dict,
        get_class_labels=lambda: {1: "Positive"},
    )

    narrative = generator.generate_narrative(
        explanation,
        problem_type="binary_classification",
        explanation_type="factual",
        expertise_level="advanced",
        feature_names=["age", "income", "savings"],
    )

    assert narrative.startswith("⚠️ Use caution")
    assert "POS age" in narrative
    assert "NEG income" in narrative
    assert "UNC savings" in narrative
    assert "⚠️ highly uncertain" in narrative
    assert "⚠️ direction uncertain" in narrative


def test_load_template_file_validates_format(tmp_path):
    bad_json = tmp_path / "template.json"
    bad_json.write_text("{broken", encoding="utf-8")

    with pytest.raises(SerializationError):
        load_template_file(str(bad_json))

    bad_format = tmp_path / "template.txt"
    bad_format.write_text("noop", encoding="utf-8")

    with pytest.raises(SerializationError):
        load_template_file(str(bad_format))


def test_load_template_file_requires_yaml_dependency(tmp_path, monkeypatch):
    yaml_file = tmp_path / "template.yaml"
    yaml_file.write_text("key: value", encoding="utf-8")

    import calibrated_explanations.core.narrative_generator as generator_module

    monkeypatch.setattr(generator_module, "yaml", None)

    with pytest.raises(SerializationError):
        load_template_file(str(yaml_file))


def test_narrative_plugin_html_output(classification_explainer, iris_data):
    """Test HTML output format."""
    _, x_test, _, _ = iris_data
    x_test_binary = x_test[iris_data[3] < 2][:2]

    explanations = classification_explainer.explain_factual(x_test_binary)

    plugin = NarrativePlotPlugin()
    result = plugin.plot(explanations, expertise_level="beginner", output="html")

    assert isinstance(result, str)
    assert "<table" in result
    assert "</table>" in result
    assert "instance_index" in result.lower()


# Test error handling
def test_narrative_plugin_invalid_expertise_level(classification_explainer, iris_data):
    """Test error handling for invalid expertise level."""
    from calibrated_explanations.utils.exceptions import ValidationError

    _, x_test, _, _ = iris_data
    x_test_binary = x_test[iris_data[3] < 2][:1]

    explanations = classification_explainer.explain_factual(x_test_binary)

    plugin = NarrativePlotPlugin()

    with pytest.raises(ValidationError, match="Invalid expertise level"):
        plugin.plot(
            explanations,
            expertise_level="expert",  # Invalid level
            output="dict",
        )


def test_narrative_plugin_invalid_output_format(classification_explainer, iris_data):
    """Test error handling for invalid output format."""
    from calibrated_explanations.utils.exceptions import ValidationError

    _, x_test, _, _ = iris_data
    x_test_binary = x_test[iris_data[3] < 2][:1]

    explanations = classification_explainer.explain_factual(x_test_binary)

    plugin = NarrativePlotPlugin()

    with pytest.raises(ValidationError, match="Invalid output format"):
        plugin.plot(
            explanations,
            expertise_level="beginner",
            output="json",  # Invalid format
        )


def test_narrative_plugin_missing_pandas_for_dataframe(
    classification_explainer, iris_data, monkeypatch
):
    """Test error when pandas is not available for dataframe output."""
    from calibrated_explanations.viz import narrative_plugin

    # Simulate pandas not being available
    monkeypatch.setattr(narrative_plugin, "_PANDAS_AVAILABLE", False)

    _, x_test, _, _ = iris_data
    x_test_binary = x_test[iris_data[3] < 2][:2]

    explanations = classification_explainer.explain_factual(x_test_binary)

    with pytest.raises(ImportError, match="Pandas is required"):
        explanations.plot(style="narrative", output="dataframe")


# Test integration with explanations.plot()
def test_narrative_via_explanations_plot(classification_explainer, iris_data):
    """Test calling narrative plugin via explanations.plot() method."""
    _, x_test, _, _ = iris_data
    x_test_binary = x_test[iris_data[3] < 2][:2]

    explanations = classification_explainer.explain_factual(x_test_binary)

    result = explanations.plot(style="narrative", expertise_level="beginner", output="dict")

    assert isinstance(result, list)
    assert len(result) == 2
    assert "factual_explanation_beginner" in result[0]


def test_narrative_via_explanations_plot_dataframe(classification_explainer, iris_data):
    """Test dataframe output via explanations.plot()."""
    pytest.importorskip("pandas")

    _, x_test, _, _ = iris_data
    x_test_binary = x_test[iris_data[3] < 2][:2]

    explanations = classification_explainer.explain_factual(x_test_binary)

    result = explanations.plot(
        style="narrative", expertise_level=("beginner", "intermediate"), output="dataframe"
    )

    import pandas as pd

    assert isinstance(result, pd.DataFrame)
    assert "factual_explanation_beginner" in result.columns
    assert "factual_explanation_intermediate" in result.columns


# Test feature names
def test_narrative_plugin_feature_names(classification_explainer, iris_data):
    """Test that feature names are properly extracted and used."""
    _, x_test, _, _ = iris_data
    x_test_binary = x_test[iris_data[3] < 2][:1]

    explanations = classification_explainer.explain_factual(x_test_binary)

    plugin = NarrativePlotPlugin()
    result = plugin.plot(explanations, expertise_level="beginner", output="dict")

    narrative = result[0]["factual_explanation_beginner"]
    # Check that feature names appear in the narrative
    assert "feature_" in narrative


# Test with multiple expertise levels as tuple
def test_narrative_plugin_multiple_levels_tuple(classification_explainer, iris_data):
    """Test with multiple expertise levels specified as tuple."""
    _, x_test, _, _ = iris_data
    x_test_binary = x_test[iris_data[3] < 2][:1]

    explanations = classification_explainer.explain_factual(x_test_binary)

    plugin = NarrativePlotPlugin()
    result = plugin.plot(explanations, expertise_level=("beginner", "advanced"), output="dict")

    assert "factual_explanation_beginner" in result[0]
    assert "factual_explanation_advanced" in result[0]
    assert "factual_explanation_intermediate" not in result[0]
    assert result[0]["expertise_level"] == ("beginner", "advanced")


# Test edge cases
def test_narrative_plugin_empty_explanations():
    """Test handling of empty explanations list."""
    plugin = NarrativePlotPlugin()

    class MockExplainer:
        mode = "classification"

        def is_multiclass(self):
            return False

        feature_names = ["f1", "f2"]

    explanations = SimpleNamespace(
        explanations=[], calibrated_explainer=MockExplainer(), y_threshold=None
    )

    result = plugin.plot(explanations, output="dict")
    assert result == []


def test_narrative_plugin_single_instance(classification_explainer, iris_data):
    """Test with a single instance."""
    _, x_test, _, _ = iris_data
    x_test_binary = x_test[iris_data[3] < 2][:1]

    explanations = classification_explainer.explain_factual(x_test_binary)

    plugin = NarrativePlotPlugin()
    result = plugin.plot(explanations, expertise_level="beginner", output="dict")

    assert len(result) == 1
    assert result[0]["instance_index"] == 0


def test_narrative_plugin_template_fallback(enable_fallbacks, tmp_path, monkeypatch):
    """Ensure missing templates fall back to the default path.

    This test explicitly validates template fallback behavior.
    """
    default_template = tmp_path / "templates" / "explain_template.yaml"
    default_template.parent.mkdir(parents=True, exist_ok=True)
    default_template.write_text("stub", encoding="utf-8")

    captured = {}

    class FakeNarrator:
        def __init__(self, template):
            captured["template"] = template

        def generate_narrative(
            self,
            explanation,
            problem_type,
            explanation_type,
            expertise_level,
            threshold,
            feature_names,
        ):
            return f"{explanation_type}:{expertise_level}:{problem_type}"

    monkeypatch.setattr(
        NarrativePlotPlugin,
        "_get_default_template_path",
        staticmethod(lambda: str(default_template)),
    )
    monkeypatch.setattr(
        "calibrated_explanations.viz.narrative_plugin.NarrativeGenerator",
        FakeNarrator,
    )

    class StubExplainer:
        def __init__(self):
            self.mode = "classification"
            self._explainer = SimpleNamespace(feature_names=["f0"])

        def is_multiclass(self):
            return False

    explanations = SimpleNamespace(
        explanations=[SimpleNamespace(y_threshold=None)],
        calibrated_explainer=StubExplainer(),
        y_threshold=None,
    )
    plugin = NarrativePlotPlugin()
    with pytest.warns(UserWarning, match=r"fall.*back|template"):
        result = plugin.plot(explanations, template_path="missing.yaml", output="dict")

    assert captured["template"] == str(default_template)
    assert result[0]["problem_type"] == "binary_classification"
    assert "factual_explanation_beginner" in result[0]


def test_narrative_plugin_detect_problem_type_variants():
    """Cover detection branches and exception handling."""
    plugin = NarrativePlotPlugin()

    class Explainer:
        def __init__(self, mode):
            self.mode = mode

        def is_multiclass(self):
            return True

    explanations = SimpleNamespace(
        explanations=[], calibrated_explainer=Explainer("classification"), y_threshold=None
    )
    assert plugin._detect_problem_type(explanations) == "multiclass_classification"

    explanations.calibrated_explainer.is_multiclass = lambda: False
    assert plugin._detect_problem_type(explanations) == "binary_classification"

    explanations.y_threshold = 0.5
    assert plugin._detect_problem_type(explanations) == "probabilistic_regression"

    explanations.y_threshold = None
    explanations.calibrated_explainer.mode = "regression"
    assert plugin._detect_problem_type(explanations) == "regression"

    def broken_multiclass():
        raise RuntimeError("boom")

    explanations.calibrated_explainer.mode = "other"
    explanations.calibrated_explainer.is_multiclass = broken_multiclass
    assert plugin._detect_problem_type(explanations) == "regression"

    explanations.calibrated_explainer = SimpleNamespace()
    assert plugin._detect_problem_type(explanations) == "regression"


def test_narrative_plugin_feature_name_helpers(monkeypatch):
    """Verify feature-name extraction and alternative detection."""
    plugin = NarrativePlotPlugin()
    explainer = SimpleNamespace(
        _explainer=SimpleNamespace(feature_names=["a", "b"]), feature_names=None
    )
    explanations = SimpleNamespace(
        explanations=[SimpleNamespace()],
        calibrated_explainer=explainer,
    )
    assert plugin._get_feature_names(explanations) == ["a", "b"]

    explainer._explainer = SimpleNamespace()
    explainer.feature_names = ("x",)
    assert plugin._get_feature_names(explanations) == ("x",)

    explainer.feature_names = None
    assert plugin._get_feature_names(explanations) is None

    class AltExplanations(list):
        pass

    alt = AltExplanations()
    alt.__class__.__name__ = "AlternativeExplanations"
    assert plugin._is_alternative(alt) is True

    alt = SimpleNamespace(_is_alternative=lambda: True)
    assert plugin._is_alternative(alt) is True
