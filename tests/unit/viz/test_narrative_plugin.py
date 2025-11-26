"""Tests for the narrative plot plugin."""

import pytest
import numpy as np
from sklearn.datasets import load_iris, load_diabetes
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

from calibrated_explanations import CalibratedExplainer
from calibrated_explanations.viz.narrative_plugin import NarrativePlotPlugin


# Fixtures for test data and models
@pytest.fixture
def iris_data():
    """Load and split iris dataset."""
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    return X_train, X_test, y_train, y_test


@pytest.fixture
def diabetes_data():
    """Load and split diabetes dataset."""
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    return X_train, X_test, y_train, y_test


@pytest.fixture
def classification_explainer(iris_data):
    """Create a classification explainer."""
    X_train, X_test, y_train, y_test = iris_data
    # Use binary classification (classes 0 and 1 only)
    mask = y_train < 2
    X_train_binary = X_train[mask]
    y_train_binary = y_train[mask]
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train_binary, y_train_binary)
    
    explainer = CalibratedExplainer(
        model, X_train_binary, y_train_binary, feature_names=[f"feature_{i}" for i in range(4)]
    )
    return explainer


@pytest.fixture
def regression_explainer(diabetes_data):
    """Create a regression explainer."""
    X_train, X_test, y_train, y_test = diabetes_data
    
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    explainer = CalibratedExplainer(
        model, X_train, y_train, feature_names=[f"feature_{i}" for i in range(10)]
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
    assert plugin._template_path == "custom_template.yaml"


# Test with classification
def test_narrative_plugin_classification_factual_beginner(classification_explainer, iris_data):
    """Test narrative generation for binary classification with beginner level."""
    _, X_test, _, _ = iris_data
    X_test_binary = X_test[iris_data[3] < 2][:3]  # Get 3 test instances
    
    explanations = classification_explainer.explain_factual(X_test_binary)
    
    plugin = NarrativePlotPlugin()
    result = plugin.plot(
        explanations,
        expertise_level="beginner",
        output="dict"
    )
    
    assert isinstance(result, list)
    assert len(result) == 3
    assert "instance_index" in result[0]
    assert "factual_explanation_beginner" in result[0]
    assert "problem_type" in result[0]
    assert result[0]["problem_type"] == "binary_classification"


def test_narrative_plugin_classification_all_levels(classification_explainer, iris_data):
    """Test narrative generation with all expertise levels."""
    _, X_test, _, _ = iris_data
    X_test_binary = X_test[iris_data[3] < 2][:2]
    
    explanations = classification_explainer.explain_factual(X_test_binary)
    
    plugin = NarrativePlotPlugin()
    result = plugin.plot(
        explanations,
        expertise_level=("beginner", "intermediate", "advanced"),
        output="dict"
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
    _, X_test, _, _ = diabetes_data
    
    explanations = regression_explainer.explain_factual(X_test[:3])
    
    plugin = NarrativePlotPlugin()
    result = plugin.plot(
        explanations,
        expertise_level="intermediate",
        output="dict"
    )
    
    assert len(result) == 3
    assert result[0]["problem_type"] == "regression"
    assert "factual_explanation_intermediate" in result[0]


def test_narrative_plugin_probabilistic_regression(regression_explainer, diabetes_data):
    """Test narrative generation for probabilistic (thresholded) regression."""
    _, X_test, _, _ = diabetes_data
    
    # Use a threshold to make it probabilistic regression
    threshold = 150.0
    explanations = regression_explainer.explain_factual(X_test[:2], threshold=threshold)
    
    plugin = NarrativePlotPlugin()
    result = plugin.plot(
        explanations,
        expertise_level="advanced",
        output="dict"
    )
    
    assert len(result) == 2
    assert result[0]["problem_type"] == "probabilistic_regression"


# Test alternative explanations
def test_narrative_plugin_alternative_explanations(regression_explainer, diabetes_data):
    """Test narrative generation for alternative explanations."""
    _, X_test, _, _ = diabetes_data
    
    explanations = regression_explainer.explain_alternative(X_test[:2])
    
    plugin = NarrativePlotPlugin()
    result = plugin.plot(
        explanations,
        expertise_level="beginner",
        output="dict"
    )
    
    assert len(result) == 2
    # Should have alternative_explanation instead of factual_explanation
    assert "alternative_explanation_beginner" in result[0]
    assert "factual_explanation_beginner" not in result[0]


# Test output formats
def test_narrative_plugin_dataframe_output(classification_explainer, iris_data):
    """Test dataframe output format."""
    pytest.importorskip("pandas")
    
    _, X_test, _, _ = iris_data
    X_test_binary = X_test[iris_data[3] < 2][:2]
    
    explanations = classification_explainer.explain_factual(X_test_binary)
    
    plugin = NarrativePlotPlugin()
    result = plugin.plot(
        explanations,
        expertise_level="beginner",
        output="dataframe"
    )
    
    import pandas as pd
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert "instance_index" in result.columns
    assert "factual_explanation_beginner" in result.columns


def test_narrative_plugin_text_output(classification_explainer, iris_data):
    """Test text output format."""
    _, X_test, _, _ = iris_data
    X_test_binary = X_test[iris_data[3] < 2][:2]
    
    explanations = classification_explainer.explain_factual(X_test_binary)
    
    plugin = NarrativePlotPlugin()
    result = plugin.plot(
        explanations,
        expertise_level="beginner",
        output="text"
    )
    
    assert isinstance(result, str)
    assert "Instance 0" in result
    assert "Instance 1" in result
    assert "Factual Explanation" in result


def test_narrative_plugin_html_output(classification_explainer, iris_data):
    """Test HTML output format."""
    _, X_test, _, _ = iris_data
    X_test_binary = X_test[iris_data[3] < 2][:2]
    
    explanations = classification_explainer.explain_factual(X_test_binary)
    
    plugin = NarrativePlotPlugin()
    result = plugin.plot(
        explanations,
        expertise_level="beginner",
        output="html"
    )
    
    assert isinstance(result, str)
    assert "<table" in result
    assert "</table>" in result
    assert "instance_index" in result.lower()


# Test error handling
def test_narrative_plugin_invalid_expertise_level(classification_explainer, iris_data):
    """Test error handling for invalid expertise level."""
    _, X_test, _, _ = iris_data
    X_test_binary = X_test[iris_data[3] < 2][:1]
    
    explanations = classification_explainer.explain_factual(X_test_binary)
    
    plugin = NarrativePlotPlugin()
    
    with pytest.raises(ValueError, match="Invalid expertise level"):
        plugin.plot(
            explanations,
            expertise_level="expert",  # Invalid level
            output="dict"
        )


def test_narrative_plugin_invalid_output_format(classification_explainer, iris_data):
    """Test error handling for invalid output format."""
    _, X_test, _, _ = iris_data
    X_test_binary = X_test[iris_data[3] < 2][:1]
    
    explanations = classification_explainer.explain_factual(X_test_binary)
    
    plugin = NarrativePlotPlugin()
    
    with pytest.raises(ValueError, match="Invalid output format"):
        plugin.plot(
            explanations,
            expertise_level="beginner",
            output="json"  # Invalid format
        )


def test_narrative_plugin_missing_pandas_for_dataframe():
    """Test error when pandas is not available for dataframe output."""
    # This test would need to mock the pandas import, skipping for now
    pytest.skip("Requires mocking pandas import")


# Test integration with explanations.plot()
def test_narrative_via_explanations_plot(classification_explainer, iris_data):
    """Test calling narrative plugin via explanations.plot() method."""
    _, X_test, _, _ = iris_data
    X_test_binary = X_test[iris_data[3] < 2][:2]
    
    explanations = classification_explainer.explain_factual(X_test_binary)
    
    result = explanations.plot(
        style="narrative",
        expertise_level="beginner",
        output="dict"
    )
    
    assert isinstance(result, list)
    assert len(result) == 2
    assert "factual_explanation_beginner" in result[0]


def test_narrative_via_explanations_plot_dataframe(classification_explainer, iris_data):
    """Test dataframe output via explanations.plot()."""
    pytest.importorskip("pandas")
    
    _, X_test, _, _ = iris_data
    X_test_binary = X_test[iris_data[3] < 2][:2]
    
    explanations = classification_explainer.explain_factual(X_test_binary)
    
    result = explanations.plot(
        style="narrative",
        expertise_level=("beginner", "intermediate"),
        output="dataframe"
    )
    
    import pandas as pd
    assert isinstance(result, pd.DataFrame)
    assert "factual_explanation_beginner" in result.columns
    assert "factual_explanation_intermediate" in result.columns


# Test feature names
def test_narrative_plugin_feature_names(classification_explainer, iris_data):
    """Test that feature names are properly extracted and used."""
    _, X_test, _, _ = iris_data
    X_test_binary = X_test[iris_data[3] < 2][:1]
    
    explanations = classification_explainer.explain_factual(X_test_binary)
    
    plugin = NarrativePlotPlugin()
    result = plugin.plot(
        explanations,
        expertise_level="beginner",
        output="dict"
    )
    
    narrative = result[0]["factual_explanation_beginner"]
    # Check that feature names appear in the narrative
    assert "feature_" in narrative


# Test with multiple expertise levels as tuple
def test_narrative_plugin_multiple_levels_tuple(classification_explainer, iris_data):
    """Test with multiple expertise levels specified as tuple."""
    _, X_test, _, _ = iris_data
    X_test_binary = X_test[iris_data[3] < 2][:1]
    
    explanations = classification_explainer.explain_factual(X_test_binary)
    
    plugin = NarrativePlotPlugin()
    result = plugin.plot(
        explanations,
        expertise_level=("beginner", "advanced"),
        output="dict"
    )
    
    assert "factual_explanation_beginner" in result[0]
    assert "factual_explanation_advanced" in result[0]
    assert "factual_explanation_intermediate" not in result[0]
    assert result[0]["expertise_level"] == ("beginner", "advanced")


# Test edge cases
def test_narrative_plugin_empty_explanations():
    """Test handling of empty explanations list."""
    # This would require creating a mock explanations object
    pytest.skip("Requires mock explanations object")


def test_narrative_plugin_single_instance(classification_explainer, iris_data):
    """Test with a single instance."""
    _, X_test, _, _ = iris_data
    X_test_binary = X_test[iris_data[3] < 2][:1]
    
    explanations = classification_explainer.explain_factual(X_test_binary)
    
    plugin = NarrativePlotPlugin()
    result = plugin.plot(
        explanations,
        expertise_level="beginner",
        output="dict"
    )
    
    assert len(result) == 1
    assert result[0]["instance_index"] == 0
