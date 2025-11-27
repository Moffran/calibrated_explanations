"""Test narrative support in WrapCalibratedExplainer."""

import pytest
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from calibrated_explanations.core.wrap_explainer import WrapCalibratedExplainer


@pytest.fixture
def fitted_wrapper():
    """Create a fitted and calibrated WrapCalibratedExplainer."""
    X, y = load_iris(return_X_y=True)
    # Binary classification
    mask = y < 2
    X, y = X[mask], y[mask]
    
    X_train, X_cal, y_train, y_cal = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    
    wrapper = WrapCalibratedExplainer(model)
    wrapper.fit(X_train, y_train)
    wrapper.calibrate(
        X_cal, y_cal,
        feature_names=["sepal_length", "sepal_width", "petal_length", "petal_width"]
    )
    
    return wrapper, X[80:83]  # Return wrapper and test data


def test_wrapper_plot_narrative_dict(fitted_wrapper):
    """Test narrative generation via wrapper.plot() with dict output."""
    wrapper, X_test = fitted_wrapper
    
    result = wrapper.plot(
        X_test,
        style="narrative",
        expertise_level="beginner",
        output="dict"
    )
    
    assert isinstance(result, list)
    assert len(result) == 3
    assert "factual_explanation_beginner" in result[0]
    assert "instance_index" in result[0]
    assert "problem_type" in result[0]


def test_wrapper_plot_narrative_dataframe(fitted_wrapper):
    """Test narrative generation via wrapper.plot() with dataframe output."""
    pytest.importorskip("pandas")
    
    wrapper, X_test = fitted_wrapper
    
    result = wrapper.plot(
        X_test,
        style="narrative",
        expertise_level=("beginner", "advanced"),
        output="dataframe"
    )
    
    import pandas as pd
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert "factual_explanation_beginner" in result.columns
    assert "factual_explanation_advanced" in result.columns


def test_wrapper_plot_narrative_text(fitted_wrapper):
    """Test narrative generation via wrapper.plot() with text output."""
    wrapper, X_test = fitted_wrapper
    
    result = wrapper.plot(
        X_test,
        style="narrative",
        expertise_level="intermediate",
        output="text"
    )
    
    assert isinstance(result, str)
    assert "Instance 0" in result
    assert "Factual Explanation" in result


def test_wrapper_plot_regular_style(fitted_wrapper):
    """Test that regular plot style still works."""
    wrapper, X_test = fitted_wrapper
    
    # Regular style should not return anything (returns None)
    result = wrapper.plot(
        X_test[:1],
        style="regular",
        show=False
    )
    
    assert result is None


def test_wrapper_explain_with_narrative(fitted_wrapper):
    """Test the existing explain_with_narrative method."""
    wrapper, X_test = fitted_wrapper
    
    result = wrapper.explain_with_narrative(
        X_test,
        expertise_level="beginner",
        return_dataframe=True
    )
    
    # Should return dataframe or dict
    assert result is not None
