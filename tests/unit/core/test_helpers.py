# pylint: disable=line-too-long, invalid-name, missing-function-docstring, missing-class-docstring, unused-argument
"""
This module contains unit tests for the helper functions in the calibrated_explanations.utils.helper module.
The tests cover the following functionalities:
- safe_import: A function to safely import modules or classes, handling ImportError and AttributeError.
- safe_isinstance: A function to safely check if an object is an instance of a specified class, given the class path as a string.
Test cases include:
- Importing modules and classes successfully.
- Handling ImportError when the module is not installed.
- Handling AttributeError when the class does not exist in the module.
- Checking isinstance with single and multiple class path strings.
- Handling invalid class path strings.
- Handling cases where the class module is not imported.
"""

import sys
import types

import numpy as np
import pytest
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from calibrated_explanations.core.exceptions import NotFittedError
from calibrated_explanations.utils.helper import (
    calculate_metrics,
    check_is_fitted,
    safe_import,
    safe_isinstance,
)


class MockEstimator(BaseEstimator):
    def __init__(self):
        self.fitted_ = False

    def fit(self, x, y):
        self.fitted_ = True


def test_safe_import_module():
    """Test importing a module successfully."""
    np_module = safe_import("numpy")
    assert np_module is np


def test_safe_import_class():
    """Test importing a class from a module successfully."""
    ndarray_class = safe_import("numpy", "ndarray")
    assert ndarray_class is np.ndarray


def test_safe_import_multiple_classes():
    """Test importing multiple classes from a module successfully."""
    classes = safe_import("numpy", ["ndarray", "generic"])
    assert classes[0] is np.ndarray
    assert classes[1] is np.generic


def test_safe_import_module_not_installed():
    """Test handling ImportError when the module is not installed."""
    with pytest.raises(ImportError) as excinfo:
        safe_import("nonexistent_module")
    assert "The required module 'nonexistent_module' is not installed." in str(excinfo.value)


def test_safe_import_class_not_exist():
    """Test handling AttributeError when the class does not exist in the module."""
    with pytest.raises(ImportError) as excinfo:
        safe_import("numpy", "nonexistent_class")
    assert "The class or function 'nonexistent_class' does not exist in the module 'numpy'." in str(
        excinfo.value
    )


def test_safe_isinstance_single_class():
    """Test safe_isinstance with a single class path string."""
    model = RandomForestRegressor()
    assert safe_isinstance(model, "sklearn.ensemble.RandomForestRegressor") is True
    assert safe_isinstance(model, "sklearn.linear_model.LinearRegression") is False


def test_safe_isinstance_multiple_classes():
    """Test safe_isinstance with multiple class path strings."""
    model = RandomForestRegressor()
    class_paths = [
        "sklearn.linear_model.LinearRegression",
        "sklearn.ensemble.RandomForestRegressor",
    ]
    assert safe_isinstance(model, class_paths) is True


def test_safe_isinstance_class_not_imported():
    """Test safe_isinstance when the class module is not imported."""
    model = RandomForestRegressor()
    assert safe_isinstance(model, "nonexistent_module.NonexistentClass") is False


def test_safe_isinstance_invalid_class_path():
    """Test safe_isinstance with an invalid class path string."""
    model = RandomForestRegressor()
    with pytest.raises(ValueError) as excinfo:
        safe_isinstance(model, "InvalidClassPath")
        assert (
            "class_path_str must be a string or list of strings specifying a full module path to a class. Eg, 'sklearn.ensemble.RandomForestRegressor'"
            in str(excinfo.value)
        )


def test_safe_isinstance_class_not_exist():
    """Test safe_isinstance when the class does not exist in the module."""
    model = RandomForestRegressor()
    assert safe_isinstance(model, "sklearn.ensemble.NonexistentClass") is False


def test_safe_import_and_safe_isinstance_plugin_integration():
    """Integration-style coverage showing plugin resolution and fallback."""

    module_name = "tests.unit.core._fake_plugin"
    plugin_name = "Plugin"

    class DefaultPlugin:
        def run(self):
            return "fallback"

    plugin_module = types.ModuleType(module_name)

    class Plugin:
        def __init__(self):
            self.state = "active"

        def run(self):
            return "plugin"

    plugin_module.Plugin = Plugin
    sys.modules[module_name] = plugin_module

    try:
        loaded_module = safe_import(module_name)
        loaded_class = safe_import(module_name, plugin_name)
        assert loaded_module is plugin_module
        instance = loaded_class()
        assert instance.run() == "plugin"
        assert safe_isinstance(instance, f"{module_name}.{plugin_name}") is True

        def resolve_plugin(name):
            try:
                return safe_import(name, plugin_name)
            except ImportError:
                return DefaultPlugin

        assert resolve_plugin(module_name) is Plugin
        assert resolve_plugin("nonexistent.module") is DefaultPlugin
    finally:
        sys.modules.pop(module_name, None)


def test_check_is_fitted_with_fitted_estimator():
    """Test check_is_fitted with a fitted estimator."""
    estimator = LinearRegression()
    estimator.fit([[0, 0], [1, 1]], [0, 1])
    check_is_fitted(estimator, attributes="coef_")


def test_check_is_fitted_with_unfitted_estimator():
    """Test check_is_fitted with an unfitted estimator."""
    estimator = LinearRegression()
    with pytest.raises(NotFittedError) as excinfo:
        check_is_fitted(estimator, attributes="coef_")
    assert "This LinearRegression instance is not fitted yet." in str(excinfo.value)


def test_check_is_fitted_with_custom_message():
    """Test check_is_fitted with a custom error message."""
    estimator = LinearRegression()
    msg = "Custom error message for %(name)s."
    with pytest.raises(NotFittedError) as excinfo:
        check_is_fitted(estimator, attributes="coef_", msg=msg)
    assert "Custom error message for LinearRegression." in str(excinfo.value)


def test_check_is_fitted_with_class():
    """Test check_is_fitted with a class instead of an instance."""
    with pytest.raises(TypeError) as excinfo:
        check_is_fitted(LinearRegression, attributes="coef_")
    assert "is a class, not an instance." in str(excinfo.value)


def test_check_is_fitted_with_non_estimator():
    """Test check_is_fitted with a non-estimator instance."""
    with pytest.raises(TypeError) as excinfo:
        check_is_fitted("not_an_estimator", attributes="coef_")
    assert "is not an estimator instance." in str(excinfo.value)


def test_check_is_fitted_with_sklearn_is_fitted_method():
    """Test check_is_fitted with an estimator having __sklearn_is_fitted__ method."""

    class CustomEstimator(BaseEstimator):
        def __sklearn_is_fitted__(self):
            return True

        def fit(self, x, y):
            pass

    estimator = CustomEstimator()
    check_is_fitted(estimator)


def test_check_is_fitted_with_no_attributes():
    """Test check_is_fitted with no attributes specified."""
    estimator = MockEstimator()
    estimator.fit(None, None)
    check_is_fitted(estimator)


def test_calculate_metrics_no_arguments():
    """Test calculate_metrics with no arguments."""
    available_metrics = calculate_metrics()
    assert "ensured" in available_metrics


def test_calculate_metrics_ensured():
    """Test calculate_metrics with 'ensured' metric."""
    uncertainty = [0.1, 0.2, 0.3]
    prediction = [0.9, 0.8, 0.7]
    result = calculate_metrics(uncertainty, prediction, metric="ensured")
    expected = (1 - 0.5) * (1 - np.array(uncertainty)) + 0.5 * np.array(prediction)
    assert np.allclose(result, expected)


def test_calculate_metrics_with_weight():
    """Test calculate_metrics with a custom weight."""
    uncertainty = [0.1, 0.2, 0.3]
    prediction = [0.9, 0.8, 0.7]
    result = calculate_metrics(uncertainty, prediction, w=0.7, metric="ensured")
    expected = (1 - 0.7) * (1 - np.array(uncertainty)) + 0.7 * np.array(prediction)
    assert np.allclose(result, expected)


def test_calculate_metrics_with_negative_weight():
    """Test calculate_metrics with a negative weight."""
    uncertainty = [0.1, 0.2, 0.3]
    prediction = [0.9, 0.8, 0.7]
    result = calculate_metrics(uncertainty, prediction, w=-0.5, metric="ensured")
    expected = (1 - 0.5) * (1 - np.array(uncertainty)) + 0.5 * (-np.array(prediction))
    assert np.allclose(result, expected)


def test_calculate_metrics_with_normalization():
    """Test calculate_metrics with normalization."""
    uncertainty = [0.1, 0.2, 0.3]
    prediction = [0.9, 0.8, 0.7]
    result = calculate_metrics(uncertainty, prediction, metric="ensured", normalize=True)
    norm_uncertainty = (np.array(uncertainty) - np.min(uncertainty)) / (
        np.max(uncertainty) - np.min(uncertainty)
    )
    norm_prediction = (np.array(prediction) - np.min(prediction)) / (
        np.max(prediction) - np.min(prediction)
    )
    expected = (1 - 0.5) * (1 - norm_uncertainty) + 0.5 * norm_prediction
    assert np.allclose(result, expected)


def test_calculate_metrics_invalid_weight():
    """Test calculate_metrics with an invalid weight."""
    uncertainty = [0.1, 0.2, 0.3]
    prediction = [0.9, 0.8, 0.7]
    with pytest.raises(ValueError) as excinfo:
        calculate_metrics(uncertainty, prediction, w=1.5, metric="ensured")
    assert "The weight must be between -1 and 1." in str(excinfo.value)


def test_calculate_metrics_missing_arguments():
    """Test calculate_metrics with missing uncertainty or prediction."""
    with pytest.raises(ValueError) as excinfo:
        calculate_metrics(uncertainty=[0.1, 0.2, 0.3])
    assert (
        "Both uncertainty and prediction must be provided if any other argument is provided"
        in str(excinfo.value)
    )
