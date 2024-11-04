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
import pytest
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator
from calibrated_explanations.utils.helper import safe_import, safe_isinstance, check_is_fitted


class MockEstimator(BaseEstimator):
    def __init__(self):
        self.fitted_ = False

    def fit(self, X, y):
        self.fitted_ = True

def test_safe_import_module():
    """Test importing a module successfully."""
    np_module = safe_import('numpy')
    assert np_module is np

def test_safe_import_class():
    """Test importing a class from a module successfully."""
    ndarray_class = safe_import('numpy', 'ndarray')
    assert ndarray_class is np.ndarray

def test_safe_import_multiple_classes():
    """Test importing multiple classes from a module successfully."""
    classes = safe_import('numpy', ['ndarray', 'generic'])
    assert classes[0] is np.ndarray
    assert classes[1] is np.generic

def test_safe_import_module_not_installed():
    """Test handling ImportError when the module is not installed."""
    with pytest.raises(ImportError) as excinfo:
        safe_import('nonexistent_module')
    assert "The required module 'nonexistent_module' is not installed." in str(excinfo.value)

def test_safe_import_class_not_exist():
    """Test handling AttributeError when the class does not exist in the module."""
    with pytest.raises(ImportError) as excinfo:
        safe_import('numpy', 'nonexistent_class')
    assert "The class or function 'nonexistent_class' does not exist in the module 'numpy'." in str(excinfo.value)

def test_safe_isinstance_single_class():
    """Test safe_isinstance with a single class path string."""
    model = RandomForestRegressor()
    assert safe_isinstance(model, 'sklearn.ensemble.RandomForestRegressor') is True
    assert safe_isinstance(model, 'sklearn.linear_model.LinearRegression') is False

def test_safe_isinstance_multiple_classes():
    """Test safe_isinstance with multiple class path strings."""
    model = RandomForestRegressor()
    class_paths = ['sklearn.linear_model.LinearRegression', 'sklearn.ensemble.RandomForestRegressor']
    assert safe_isinstance(model, class_paths) is True

def test_safe_isinstance_class_not_imported():
    """Test safe_isinstance when the class module is not imported."""
    model = RandomForestRegressor()
    assert safe_isinstance(model, 'nonexistent_module.NonexistentClass') is False

def test_safe_isinstance_invalid_class_path():
    """Test safe_isinstance with an invalid class path string."""
    model = RandomForestRegressor()
    with pytest.raises(ValueError) as excinfo:
        safe_isinstance(model, 'InvalidClassPath')
        assert "class_path_str must be a string or list of strings specifying a full module path to a class. Eg, 'sklearn.ensemble.RandomForestRegressor'" in str(excinfo.value)

def test_safe_isinstance_class_not_exist():
    """Test safe_isinstance when the class does not exist in the module."""
    model = RandomForestRegressor()
    assert safe_isinstance(model, 'sklearn.ensemble.NonexistentClass') is False

def test_check_is_fitted_with_fitted_estimator():
    """Test check_is_fitted with a fitted estimator."""
    estimator = LinearRegression()
    estimator.fit([[0, 0], [1, 1]], [0, 1])
    check_is_fitted(estimator, attributes='coef_')

def test_check_is_fitted_with_unfitted_estimator():
    """Test check_is_fitted with an unfitted estimator."""
    estimator = LinearRegression()
    with pytest.raises(RuntimeError) as excinfo:
        check_is_fitted(estimator, attributes='coef_')
    assert "This LinearRegression instance is not fitted yet." in str(excinfo.value)

def test_check_is_fitted_with_custom_message():
    """Test check_is_fitted with a custom error message."""
    estimator = LinearRegression()
    msg = "Custom error message for %(name)s."
    with pytest.raises(RuntimeError) as excinfo:
        check_is_fitted(estimator, attributes='coef_', msg=msg)
    assert "Custom error message for LinearRegression." in str(excinfo.value)

def test_check_is_fitted_with_class():
    """Test check_is_fitted with a class instead of an instance."""
    with pytest.raises(TypeError) as excinfo:
        check_is_fitted(LinearRegression, attributes='coef_')
    assert "is a class, not an instance." in str(excinfo.value)

def test_check_is_fitted_with_non_estimator():
    """Test check_is_fitted with a non-estimator instance."""
    with pytest.raises(TypeError) as excinfo:
        check_is_fitted("not_an_estimator", attributes='coef_')
    assert "is not an estimator instance." in str(excinfo.value)

def test_check_is_fitted_with_sklearn_is_fitted_method():
    """Test check_is_fitted with an estimator having __sklearn_is_fitted__ method."""
    class CustomEstimator(BaseEstimator):
        def __sklearn_is_fitted__(self):
            return True
        def fit(self, X, y):
            pass

    estimator = CustomEstimator()
    check_is_fitted(estimator)

def test_check_is_fitted_with_no_attributes():
    """Test check_is_fitted with no attributes specified."""
    estimator = MockEstimator()
    estimator.fit(None, None)
    check_is_fitted(estimator)
