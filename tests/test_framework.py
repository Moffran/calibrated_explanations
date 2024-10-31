# pylint: disable=invalid-name, protected-access, too-many-locals, line-too-long, redefined-outer-name, unused-import
"""
This module contains unit tests for the `CalibratedExplainer` class and utility functions from the `calibrated_explanations` package.
Fixtures:
    binary_dataset: Prepares a binary classification dataset for testing.
Tests:
    test_failure: Tests the failure case for initializing `CalibratedExplainer`.
    test_check_is_fitted_with_fitted_model: Tests `check_is_fitted` with a fitted model.
    test_check_is_fitted_with_non_fitted_model: Tests `check_is_fitted` with a non-fitted model.
    test_safe_import: Tests the `safe_import` utility function.
    test_make_directory_invalid_path: Tests `make_directory` with an invalid path.
    test_is_notebook_false: Tests `is_notebook` function when not running in a notebook.
    test_explanation_functions: Tests various explanation functions of `CalibratedExplainer`.
"""

from unittest.mock import patch
import pytest
from sklearn.ensemble import RandomForestClassifier

from crepes.extras import DifficultyEstimator

from calibrated_explanations import CalibratedExplainer
from calibrated_explanations.utils.helper import safe_import, check_is_fitted, make_directory, is_notebook
from tests.test_classification import binary_dataset, get_classification_model
from tests.test_regression import regression_dataset, get_regression_model

def test_failure():
    """
    Tests the failure case for initializing `CalibratedExplainer`.
    """
    with pytest.raises(RuntimeError):
        CalibratedExplainer(RandomForestClassifier(), [], [])

def test_check_is_fitted_with_fitted_model(binary_dataset):
    """
    Tests `check_is_fitted` with a fitted model.
    Args:
        binary_dataset (tuple): The binary classification dataset.
    """
    X_prop_train, y_prop_train, _, _, _, _, _, _, _, _ = binary_dataset
    model, _ = get_classification_model('RF', X_prop_train, y_prop_train)
    try:
        check_is_fitted(model)
    except TypeError:
        pytest.fail("check_is_fitted raised TypeError unexpectedly!")
    except RuntimeError:
        pytest.fail("check_is_fitted raised RuntimeError unexpectedly!")

def test_check_is_fitted_with_non_fitted_model():
    """
    Tests `check_is_fitted` with a non-fitted model.
    """
    with pytest.raises(RuntimeError):
        check_is_fitted(RandomForestClassifier())
    with pytest.raises(TypeError):
        check_is_fitted(RandomForestClassifier)

def test_safe_import():
    """
    Tests the `safe_import` utility function.
    """
    assert safe_import("sklearn") is not None
    with pytest.raises(ImportError):
        safe_import("p1337")

def test_make_directory_invalid_path():
    """
    Tests `make_directory` with an invalid path.
    """
    with pytest.raises(Exception):
        make_directory("/invalid/path/to/directory")

@patch("IPython.get_ipython")
def test_is_notebook_false(mock_get_ipython):
    """
    Tests `is_notebook` function when not running in a notebook.
    Args:
        mock_get_ipython (Mock): Mocked `get_ipython` function.
    """
    mock_get_ipython.return_value = None
    assert not is_notebook()

def test_explanation_functions(binary_dataset, regression_dataset):
    """
    Tests various explanation functions of `CalibratedExplainer`.
    Args:
        binary_dataset (tuple): The binary classification dataset.
    """
    X_prop_train, y_prop_train, X_cal, y_cal, X_test, _, _, _, _, _ = binary_dataset
    model, _ = get_classification_model('RF', X_prop_train, y_prop_train)
    ce = CalibratedExplainer(model, X_cal, y_cal, verbose=True)

    factual_explanations = ce.explain_factual(X_test)
    factual_explanations._get_rules()
    factual_explanations._is_alternative()
    factual_explanations._is_one_sided()
    factual_explanations._is_thresholded()
    factual_explanations.as_lime()
    factual_explanations.as_shap()

    alternative_explanations = ce.explore_alternatives(X_test)
    alternative_explanations._get_rules()
    alternative_explanations._is_alternative()
    alternative_explanations._is_one_sided()
    alternative_explanations._is_thresholded()

    ce._preload_lime()
    ce._preload_shap()

    print(ce)

    X_prop_train, y_prop_train, X_cal, y_cal, X_test, _, _, _, _ = regression_dataset
    model, _ = get_regression_model('RF', X_prop_train, y_prop_train)
    ce = CalibratedExplainer(model, X_cal, y_cal, mode='regression', verbose=True)

    factual_explanations = ce.explain_factual(X_test)
    factual_explanations._get_rules()
    factual_explanations._is_alternative()
    factual_explanations._is_one_sided()
    factual_explanations._is_thresholded()
    factual_explanations[0].is_multiclass()
    factual_explanations.as_lime()
    factual_explanations.as_shap()
    
    de = DifficultyEstimator().fit(X=X_prop_train, y=y_prop_train, scaler=True)
    ce = CalibratedExplainer(model, X_cal, y_cal, mode='regression', difficulty_estimator=de, verbose=True)
    ce.predict(X_test)

    alternative_explanations = ce.explore_alternatives(X_test)
    alternative_explanations._get_rules()
    alternative_explanations._is_alternative()
    alternative_explanations._is_one_sided()
    alternative_explanations._is_thresholded()

    ce._preload_lime()
    ce._preload_shap()

    print(ce)
