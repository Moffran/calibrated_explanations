# pylint: disable=missing-docstring, missing-module-docstring, invalid-name, protected-access, too-many-locals, line-too-long, duplicate-code
# flake8: noqa: E501
from __future__ import absolute_import
# import tempfile
# import os

import unittest
from unittest.mock import patch#, MagicMock
import pytest

from sklearn.ensemble import RandomForestClassifier

from calibrated_explanations import CalibratedExplainer
from calibrated_explanations.utils.helper import safe_import, check_is_fitted, make_directory, is_notebook

from tests.test_classification import load_binary_dataset, get_classification_model

class TestFramework(unittest.TestCase):
    def test_failure(self):
        with pytest.raises(RuntimeError):
            CalibratedExplainer(RandomForestClassifier(), [], [])

    def test_check_is_fitted_with_fitted_model(self):
        X_prop_train, y_prop_train, _, _, _, _, _, _, _, _ = load_binary_dataset()
        model, _ = get_classification_model('RF', X_prop_train, y_prop_train) # pylint: disable=redefined-outer-name
        # Assuming check_is_fitted does not return anything but raises an error if the model is not fitted
        try:
            check_is_fitted(model)
        except TypeError:
            pytest.fail("check_is_fitted raised TypeError unexpectedly!")
        except RuntimeError:
            pytest.fail("check_is_fitted raised RuntimeError unexpectedly!")

    def test_check_is_fitted_with_non_fitted_model(self):
        with pytest.raises(RuntimeError):
            check_is_fitted(RandomForestClassifier())
        with pytest.raises(TypeError):
            check_is_fitted(RandomForestClassifier)

    def test_check_safe_import(self):
        self.assertIsNotNone(safe_import("sklearn"))
        with self.assertRaises(ImportError):  # Replace Exception with the specific exception make_directory raises for invalid paths
            safe_import("p1337")

    # def test_make_directory_success(self):
    #     with tempfile.TemporaryDirectory() as tmp_dir:
    #         new_dir_path = os.path.join(tmp_dir, "new_directory")
    #         self.assertFalse(os.path.exists(new_dir_path))  # Ensure the directory does not exist before testing
    #         make_directory(new_dir_path)
    #         self.assertTrue(os.path.exists(new_dir_path))  # The directory should exist after calling make_directory

    # def test_make_directory_already_exists(self):
    #     with tempfile.TemporaryDirectory() as tmp_dir:
    #         # tmp_dir is already a directory, so calling make_directory should not raise an error
    #         make_directory(tmp_dir)  # No exception should be raised

    def test_make_directory_invalid_path(self):
        with self.assertRaises(Exception):  # Replace Exception with the specific exception make_directory raises for invalid paths
            make_directory("/invalid/path/to/directory")

    # @patch("IPython.get_ipython")
    # def test_is_notebook_true(self, mock_get_ipython):
    #     # Mock the environment to simulate running in a Jupyter notebook
    #     mock_get_ipython.return_value = MagicMock()
    #     self.assertTrue(is_notebook())

    @patch("IPython.get_ipython")
    def test_is_notebook_false(self, mock_get_ipython):
        # Mock the environment to simulate not running in a Jupyter notebook
        mock_get_ipython.return_value = None
        self.assertFalse(is_notebook())

    def test_explanation_functions(self):
        X_prop_train, y_prop_train, X_cal, y_cal, X_test, y_test, no_of_classes, no_of_features, categorical_features, columns = load_binary_dataset() # pylint: disable=unused-variable
        model, _ = get_classification_model('RF', X_prop_train, y_prop_train) # pylint: disable=redefined-outer-name
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

if __name__ == '__main__':
    # unittest.main()
    pytest.main()
