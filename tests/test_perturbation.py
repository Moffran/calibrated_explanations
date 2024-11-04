# pylint: disable=line-too-long, unused-import
"""
This module contains unit tests for the `uniform_perturbation` function from the
`calibrated_explanations.utils.perturbation` module.
The `uniform_perturbation` function is tested with various inputs to ensure its
correctness and robustness. The tests cover basic functionality, edge cases, and
performance with large inputs.
Tests included:
- `test_uniform_perturbation_basic`: Tests the function with a basic input array.
- `test_uniform_perturbation_severity_zero`: Tests the function with zero severity.
- `test_uniform_perturbation_severity_high`: Tests the function with high severity.
- `test_uniform_perturbation_negative_values`: Tests the function with negative values in the input array.
- `test_uniform_perturbation_large_column`: Tests the function with a large input array.
"""
import pytest
import numpy as np
from calibrated_explanations.utils.perturbation import uniform_perturbation, gaussian_perturbation, categorical_perturbation

def test_categorical_perturbation():
    """Test categorical_perturbation with basic input."""
    column = np.array(['a', 'b', 'c', 'd', 'e'])
    num_permutations = 5
    perturbed_column = categorical_perturbation(column, num_permutations)
    assert len(perturbed_column) == len(column)
    assert not np.array_equal(perturbed_column, column)

def test_gaussian_perturbation_basic():
    """Test gaussian_perturbation with basic input."""
    column = np.array([1, 2, 3, 4, 5])
    severity = 0.1
    perturbed_column = gaussian_perturbation(column, severity)
    assert len(perturbed_column) == len(column)
    assert not np.array_equal(perturbed_column, column)

# def test_gaussian_perturbation_severity_zero():
#     """Test gaussian_perturbation with zero severity."""
#     column = np.array([1, 2, 3, 4, 5])
#     severity = 0
#     perturbed_column = gaussian_perturbation(column, severity)
#     assert len(perturbed_column) == len(column)
#     assert np.array_equal(perturbed_column, column)

def test_gaussian_perturbation_high_severity():
    """Test gaussian_perturbation with high severity."""
    column = np.array([1, 2, 3, 4, 5])
    severity = 10
    perturbed_column = gaussian_perturbation(column, severity)
    assert len(perturbed_column) == len(column)
    assert not np.array_equal(perturbed_column, column)

def test_uniform_perturbation_basic():
    """Test uniform_perturbation with basic input."""
    column = np.array([1, 2, 3, 4, 5])
    severity = 0.1
    perturbed_column = uniform_perturbation(column, severity)
    assert len(perturbed_column) == len(column)
    assert not np.array_equal(perturbed_column, column)

def test_uniform_perturbation_severity_zero():
    """Test uniform_perturbation with zero severity."""
    column = np.array([1, 2, 3, 4, 5])
    severity = 0.0
    perturbed_column = uniform_perturbation(column, severity)
    assert np.array_equal(perturbed_column, column)

def test_uniform_perturbation_severity_high():
    """Test uniform_perturbation with high severity."""
    column = np.array([1, 2, 3, 4, 5])
    severity = 1.0
    perturbed_column = uniform_perturbation(column, severity)
    assert len(perturbed_column) == len(column)
    assert not np.array_equal(perturbed_column, column)

def test_uniform_perturbation_negative_values():
    """Test uniform_perturbation with negative values in the column."""
    column = np.array([-1, -2, -3, -4, -5])
    severity = 0.1
    perturbed_column = uniform_perturbation(column, severity)
    assert len(perturbed_column) == len(column)
    assert not np.array_equal(perturbed_column, column)

def test_uniform_perturbation_large_column():
    """Test uniform_perturbation with a large column."""
    column = np.random.rand(1000)
    severity = 0.1
    perturbed_column = uniform_perturbation(column, severity)
    assert len(perturbed_column) == len(column)
    assert not np.array_equal(perturbed_column, column)

def test_gaussian_perturbation_severity_high():
    """Test gaussian_perturbation with high severity."""
    column = np.array([1, 2, 3, 4, 5])
    severity = 1.0
    perturbed_column = gaussian_perturbation(column, severity)
    assert len(perturbed_column) == len(column)
    assert not np.array_equal(perturbed_column, column)

def test_gaussian_perturbation_negative_values():
    """Test gaussian_perturbation with negative values in the column."""
    column = np.array([-1, -2, -3, -4, -5])
    severity = 0.1
    perturbed_column = gaussian_perturbation(column, severity)
    assert len(perturbed_column) == len(column)
    assert not np.array_equal(perturbed_column, column)

def test_gaussian_perturbation_large_column():
    """Test gaussian_perturbation with a large column."""
    column = np.random.rand(1000)
    severity = 0.1
    perturbed_column = gaussian_perturbation(column, severity)
    assert len(perturbed_column) == len(column)
    assert not np.array_equal(perturbed_column, column)
