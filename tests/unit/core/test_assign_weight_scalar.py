import numpy as np
import pytest
from calibrated_explanations.core.explain.feature_task import assign_weight_scalar

def test_assign_weight_scalar_scalars():
    """Test assign_weight_scalar with simple scalar values."""
    # Simple subtraction: 0.8 - 0.2 = 0.6
    assert pytest.approx(assign_weight_scalar(0.2, 0.8)) == 0.6
    # Negative numbers: -0.5 - 0.5 = -1.0
    assert pytest.approx(assign_weight_scalar(0.5, -0.5)) == -1.0
    # Integers
    assert assign_weight_scalar(2, 5) == 3.0

def test_assign_weight_scalar_arrays():
    """Test assign_weight_scalar with numpy arrays."""
    # 1D arrays: returns first element of difference
    a = np.array([0.1, 0.2])
    b = np.array([0.5, 0.6])
    # (0.5 - 0.1) = 0.4
    assert pytest.approx(assign_weight_scalar(a, b)) == 0.4

    # 2D arrays
    a_2d = np.array([[0.1, 0.2], [0.3, 0.4]])
    b_2d = np.array([[0.5, 0.6], [0.7, 0.8]])
    # (0.5 - 0.1) = 0.4
    assert pytest.approx(assign_weight_scalar(a_2d, b_2d)) == 0.4

def test_assign_weight_scalar_empty():
    """Test assign_weight_scalar with empty inputs."""
    assert assign_weight_scalar([], []) == 0.0
    assert assign_weight_scalar(np.array([]), np.array([])) == 0.0

def test_assign_weight_scalar_object_arrays():
    """Test assign_weight_scalar with object arrays (fallback path)."""
    # Object arrays with numeric strings
    pred = np.asarray(["1.5"], dtype=object)
    inst = np.asarray(["0.5"], dtype=object)
    assert assign_weight_scalar(inst, pred) == 1.0

def test_assign_weight_scalar_nan():
    """Test assign_weight_scalar with NaN values."""
    assert np.isnan(assign_weight_scalar(np.nan, 1.0))
    assert np.isnan(assign_weight_scalar(1.0, np.nan))
    
    # Arrays with NaN
    assert np.isnan(assign_weight_scalar(np.array([np.nan]), np.array([1.0])))

def test_assign_weight_scalar_mismatched_shapes():
    """Test assign_weight_scalar with mismatched shapes (broadcasting)."""
    # Broadcasting: scalar vs array
    # prediction (array) - instance (scalar)
    # [0.5, 0.6] - 0.1 = [0.4, 0.5] -> first element 0.4
    assert pytest.approx(assign_weight_scalar(0.1, np.array([0.5, 0.6]))) == 0.4
    
    # instance (array) - prediction (scalar)
    # 0.5 - [0.1, 0.2] = [0.4, 0.3] -> first element 0.4
    assert pytest.approx(assign_weight_scalar(np.array([0.1, 0.2]), 0.5)) == 0.4
