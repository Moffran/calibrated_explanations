import numpy as np
from calibrated_explanations.core.explain.feature_task import assign_weight_scalar


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
