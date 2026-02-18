"""Tests for explain helpers module.

This module tests the helper functions that were previously called as private
methods on CalibratedExplainer. Tests are refactored to test behavior through
the public explain module API rather than implementation details.
"""

import numpy as np
import pytest
from typing import Any

from calibrated_explanations.core.explain import helpers as helpers


class TestSliceThreshold:
    """Test behavior of slice_threshold helper function."""


class TestSliceBins:
    """Test behavior of slice_bins helper function."""


class TestComputeWeightDelta:
    """Test behavior of compute_weight_delta helper function."""

    def test_compute_weight_delta_fallback_path_for_object_values(self):
        """Object subtraction should fall back to element-wise scalar extraction."""

        class UnevenDelta:
            def __init__(self, value: float, width: int) -> None:
                self.value = value
                self.width = width

            def __rsub__(self, other: float):
                return [float(other) - self.value] * self.width

        baseline = [4.0, 6.0]
        perturbed = [UnevenDelta(1.0, 2), UnevenDelta(2.0, 1)]

        result = helpers.compute_weight_delta(baseline, perturbed)

        np.testing.assert_array_almost_equal(result, np.array([3.0, 4.0]))


@pytest.mark.parametrize(
    "func_name,args,kwargs",
    [
        ("compute_feature_effects", ("payload",), {}),
        ("initialize_explanation", ("init",), {"mode": "factual"}),
        ("explain_predict_step", ("state",), {"step": 2}),
        ("feature_effect_for_index", ("state",), {"index": 1}),
        ("validate_and_prepare_input", ("x",), {"y": "z"}),
    ],
)
def test_should_delegate_to_internal_helpers_when_called(func_name, args, kwargs, monkeypatch):
    """Each public helper should forward to the internal implementation."""
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def stub(*stub_args: Any, **stub_kwargs: Any) -> str:
        calls.append((stub_args, stub_kwargs))
        return "sentinel"

    monkeypatch.setattr(helpers.impl, func_name, stub)

    result = getattr(helpers, func_name)(*args, **kwargs)

    assert result == "sentinel"
    assert calls == [(args, kwargs)]
