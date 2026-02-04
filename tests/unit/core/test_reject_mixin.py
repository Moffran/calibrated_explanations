# tests/unit/core/test_reject_mixin.py
import pytest
import numpy as np
from unittest.mock import MagicMock
from copy import copy
from dataclasses import dataclass

from calibrated_explanations.explanations.explanations import CalibratedExplanations
from calibrated_explanations.explanations.reject import (
    RejectCalibratedExplanations, 
    RejectPolicy
)

@dataclass
class FakeExplanation:
    index: int

class FakeCalibratedExplanations:
    """Minimal fake for CalibratedExplanations to avoid complex mocking issues."""
    def __init__(self):
        self.explanations = []
        self.x_test = None # lowercase as per traceback
        self.y_threshold = 0.5
        self.bins = None  # Required by CalibratedExplanations.__getitem__

    def __getitem__(self, key):
        pass

class TestRejectMixin:
    """Tests for the RejectMixin mechanics via RejectCalibratedExplanations."""

    def test_metadata_property_parity(self):
        """Test that the metadata property correctly integrates masks."""
        # Setup base
        base = FakeCalibratedExplanations()
        base.explanations = [FakeExplanation(0), FakeExplanation(1)]
        
        metadata_in = {
            "original_key": "value",
            "ambiguity_mask": np.array([1, 0]),
            "prediction_set_size": np.array([2, 1]),
            # No novelty_mask provided
        }
        
        # Create SUT
        sut = RejectCalibratedExplanations.from_collection(
            base,
            metadata_in,
            RejectPolicy.FLAG
        )
        
        md = sut.metadata
        
        assert "original_key" in md
        assert "ambiguity_mask" in md
        assert "novelty_mask" not in md
        assert "prediction_set_size" in md
        np.testing.assert_array_equal(md["ambiguity_mask"], np.array([1, 0]))

    def test_slice_handles_all_masks(self):
        """Test slicing behaves correctly via __getitem__."""
        # Setup base
        base = FakeCalibratedExplanations()
        base.explanations = [FakeExplanation(i) for i in range(5)]
        base.x_test = np.zeros((5, 2)) # Dummy data
        
        metadata_in = {
            "ambiguity_mask": np.array([0, 1, 0, 0, 0]),
            "novelty_mask": np.array([0, 0, 0, 1, 0]),
            "prediction_set_size": np.array([1, 2, 1, 2, 1]),
            "rejected": np.array([0, 1, 0, 1, 0]), # logical OR of masks
            "epsilon": 0.1,
            "extra": "persist"
        }

        sut = RejectCalibratedExplanations.from_collection(
            base,
            metadata_in,
            RejectPolicy.FLAG,
            rejected=metadata_in["rejected"]
        )
        
        # Act: Slice via public API
        target = sut[1:3] # indices 1, 2

        # Assert
        np.testing.assert_array_equal(target.rejected, np.array([1, 0]))
        np.testing.assert_array_equal(target.ambiguity_mask, np.array([1, 0]))
        np.testing.assert_array_equal(target.novelty_mask, np.array([0, 0]))
        assert target.policy == RejectPolicy.FLAG
        # Ensure metadata dict preserves static values
        assert target.metadata["extra"] == "persist"

    def test_slice_handles_none_fields(self):
        """Test slicing when optional fields are None."""
        base = FakeCalibratedExplanations()
        base.explanations = [FakeExplanation(0), FakeExplanation(1), FakeExplanation(2)]
        base.x_test = np.zeros((3, 2))
        
        metadata_in = {
            "rejected": np.array([0, 1, 0]),
            # ambiguity_mask missing/None
        }
        
        sut = RejectCalibratedExplanations.from_collection(
            base,
            metadata_in,
            RejectPolicy.FLAG,
            rejected=metadata_in["rejected"]
        )
        
        target = sut[0:2] # Return collection of 2
        
        assert target.ambiguity_mask is None
        np.testing.assert_array_equal(target.rejected, np.array([0, 1]))
