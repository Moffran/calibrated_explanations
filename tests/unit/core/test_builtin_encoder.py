"""Unit tests for the deterministic builtin categorical encoder."""

from __future__ import annotations

import numpy as np
import pytest

from calibrated_explanations.preprocessing.builtin_encoder import BuiltinEncoder
from calibrated_explanations.utils.exceptions import NotFittedError, ValidationError


class BadStringValue:
    """Value whose ``__str__`` fails to exercise repr fallback logic."""

    def __str__(self) -> str:
        raise ValueError("cannot stringify")

    def __repr__(self) -> str:
        return "BadStringValue()"


def test_transform_requires_fit() -> None:
    """Transform should fail when no mapping has been learned."""
    enc = BuiltinEncoder()
    with pytest.raises(NotFittedError, match="not fitted"):
        enc.transform(np.array(["a"], dtype=object))


def test_unseen_category_policy_ignore_sets_negative_one() -> None:
    """Unknown categories should map to -1 when ignore policy is selected."""
    enc = BuiltinEncoder(unseen_policy="ignore")
    enc.fit(np.array(["a", "b"], dtype=object))
    out = enc.transform(np.array(["b", "c"], dtype=object))
    np.testing.assert_allclose(out, np.array([[1.0], [-1.0]]))


def test_unseen_category_policy_error_raises_validation_error() -> None:
    """Unknown categories should raise a structured validation error by default."""
    enc = BuiltinEncoder(unseen_policy="error")
    enc.fit(np.array(["a", "b"], dtype=object))
    with pytest.raises(ValidationError, match="Unseen category"):
        enc.transform(np.array(["c"], dtype=object))


def test_mapping_snapshot_none_and_set_mapping_none_roundtrip() -> None:
    """Snapshot helpers should preserve the unfitted state."""
    enc = BuiltinEncoder()
    assert enc.get_mapping_snapshot() is None
    enc.set_mapping(None)
    assert enc.get_mapping_snapshot() is None


def test_fit_handles_none_and_repr_fallback_when_str_fails() -> None:
    """Fit should normalize None and fallback to repr for unstringifiable values."""
    enc = BuiltinEncoder()
    enc.fit(np.array([None, BadStringValue()], dtype=object))
    mapping = enc.get_mapping_snapshot()
    assert mapping is not None
    assert "__none__" in mapping["col_0"]
    assert "BadStringValue()" in mapping["col_0"]
