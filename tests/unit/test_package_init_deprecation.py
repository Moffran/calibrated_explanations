"""Tests for package-root exports after v0.11.0 deprecation removals."""

from __future__ import annotations

import calibrated_explanations as ce
import pytest


def test_sanctioned_symbols_remain_available():
    assert ce.CalibratedExplainer is not None
    assert ce.WrapCalibratedExplainer is not None
    assert ce.transform_to_numeric is not None


def test_removed_top_level_aliases_raise_attribute_error():
    removed = (
        "AlternativeExplanation",
        "FactualExplanation",
        "FastExplanation",
        "AlternativeExplanations",
        "CalibratedExplanations",
        "BinaryEntropyDiscretizer",
        "BinaryRegressorDiscretizer",
        "EntropyDiscretizer",
        "RegressorDiscretizer",
        "IntervalRegressor",
        "VennAbers",
    )
    for symbol in removed:
        with pytest.raises(AttributeError):
            getattr(ce, symbol)
