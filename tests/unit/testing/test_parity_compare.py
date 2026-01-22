import numpy as np

from calibrated_explanations.testing import parity_compare


def test_parity_compare_allows_float_tolerance():
    expected = {"score": 0.500001}
    actual = {"score": 0.500002}
    diffs = parity_compare(expected, actual, rtol=1e-5, atol=1e-6)
    assert diffs == []


def test_parity_compare_reports_value_mismatch_with_path():
    expected = {"outer": [1, {"inner": 3.0}]}
    actual = {"outer": [1, {"inner": 3.5}]}
    diffs = parity_compare(expected, actual, rtol=1e-6, atol=1e-8)
    assert diffs
    assert diffs[0]["path"] == "$.outer[1].inner"
    assert diffs[0]["reason"] == "value_mismatch"


def test_parity_compare_reports_missing_and_extra_keys():
    expected = {"a": 1, "b": 2}
    actual = {"a": 1, "c": 3}
    diffs = parity_compare(expected, actual)
    reasons = {diff["reason"] for diff in diffs}
    assert "missing_key" in reasons
    assert "extra_key" in reasons


def test_parity_compare_handles_numpy_arrays():
    expected = {"values": np.array([1.0, 2.0, 3.0])}
    actual = {"values": [1.0, 2.0000001, 3.0]}
    diffs = parity_compare(expected, actual, rtol=1e-5, atol=1e-8)
    assert diffs == []
