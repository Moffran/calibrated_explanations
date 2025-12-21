import numpy as np

from calibrated_explanations.utils.int_utils import as_int_array, coerce_to_int, collect_ints


def test_coerce_to_int_only_allows_safe_numeric_values() -> None:
    """Ensure strings/ints/floats that represent whole numbers convert cleanly."""
    assert coerce_to_int(5) == 5
    assert coerce_to_int(5.0) == 5
    assert coerce_to_int(-7.0) == -7
    assert coerce_to_int("42") == 42
    assert coerce_to_int("-003") == -3
    assert coerce_to_int("+5") is None
    assert coerce_to_int("not-a-number") is None
    assert coerce_to_int(3.14) is None
    assert coerce_to_int(None) is None


def test_collect_ints_processes_iterables_and_bytes() -> None:
    """Verify that mixed iterables yield the expected safe integers."""
    payload = [1, "2", b"3", 4.5, "invalid", None]
    assert collect_ints(payload) == [1, 2, 3]
    assert collect_ints(" 5 ") == [5]
    assert collect_ints(b"-6") == [-6]
    assert collect_ints(None) == []


def test_as_int_array_filters_non_integer_values() -> None:
    """Ensure the numpy array only contains successfully converted integers."""
    raw_values = ["7", 8.0, "skip", b"9"]
    assert np.array_equal(as_int_array(raw_values), np.array([7, 8, 9], dtype=int))
