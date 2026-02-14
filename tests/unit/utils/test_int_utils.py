from calibrated_explanations.utils.int_utils import coerce_to_int, collect_ints


def test_collect_ints_processes_iterables_and_bytes() -> None:
    """Verify that mixed iterables yield the expected safe integers."""
    payload = [1, "2", b"3", 4.5, "invalid", None]
    assert collect_ints(payload) == [1, 2, 3]
    assert collect_ints(" 5 ") == [5]
    assert collect_ints(b"-6") == [-6]
    assert collect_ints(None) == []


def test_coerce_to_int_handles_whitespace_and_invalid_strings() -> None:
    assert coerce_to_int("  12 ") == 12
    assert coerce_to_int(" ") is None
    assert coerce_to_int("not digits") is None


def test_collect_ints_handles_integrals_and_iterables() -> None:
    assert collect_ints(5) == [5]
    assert collect_ints((1, b"2", 3.0, None, "x")) == [1, 2, 3]
