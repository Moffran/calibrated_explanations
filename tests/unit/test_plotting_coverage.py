from calibrated_explanations.plotting import _split_csv


def test_split_csv_coverage():
    assert _split_csv(None) == ()
    assert _split_csv("") == ()
    assert _split_csv("a, b, c") == ("a", "b", "c")
    assert _split_csv(["a ", " b"]) == ("a", "b")
    assert _split_csv(123) == ()
    assert _split_csv(" , , ") == ()


def test_split_csv_sequence_non_str():
    assert _split_csv([1, 2, "a"]) == ("a",)
