from calibrated_explanations.plotting import split_csv


def test_split_csv_coverage():
    assert split_csv(None) == ()
    assert split_csv("") == ()
    assert split_csv("a, b, c") == ("a", "b", "c")
    assert split_csv(["a ", " b"]) == ("a", "b")
    assert split_csv(123) == ()
    assert split_csv(" , , ") == ()


def test_split_csv_sequence_non_str():
    assert split_csv([1, 2, "a"]) == ("a",)
