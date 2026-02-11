from calibrated_explanations.core.config_helpers import split_csv, coerce_string_tuple


def test_config_helpers_split_and_coerce():
    assert split_csv(None) == ()
    assert split_csv("a,b , c") == ("a", "b", "c")
    assert coerce_string_tuple(None) == ()
    assert coerce_string_tuple("x") == ("x",)
    assert coerce_string_tuple(["a", "", "b"]) == ("a", "b")
