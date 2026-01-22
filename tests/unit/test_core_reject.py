from calibrated_explanations.core.reject.policy import RejectPolicy, is_policy_enabled


def test_is_policy_enabled_with_enum():
    assert is_policy_enabled(RejectPolicy.PREDICT_AND_FLAG)
    assert not is_policy_enabled(RejectPolicy.NONE)


def test_is_policy_enabled_with_strings():
    assert is_policy_enabled("predict_and_flag")
    assert not is_policy_enabled("none")


def test_is_policy_enabled_with_invalid_values():
    # arbitrary types and unknown strings should be treated as disabled
    assert not is_policy_enabled(None)
    assert not is_policy_enabled(123)
    assert not is_policy_enabled("not-a-policy")
