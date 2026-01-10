from calibrated_explanations.core.reject.policy import RejectPolicy, is_policy_enabled


def test_is_policy_enabled_basic():
    assert is_policy_enabled(RejectPolicy.NONE) is False
    assert is_policy_enabled(RejectPolicy.PREDICT_AND_FLAG) is True
    assert is_policy_enabled("explain_all") is True
    assert is_policy_enabled("none") is False
    assert is_policy_enabled(None) is False
    assert is_policy_enabled(12345) is False

