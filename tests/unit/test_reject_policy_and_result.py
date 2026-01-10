from calibrated_explanations.core.reject.policy import RejectPolicy, is_policy_enabled
from calibrated_explanations.explanations.reject import RejectResult


def test_policy_enabled_and_conversion():
    assert is_policy_enabled(RejectPolicy.PREDICT_AND_FLAG) is True
    assert is_policy_enabled("predict_and_flag") is True
    assert is_policy_enabled(RejectPolicy.NONE) is False
    assert is_policy_enabled("none") is False
    assert is_policy_enabled(object()) is False


def test_reject_result_defaults():
    rr = RejectResult()
    assert rr.prediction is None
    assert rr.explanation is None
    assert rr.rejected is None
    assert rr.metadata is None
    assert rr.policy == RejectPolicy.NONE
