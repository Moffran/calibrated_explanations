import pytest
import numpy as np
from types import SimpleNamespace

from calibrated_explanations.core.reject.orchestrator import (
    RejectOrchestrator,
    default_score_cal,
    default_score_test,
    interval_width_score,
    legacy_base_ncf,
    margin_score,
    normalize_stored_ncf,
)
from calibrated_explanations.core.reject.policy import RejectPolicy
from calibrated_explanations.explanations.reject import RejectResult
from calibrated_explanations.utils.exceptions import ValidationError


def make_stub():
    stub = SimpleNamespace()
    stub.x_cal = []
    stub.y_cal = []
    stub.mode = "classification"
    stub.interval_learner = SimpleNamespace()
    stub.predict = lambda x, **kw: [0 for _ in x]
    return stub


def test_apply_policy_none_and_invalid_policy_use_builtin():
    stub = make_stub()
    ro = RejectOrchestrator(stub)

    # explicit NONE -> no action via public apply_policy
    res = ro.apply_policy(RejectPolicy.NONE, x=[1, 2, 3])
    assert isinstance(res, RejectResult)
    assert res.prediction is None and res.explanation is None and res.rejected is None

    # unknown policy string is treated as NONE
    res2 = ro.apply_policy("not-a-policy", x=[1])
    assert isinstance(res2, RejectResult)
    assert res2.prediction is None and res2.rejected is None


def test_register_strategy_validation_errors():
    stub = make_stub()
    ro = RejectOrchestrator(stub)

    with pytest.raises(ValidationError):
        ro.register_strategy("", lambda: None)

    with pytest.raises(ValidationError):
        ro.register_strategy("valid.name", 123)


def test_reject_score_helpers_cover_edge_and_error_paths():
    proba_single = np.array([[0.7], [0.2]])
    assert np.allclose(interval_width_score(proba_single), np.array([0.0, 0.0]))
    assert np.allclose(margin_score(proba_single), np.array([0.0, 0.0]))

    proba = np.array([[0.2, 0.8], [0.6, 0.4]])
    classes = np.array([0, 1])
    labels = np.array([1, 0])

    with pytest.raises(ValidationError, match="Unsupported internal default score kind"):
        default_score_cal(proba, classes, labels, "unknown")
    with pytest.raises(ValidationError, match="Unsupported internal default score kind"):
        default_score_test(proba, "unknown")
    with pytest.raises(ValidationError, match="Unsupported ncf type"):
        legacy_base_ncf(proba, "unknown")

    assert normalize_stored_ncf(" custom ") == "custom"
    assert normalize_stored_ncf(None) is None


def test_predict_reject_breakdown_uses_predict_p_per_instance_fallback():
    class IntervalLearner:
        def predict_proba(self, x, bins=None):
            proba = []
            for row in x:
                marker = int(row[0])
                if marker == 1:
                    proba.append([0.4, 0.5])  # width 0.1
                elif marker == 2:
                    proba.append([0.3, 0.5])  # width 0.2
                else:
                    proba.append([0.2, 0.5])  # width 0.3
            return np.asarray(proba, dtype=float)

    class RejectLearner:
        def predict_set(self, *_args, **_kwargs):
            return np.array([[True, False]], dtype=bool)  # wrong bulk shape triggers fallback

        def predict_p(self, alphas, **_kwargs):
            # First item accepted (one label), second rejected (none), third ambiguous (two labels)
            mapping = {
                0.1: np.array([[0.9, 0.01]]),
                0.2: np.array([[0.05, 0.05]]),
                0.3: np.array([[0.9, 0.9]]),
            }
            key = round(float(alphas[0, 0]), 1)
            return mapping[key]

    stub = make_stub()
    stub.interval_learner = IntervalLearner()
    stub.reject_learner = RejectLearner()
    stub.is_multiclass = lambda: False
    stub.reject_ncf = "ensured"
    stub.reject_ncf_w = 0.0
    stub.seed = 1
    stub.bins = np.array([10, 11, 12])
    ro = RejectOrchestrator(stub)

    breakdown = ro.predict_reject_breakdown([[1], [2], [3]], bins=stub.bins, confidence=0.95)
    assert breakdown["prediction_set"].shape == (3, 2)
    assert breakdown["prediction_set_size"].tolist() == [1, 0, 2]
    assert breakdown["error_rate_defined"] is True
