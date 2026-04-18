import logging
import numpy as np
import pytest

from tests.helpers.model_utils import get_classification_model, get_regression_model
from tests.helpers.dataset_utils import make_binary_dataset, make_regression_dataset
from tests.helpers.explainer_utils import initiate_explainer
from calibrated_explanations.explanations.reject import RejectResult, RejectPolicy
from calibrated_explanations.utils.exceptions import ValidationError


def test_predict_skip_reject_internal_returns_prediction():
    dataset = make_binary_dataset()
    (
        x_prop_train,
        y_prop_train,
        x_cal,
        y_cal,
        x_test,
        _y_test,
        _,
        _,
        categorical_features,
        feature_names,
    ) = dataset

    model, _ = get_classification_model("RF", x_prop_train, y_prop_train)
    cal_exp = initiate_explainer(
        model, x_cal, y_cal, feature_names, categorical_features, mode="classification"
    )

    # When _ce_skip_reject is True the legacy calibrated prediction path is used
    res = cal_exp.predict(x_test, _ce_skip_reject=True)
    # Expect numpy array (formatted classification labels)
    assert isinstance(res, (list, np.ndarray))


def test_predict_with_implicit_default_reject_policy_logs(monkeypatch):
    dataset = make_binary_dataset()
    (
        x_prop_train,
        y_prop_train,
        x_cal,
        y_cal,
        x_test,
        _y_test,
        _,
        _,
        categorical_features,
        feature_names,
    ) = dataset

    model, _ = get_classification_model("RF", x_prop_train, y_prop_train)
    cal_exp = initiate_explainer(
        model, x_cal, y_cal, feature_names, categorical_features, mode="classification"
    )

    # Set a non-NONE default to trigger implicit_default_used when no reject_policy provided
    cal_exp.default_reject_policy = RejectPolicy.FLAG

    records = []

    def fake_info(msg, *a, **k):
        records.append(msg)

    monkeypatch.setattr(
        logging.getLogger("calibrated_explanations.core.calibrated_explainer"), "info", fake_info
    )

    rr = RejectResult(prediction=None, policy=RejectPolicy.FLAG)
    current = cal_exp.reject_orchestrator
    monkeypatch.setattr(current, "apply_policy", lambda *a, **k: rr, raising=False)

    res = cal_exp.predict(x_test)
    assert isinstance(res, RejectResult)
    assert any("Default reject policy" in str(r) or "Default reject policy" in r for r in records)


def test_predict_rr_prediction_none_preserved(monkeypatch):
    dataset = make_binary_dataset()
    (
        x_prop_train,
        y_prop_train,
        x_cal,
        y_cal,
        x_test,
        _y_test,
        _,
        _,
        categorical_features,
        feature_names,
    ) = dataset

    model, _ = get_classification_model("RF", x_prop_train, y_prop_train)
    cal_exp = initiate_explainer(
        model, x_cal, y_cal, feature_names, categorical_features, mode="classification"
    )

    rr = RejectResult(prediction=None, policy=RejectPolicy.FLAG)
    current = cal_exp.reject_orchestrator
    monkeypatch.setattr(current, "apply_policy", lambda *a, **k: rr, raising=False)

    res = cal_exp.predict(x_test, reject_policy="flag")
    assert isinstance(res, RejectResult)
    assert res.prediction is None


def test_predict_proba_uncalibrated_regression_raises_when_threshold():
    dataset = make_regression_dataset()
    (
        x_prop_train,
        y_prop_train,
        x_cal,
        y_cal,
        x_test,
        _y_test,
        _,
        no_of_features,
        categorical_features,
        feature_names,
    ) = dataset

    model, _ = get_regression_model("RF", x_prop_train, y_prop_train)
    cal_exp = initiate_explainer(
        model, x_cal, y_cal, feature_names, categorical_features, mode="regression"
    )

    # uncalibrated regression with threshold should raise ValidationError inside helper
    with pytest.raises(Exception):
        cal_exp.predict(x_test, calibrated=False, threshold=0.5)


def test_invalid_default_policy_falls_back_to_legacy_payloads():
    dataset = make_binary_dataset()
    (
        x_prop_train,
        y_prop_train,
        x_cal,
        y_cal,
        x_test,
        _y_test,
        _,
        _,
        categorical_features,
        feature_names,
    ) = dataset

    model, _ = get_classification_model("RF", x_prop_train, y_prop_train)
    cal_exp = initiate_explainer(
        model, x_cal, y_cal, feature_names, categorical_features, mode="classification"
    )
    cal_exp.default_reject_policy = "not-a-policy"

    with pytest.warns(UserWarning, match="Invalid default_reject_policy"):
        pred = cal_exp.predict(x_test[:4])
    assert not isinstance(pred, RejectResult)

    with pytest.warns(UserWarning, match="Invalid default_reject_policy"):
        proba = cal_exp.predict_proba(x_test[:4], uq_interval=False)
    assert not isinstance(proba, RejectResult)

    with pytest.warns(UserWarning, match="Invalid default_reject_policy"):
        expl = cal_exp.explain_factual(x_test[:2])
    assert not isinstance(expl, RejectResult)
    assert hasattr(expl, "explanations")


def test_invalid_explicit_policy_fails_fast_across_predict_and_explain():
    dataset = make_binary_dataset()
    (
        x_prop_train,
        y_prop_train,
        x_cal,
        y_cal,
        x_test,
        _y_test,
        _,
        _,
        categorical_features,
        feature_names,
    ) = dataset

    model, _ = get_classification_model("RF", x_prop_train, y_prop_train)
    cal_exp = initiate_explainer(
        model, x_cal, y_cal, feature_names, categorical_features, mode="classification"
    )

    with pytest.raises(ValidationError, match="Unknown reject policy string"):
        cal_exp.predict(x_test[:4], reject_policy="not-a-policy")

    with pytest.raises(ValidationError, match="Unknown reject policy string"):
        cal_exp.predict_proba(x_test[:4], reject_policy="not-a-policy")

    with pytest.raises(ValidationError, match="Unknown reject policy string"):
        cal_exp.explain_factual(x_test[:2], reject_policy="not-a-policy")


def test_reject_confidence_forwarded_across_explain_and_guarded_paths(monkeypatch):
    dataset = make_binary_dataset()
    (
        x_prop_train,
        y_prop_train,
        x_cal,
        y_cal,
        x_test,
        _y_test,
        _,
        _,
        categorical_features,
        feature_names,
    ) = dataset

    model, _ = get_classification_model("RF", x_prop_train, y_prop_train)
    cal_exp = initiate_explainer(
        model, x_cal, y_cal, feature_names, categorical_features, mode="classification"
    )

    seen_confidences = []

    def fake_apply_policy(policy, x, explain_fn=None, bins=None, confidence=0.95, **kwargs):
        seen_confidences.append(float(confidence))
        return RejectResult(
            prediction=None,
            explanation=None,
            rejected=np.zeros(len(x), dtype=bool),
            policy=RejectPolicy.FLAG,
            metadata={},
        )

    monkeypatch.setattr(cal_exp.reject_orchestrator, "apply_policy", fake_apply_policy)

    cal_exp.explain_factual(x_test[:2], reject_policy=RejectPolicy.FLAG, confidence=0.81)
    cal_exp.explore_alternatives(x_test[:2], reject_policy=RejectPolicy.FLAG, confidence=0.82)
    cal_exp.explain_guarded_factual(x_test[:2], reject_policy=RejectPolicy.FLAG, confidence=0.83)
    cal_exp.explore_guarded_alternatives(
        x_test[:2], reject_policy=RejectPolicy.FLAG, confidence=0.84
    )

    assert seen_confidences == [0.81, 0.82, 0.83, 0.84]


@pytest.mark.parametrize("bad_confidence", [0.0, 1.0, -0.1, 1.1])
def test_invalid_confidence_rejected_across_predict_and_explain(bad_confidence):
    dataset = make_binary_dataset()
    (
        x_prop_train,
        y_prop_train,
        x_cal,
        y_cal,
        x_test,
        _y_test,
        _,
        _,
        categorical_features,
        feature_names,
    ) = dataset

    model, _ = get_classification_model("RF", x_prop_train, y_prop_train)
    cal_exp = initiate_explainer(
        model, x_cal, y_cal, feature_names, categorical_features, mode="classification"
    )

    with pytest.raises(ValidationError, match="confidence must be a float"):
        cal_exp.predict(x_test[:3], reject_policy=RejectPolicy.FLAG, confidence=bad_confidence)
    with pytest.raises(ValidationError, match="confidence must be a float"):
        cal_exp.predict_proba(
            x_test[:3], reject_policy=RejectPolicy.FLAG, confidence=bad_confidence
        )
    with pytest.raises(ValidationError, match="confidence must be a float"):
        cal_exp.explain_factual(
            x_test[:2], reject_policy=RejectPolicy.FLAG, confidence=bad_confidence
        )


def test_reject_context_uses_source_indices_for_only_accepted(monkeypatch):
    dataset = make_binary_dataset()
    (
        x_prop_train,
        y_prop_train,
        x_cal,
        y_cal,
        x_test,
        _y_test,
        _,
        _,
        categorical_features,
        feature_names,
    ) = dataset

    model, _ = get_classification_model("RF", x_prop_train, y_prop_train)
    cal_exp = initiate_explainer(
        model, x_cal, y_cal, feature_names, categorical_features, mode="classification"
    )

    class DummyExplanation:
        def __init__(self, name):
            self.name = name
            self.reject_context = None

    payload = [DummyExplanation("accepted-1"), DummyExplanation("accepted-3")]

    def fake_apply_policy(policy, x, explain_fn=None, bins=None, confidence=0.95, **kwargs):
        return RejectResult(
            prediction=None,
            explanation=payload,
            rejected=np.array([True, False, True, False]),
            policy=RejectPolicy.ONLY_ACCEPTED,
            metadata={
                "source_indices": [1, 3],
                "original_count": 4,
                "prediction_set_size": np.array([2, 1, 2, 1]),
                "ambiguity_mask": np.array([True, False, True, False]),
                "novelty_mask": np.array([False, False, False, False]),
                "epsilon": 0.05,
            },
        )

    monkeypatch.setattr(cal_exp.reject_orchestrator, "apply_policy", fake_apply_policy)

    result = cal_exp.explain_factual(x_test[:4], reject_policy=RejectPolicy.ONLY_ACCEPTED)
    assert isinstance(result, RejectResult)
    assert result.explanation[0].reject_context.rejected is False
    assert result.explanation[1].reject_context.rejected is False


def test_reject_context_fallback_mapping_warns_when_source_indices_missing(monkeypatch):
    dataset = make_binary_dataset()
    (
        x_prop_train,
        y_prop_train,
        x_cal,
        y_cal,
        x_test,
        _y_test,
        _,
        _,
        categorical_features,
        feature_names,
    ) = dataset

    model, _ = get_classification_model("RF", x_prop_train, y_prop_train)
    cal_exp = initiate_explainer(
        model, x_cal, y_cal, feature_names, categorical_features, mode="classification"
    )

    class DummyExplanation:
        def __init__(self):
            self.reject_context = None

    payload = [DummyExplanation(), DummyExplanation()]

    def fake_apply_policy(policy, x, explain_fn=None, bins=None, confidence=0.95, **kwargs):
        return RejectResult(
            prediction=None,
            explanation=payload,
            rejected=np.array([True, False, True, False]),
            policy=RejectPolicy.ONLY_REJECTED,
            metadata={},
        )

    monkeypatch.setattr(cal_exp.reject_orchestrator, "apply_policy", fake_apply_policy)

    with pytest.warns(UserWarning, match="missing source_indices"):
        result = cal_exp.explain_factual(x_test[:4], reject_policy=RejectPolicy.ONLY_REJECTED)
    assert isinstance(result, RejectResult)
    assert result.explanation[0].reject_context.rejected is True
    assert result.explanation[1].reject_context.rejected is True


def test_regression_predict_proba_forwards_threshold_to_reject_policy(monkeypatch):
    dataset = make_regression_dataset()
    (
        x_prop_train,
        y_prop_train,
        x_cal,
        y_cal,
        x_test,
        _y_test,
        _,
        _,
        categorical_features,
        feature_names,
    ) = dataset

    model, _ = get_regression_model("RF", x_prop_train, y_prop_train)
    cal_exp = initiate_explainer(
        model, x_cal, y_cal, feature_names, categorical_features, mode="regression"
    )

    seen_thresholds = []

    def fake_apply_policy(
        policy, x, explain_fn=None, bins=None, confidence=0.95, threshold=None, **kwargs
    ):
        seen_thresholds.append(threshold)
        return RejectResult(
            prediction=None,
            explanation=None,
            rejected=np.zeros(len(x), dtype=bool),
            policy=RejectPolicy.FLAG,
            metadata={},
        )

    monkeypatch.setattr(cal_exp.reject_orchestrator, "apply_policy", fake_apply_policy)
    cal_exp.predict_proba(x_test[:3], reject_policy=RejectPolicy.FLAG, threshold=0.42)
    assert seen_thresholds == [0.42]


def test_regression_reject_without_threshold_raises_across_paths():
    dataset = make_regression_dataset()
    (
        x_prop_train,
        y_prop_train,
        x_cal,
        y_cal,
        x_test,
        _y_test,
        _,
        _,
        categorical_features,
        feature_names,
    ) = dataset

    model, _ = get_regression_model("RF", x_prop_train, y_prop_train)
    cal_exp = initiate_explainer(
        model, x_cal, y_cal, feature_names, categorical_features, mode="regression"
    )

    with pytest.raises(
        ValidationError, match="reject learner unavailable for regression without threshold"
    ):
        cal_exp.predict(x_test[:3], reject_policy=RejectPolicy.FLAG)
    with pytest.raises(
        ValidationError, match="reject learner unavailable for regression without threshold"
    ):
        cal_exp.predict_proba(x_test[:3], reject_policy=RejectPolicy.FLAG)
    with pytest.raises(
        ValidationError, match="reject learner unavailable for regression without threshold"
    ):
        cal_exp.explain_factual(x_test[:2], reject_policy=RejectPolicy.FLAG)


def test_reject_metadata_contract_present_across_predict_proba_and_explain():
    dataset = make_binary_dataset()
    (
        x_prop_train,
        y_prop_train,
        x_cal,
        y_cal,
        x_test,
        _y_test,
        _,
        _,
        categorical_features,
        feature_names,
    ) = dataset

    model, _ = get_classification_model("RF", x_prop_train, y_prop_train)
    cal_exp = initiate_explainer(
        model, x_cal, y_cal, feature_names, categorical_features, mode="classification"
    )

    required = {
        "policy",
        "reject_rate",
        "accepted_count",
        "rejected_count",
        "effective_confidence",
        "effective_threshold",
        "source_indices",
        "original_count",
        "init_ok",
        "fallback_used",
        "init_error",
        "degraded_mode",
    }

    pred = cal_exp.predict(x_test[:6], reject_policy=RejectPolicy.FLAG)
    proba = cal_exp.predict_proba(x_test[:6], reject_policy=RejectPolicy.FLAG, uq_interval=False)
    expl = cal_exp.explain_factual(x_test[:6], reject_policy=RejectPolicy.FLAG)

    assert isinstance(pred, RejectResult)
    assert isinstance(proba, RejectResult)
    assert required.issubset((pred.metadata or {}).keys())
    assert required.issubset((proba.metadata or {}).keys())
    assert required.issubset(expl.metadata.keys())
