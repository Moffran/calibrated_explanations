from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from calibrated_explanations.core.reject import orchestrator as orch
from calibrated_explanations.core.reject.orchestrator import default_score_cal, default_score_test
from calibrated_explanations.utils.exceptions import ValidationError


def test_strategy_is_registered():
    explainer = make_explainer([[0.2, 0.8], [0.6, 0.4]], [1, 0])
    orchestrator = orch.RejectOrchestrator(explainer)
    # Should not raise
    fn = orchestrator.resolve_strategy("experimental.difficulty_normalized")
    assert callable(fn)


def test_unknown_strategy_raises():
    explainer = make_explainer([[0.2, 0.8], [0.6, 0.4]], [1, 0])
    orchestrator = orch.RejectOrchestrator(explainer)
    with pytest.raises(KeyError, match="not registered"):
        orchestrator.resolve_strategy("not.a.strategy")


def test_experimental_matches_builtin_without_difficulty(monkeypatch):
    proba = np.array([[0.2, 0.8], [0.6, 0.4]], dtype=float)
    labels = np.array([1, 0])
    explainer = make_explainer(proba, labels)
    orchestrator = orch.RejectOrchestrator(explainer)
    # Use both strategies
    res_builtin = orchestrator.apply_policy(
        policy="flag", x=explainer.x_cal, strategy="builtin.default"
    )
    res_experimental = orchestrator.apply_policy(
        policy="flag", x=explainer.x_cal, strategy="experimental.difficulty_normalized"
    )
    np.testing.assert_array_equal(res_builtin.rejected, res_experimental.rejected)
    np.testing.assert_array_equal(
        res_builtin.metadata["prediction_set_size"],
        res_experimental.metadata["prediction_set_size"],
    )
    assert res_experimental.metadata["difficulty_normalized"] is True
    assert res_experimental.metadata["reject_strategy"] == "experimental.difficulty_normalized"


def test_experimental_changes_with_difficulty(monkeypatch):
    proba = np.array([[0.05, 0.95], [0.52, 0.48], [0.95, 0.05]], dtype=float)
    labels = np.array([1, 0, 0])
    explainer = make_explainer(proba, labels, difficulty_values=[1.0, 8.0, 1.0])
    monkeypatch.setattr(orch, "ConformalClassifier", InverseAlphaConformalClassifier)
    orchestrator = orch.RejectOrchestrator(explainer)
    res_builtin = orchestrator.apply_policy(
        policy="flag", x=explainer.x_cal, strategy="builtin.default", confidence=0.5
    )
    res_experimental = orchestrator.apply_policy(
        policy="flag",
        x=explainer.x_cal,
        strategy="experimental.difficulty_normalized",
        confidence=0.5,
    )
    assert not np.array_equal(
        res_builtin.rejected, res_experimental.rejected
    ) or not np.array_equal(
        res_builtin.metadata["prediction_set_size"],
        res_experimental.metadata["prediction_set_size"],
    )
    # Difficulty stats present
    assert "difficulty_min" in res_experimental.metadata
    assert "difficulty_max" in res_experimental.metadata
    assert "difficulty_mean" in res_experimental.metadata


def test_experimental_flag_and_subset_policies(monkeypatch):
    proba = np.array([[0.2, 0.8], [0.6, 0.4]], dtype=float)
    labels = np.array([1, 0])
    explainer = make_explainer(proba, labels, difficulty_values=[1.0, 2.0])
    orchestrator = orch.RejectOrchestrator(explainer)
    # FLAG
    res_flag = orchestrator.apply_policy(
        policy="flag", x=explainer.x_cal, strategy="experimental.difficulty_normalized"
    )
    # ONLY_ACCEPTED
    res_acc = orchestrator.apply_policy(
        policy="only_accepted", x=explainer.x_cal, strategy="experimental.difficulty_normalized"
    )
    # ONLY_REJECTED
    res_rej = orchestrator.apply_policy(
        policy="only_rejected", x=explainer.x_cal, strategy="experimental.difficulty_normalized"
    )
    assert res_flag.metadata["matched_count"] is None
    assert len(res_acc.metadata["source_indices"]) + len(res_rej.metadata["source_indices"]) == len(
        explainer.x_cal
    )


def test_metadata_fields_and_monotonicity(monkeypatch):
    proba = np.array([[0.1, 0.9], [0.6, 0.4], [0.8, 0.2]], dtype=float)
    labels = np.array([1, 0, 1])
    explainer = make_explainer(proba, labels, difficulty_values=[1.0, 2.0, 3.0])
    orchestrator = orch.RejectOrchestrator(explainer)
    # Check metadata fields
    res = orchestrator.apply_policy(
        policy="flag", x=explainer.x_cal, strategy="experimental.difficulty_normalized"
    )
    for key in [
        "reject_strategy",
        "difficulty_normalized",
        "difficulty_min",
        "difficulty_max",
        "difficulty_mean",
        "base_ncf",
        "base_default_kind",
        "normalization_epsilon",
    ]:
        assert key in res.metadata
    # Monotonicity: as confidence increases, prediction sets should be nested.
    confs = [0.7, 0.8, 0.9, 0.95]
    prev_prediction_set = None
    for conf in confs:
        out = orchestrator.apply_policy(
            policy="flag",
            x=explainer.x_cal,
            strategy="experimental.difficulty_normalized",
            confidence=conf,
        )
        prediction_set = np.asarray(out.metadata["prediction_set"], dtype=bool)
        if prev_prediction_set is not None:
            assert np.all(prev_prediction_set <= prediction_set)
        prev_prediction_set = prediction_set


def test_experimental_strategy_regression_mode_raises():
    proba = np.array([[0.2, 0.8], [0.6, 0.4]], dtype=float)
    labels = np.array([1, 0])
    explainer = make_explainer(proba, labels)
    explainer.mode = "regression"
    orchestrator = orch.RejectOrchestrator(explainer)
    with pytest.raises(ValidationError, match="does not support regression"):
        orchestrator.apply_policy(
            policy="flag", x=explainer.x_cal, strategy="experimental.difficulty_normalized"
        )


class DifficultyEstimator:
    def __init__(self, values):
        self.values = values

    def apply(self, x):
        return self.values


class ProvenanceDifficultyEstimator(DifficultyEstimator):
    def __init__(
        self,
        values,
        *,
        fit_source="proper_train",
        uses_calibration_labels=False,
        uses_calibration_residuals=False,
        cross_fitted=None,
    ):
        super().__init__(values)
        self.fit_source = fit_source
        self.uses_calibration_labels = uses_calibration_labels
        self.uses_calibration_residuals = uses_calibration_residuals
        self.cross_fitted = cross_fitted


class RecordingConformalClassifier:
    def __init__(self):
        self.fit_alphas = None
        self.predict_p_alphas = None

    def fit(self, *, alphas, bins=None):
        self.fit_alphas = np.asarray(alphas, dtype=float)
        return self

    def predict_p(self, alphas, **kwargs):
        self.predict_p_alphas = np.asarray(alphas, dtype=float)
        return np.ones_like(self.predict_p_alphas, dtype=float)


class InverseAlphaConformalClassifier(RecordingConformalClassifier):
    def predict_p(self, alphas, **kwargs):
        self.predict_p_alphas = np.asarray(alphas, dtype=float)
        return np.clip(1.0 - self.predict_p_alphas, 0.0, 1.0)


class IntervalLearnerStub:
    def __init__(self, proba):
        self.proba = np.asarray(proba, dtype=float)

    def predict_proba(self, x, bins=None):
        return np.array(self.proba, copy=True)


def make_explainer(proba, labels, difficulty_values=None):
    proba_array = np.asarray(proba, dtype=float)
    explainer = SimpleNamespace(
        mode="classification",
        x_cal=np.arange(proba_array.shape[0], dtype=float).reshape(-1, 1),
        y_cal=np.asarray(labels, dtype=int),
        bins=None,
        interval_learner=IntervalLearnerStub(proba_array),
        reject_learner=RecordingConformalClassifier(),
        difficulty_estimator=(
            None if difficulty_values is None else DifficultyEstimator(difficulty_values)
        ),
        seed=7,
    )
    explainer.is_multiclass = lambda: False
    return explainer


def test_experimental_strategy_records_provenance_metadata_permissive_warning():
    proba = np.array([[0.2, 0.8], [0.6, 0.4]], dtype=float)
    labels = np.array([1, 0])
    explainer = make_explainer(proba, labels)
    explainer.difficulty_estimator = ProvenanceDifficultyEstimator(
        [1.0, 2.0],
        fit_source="calibration_labels",
        uses_calibration_labels=True,
        cross_fitted=False,
    )
    setattr(explainer, "reject_difficulty_provenance_policy", "permissive")

    orchestrator = orch.RejectOrchestrator(explainer)
    with pytest.warns(UserWarning, match="calibration labels/residuals"):
        result = orchestrator.apply_policy(
            policy="flag", x=explainer.x_cal, strategy="experimental.difficulty_normalized"
        )

    assert result.metadata["difficulty_estimator_provenance_available"] is True
    assert result.metadata["difficulty_estimator_provenance_warning_emitted"] is True
    assert result.metadata["difficulty_estimator_provenance_validation_mode"] == "permissive"


def test_experimental_strategy_raises_on_provenance_violation_when_strict():
    proba = np.array([[0.2, 0.8], [0.6, 0.4]], dtype=float)
    labels = np.array([1, 0])
    explainer = make_explainer(proba, labels)
    explainer.difficulty_estimator = ProvenanceDifficultyEstimator(
        [1.0, 2.0],
        fit_source="calibration_labels",
        uses_calibration_labels=True,
        cross_fitted=False,
    )
    setattr(explainer, "reject_difficulty_provenance_policy", "strict")

    orchestrator = orch.RejectOrchestrator(explainer)
    with pytest.raises(ValidationError, match="may invalidate conformal reject calibration"):
        orchestrator.apply_policy(
            policy="flag", x=explainer.x_cal, strategy="experimental.difficulty_normalized"
        )


def test_builtin_default_strategy_ignores_provenance_policy_warning_path():
    proba = np.array([[0.2, 0.8], [0.6, 0.4]], dtype=float)
    labels = np.array([1, 0])
    explainer = make_explainer(proba, labels)
    explainer.difficulty_estimator = ProvenanceDifficultyEstimator(
        [1.0, 2.0],
        fit_source="calibration_labels",
        uses_calibration_labels=True,
        cross_fitted=False,
    )
    setattr(explainer, "reject_difficulty_provenance_policy", "strict")

    orchestrator = orch.RejectOrchestrator(explainer)
    result = orchestrator.apply_policy(policy="flag", x=explainer.x_cal, strategy="builtin.default")
    assert "difficulty_estimator_provenance_available" not in (result.metadata or {})


def test_should_return_unchanged_calibration_scores_when_difficulty_estimator_is_missing(
    monkeypatch,
):
    proba = np.array([[0.2, 0.8], [0.6, 0.4]], dtype=float)
    labels = np.array([1, 0])
    explainer = make_explainer(proba, labels)
    conformal = RecordingConformalClassifier()
    monkeypatch.setattr(orch, "ConformalClassifier", lambda: conformal)

    orch.RejectOrchestrator(explainer).initialize_reject_learner(ncf="default", w=0.5)
    expected = default_score_cal(proba, np.unique(labels), labels, "hinge")

    np.testing.assert_allclose(conformal.fit_alphas, expected)


def test_should_return_unchanged_test_scores_when_difficulty_estimator_is_missing(monkeypatch):
    proba = np.array([[0.2, 0.8], [0.6, 0.4]], dtype=float)
    labels = np.array([1, 0])
    explainer = make_explainer(proba, labels)
    conformal = RecordingConformalClassifier()
    monkeypatch.setattr(orch, "ConformalClassifier", lambda: conformal)

    orchestrator = orch.RejectOrchestrator(explainer)
    orchestrator.initialize_reject_learner(ncf="default", w=0.5)
    orchestrator.predict_reject_breakdown(explainer.x_cal, confidence=0.9)
    expected = default_score_test(proba, "hinge")

    np.testing.assert_allclose(conformal.predict_p_alphas, expected)


def test_should_lower_calibration_scores_when_difficulty_increases(monkeypatch):
    proba = np.array([[0.2, 0.8], [0.6, 0.4]], dtype=float)
    labels = np.array([1, 0])
    explainer = make_explainer(proba, labels, difficulty_values=[1.0, 2.0])
    setattr(explainer, "_reject_difficulty_normalized", True)
    conformal = RecordingConformalClassifier()
    monkeypatch.setattr(orch, "ConformalClassifier", lambda: conformal)

    orch.RejectOrchestrator(explainer).initialize_reject_learner(ncf="default", w=0.5)
    base_scores = default_score_cal(proba, np.unique(labels), labels, "hinge")

    np.testing.assert_allclose(conformal.fit_alphas, base_scores / np.array([1.0, 2.0]))
    assert conformal.fit_alphas[1] < base_scores[1]


def test_should_normalize_test_scores_row_wise_when_difficulty_varies_by_instance(monkeypatch):
    proba = np.array([[0.2, 0.8], [0.6, 0.4]], dtype=float)
    labels = np.array([1, 0])
    explainer = make_explainer(proba, labels, difficulty_values=[2.0, 4.0])
    setattr(explainer, "_reject_difficulty_normalized", True)
    conformal = RecordingConformalClassifier()
    monkeypatch.setattr(orch, "ConformalClassifier", lambda: conformal)

    orchestrator = orch.RejectOrchestrator(explainer)
    orchestrator.initialize_reject_learner(ncf="default", w=0.5)
    orchestrator.predict_reject_breakdown(explainer.x_cal, confidence=0.9)
    base_scores = default_score_test(proba, "hinge")

    expected = base_scores / np.array([[2.0], [4.0]])
    np.testing.assert_allclose(conformal.predict_p_alphas, expected)


def test_should_raise_validation_error_when_difficulty_shape_is_invalid(monkeypatch):
    proba = np.array([[0.2, 0.8], [0.6, 0.4]], dtype=float)
    labels = np.array([1, 0])
    explainer = make_explainer(proba, labels, difficulty_values=[[1.0, 2.0, 3.0]])
    setattr(explainer, "_reject_difficulty_normalized", True)
    monkeypatch.setattr(orch, "ConformalClassifier", RecordingConformalClassifier)

    with pytest.raises(ValidationError, match="one value per instance"):
        orch.RejectOrchestrator(explainer).initialize_reject_learner(ncf="default", w=0.5)


@pytest.mark.parametrize("values", ([1.0, np.nan], [1.0, np.inf]))
def test_should_raise_validation_error_when_difficulty_values_are_not_finite(monkeypatch, values):
    proba = np.array([[0.2, 0.8], [0.6, 0.4]], dtype=float)
    labels = np.array([1, 0])
    explainer = make_explainer(proba, labels, difficulty_values=values)
    setattr(explainer, "_reject_difficulty_normalized", True)
    monkeypatch.setattr(orch, "ConformalClassifier", RecordingConformalClassifier)

    with pytest.raises(ValidationError, match="difficulty values must be finite"):
        orch.RejectOrchestrator(explainer).initialize_reject_learner(ncf="default", w=0.5)


def test_should_raise_validation_error_when_difficulty_values_are_negative(monkeypatch):
    proba = np.array([[0.2, 0.8], [0.6, 0.4]], dtype=float)
    labels = np.array([1, 0])
    explainer = make_explainer(proba, labels, difficulty_values=[1.0, -0.5])
    setattr(explainer, "_reject_difficulty_normalized", True)
    monkeypatch.setattr(orch, "ConformalClassifier", RecordingConformalClassifier)

    with pytest.raises(ValidationError, match="difficulty values must be non-negative"):
        orch.RejectOrchestrator(explainer).initialize_reject_learner(ncf="default", w=0.5)


def test_should_clip_near_zero_difficulty_when_normalizing_scores(monkeypatch):
    proba = np.array([[0.2, 0.8], [0.6, 0.4]], dtype=float)
    labels = np.array([1, 0])
    explainer = make_explainer(proba, labels, difficulty_values=[0.0, 1e-12])
    setattr(explainer, "_reject_difficulty_normalized", True)
    conformal = RecordingConformalClassifier()
    monkeypatch.setattr(orch, "ConformalClassifier", lambda: conformal)

    orch.RejectOrchestrator(explainer).initialize_reject_learner(ncf="default", w=0.5)
    base_scores = default_score_cal(proba, np.unique(labels), labels, "hinge")

    np.testing.assert_allclose(conformal.fit_alphas, base_scores / np.array([1e-6, 1e-6]))
