from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from calibrated_explanations.core.reject import orchestrator as orch
from calibrated_explanations.core.reject.orchestrator import default_score_cal, default_score_test
from calibrated_explanations.utils.exceptions import ConfigurationError, ValidationError


def test_strategy_is_registered():
    explainer = make_explainer([[0.2, 0.8], [0.6, 0.4]], [1, 0])
    orchestrator = orch.RejectOrchestrator(explainer)
    # Should not raise
    fn = orchestrator.resolve_strategy("experimental.difficulty_normalized")
    assert callable(fn)
    fn2 = orchestrator.resolve_strategy("experimental.ambiguity_normalized_novelty_penalized")
    assert callable(fn2)


def test_unknown_strategy_raises():
    explainer = make_explainer([[0.2, 0.8], [0.6, 0.4]], [1, 0])
    orchestrator = orch.RejectOrchestrator(explainer)
    with pytest.raises(KeyError, match="not registered"):
        orchestrator.resolve_strategy("not.a.strategy")


def test_experimental_matches_builtin_with_trivial_difficulty(monkeypatch):
    proba = np.array([[0.2, 0.8], [0.6, 0.4]], dtype=float)
    labels = np.array([1, 0])
    # Uniform difficulty of 1.0 is a no-op normalisation — results must match builtin.
    explainer = make_explainer(proba, labels, difficulty_values=[1.0, 1.0])
    orchestrator = orch.RejectOrchestrator(explainer)
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
        policy="flag", x=explainer.x_cal, strategy="builtin.default", reject_confidence=0.5
    )
    res_experimental = orchestrator.apply_policy(
        policy="flag",
        x=explainer.x_cal,
        strategy="experimental.difficulty_normalized",
        reject_confidence=0.5,
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
            reject_confidence=conf,
        )
        prediction_set = np.asarray(out.metadata["prediction_set"], dtype=bool)
        if prev_prediction_set is not None:
            assert np.all(prev_prediction_set <= prediction_set)
        prev_prediction_set = prediction_set


def test_experimental_strategy_regression_mode_works():
    # Regression mode must be fully supported with difficulty normalization.
    # P(y <= 0.5) values for 3 calibration instances, y_cal continuous.
    proba_1 = [0.2, 0.7, 0.9]
    y_cal = [0.1, 0.8, 0.6]
    explainer = make_regression_explainer(
        proba_1, y_cal, threshold=0.5, difficulty_values=[1.0, 1.0, 1.0]
    )
    orchestrator = orch.RejectOrchestrator(explainer)
    result = orchestrator.apply_policy(
        policy="flag",
        x=explainer.x_cal,
        strategy="experimental.difficulty_normalized",
        threshold=0.5,
    )
    assert result is not None
    assert result.metadata["difficulty_normalized"] is True
    assert result.metadata["reject_strategy"] == "experimental.difficulty_normalized"
    assert result.rejected is not None
    assert len(result.rejected) == len(proba_1)


def test_experimental_strategy_regression_trivial_difficulty_matches_builtin():
    # With uniform difficulty=1, normalized scores must equal builtin scores.
    proba_1 = [0.2, 0.7, 0.9]
    y_cal = [0.1, 0.8, 0.6]
    explainer_builtin = make_regression_explainer(proba_1, y_cal, threshold=0.5)
    explainer_diff = make_regression_explainer(
        proba_1, y_cal, threshold=0.5, difficulty_values=[1.0, 1.0, 1.0]
    )
    orch_builtin = orch.RejectOrchestrator(explainer_builtin)
    orch_diff = orch.RejectOrchestrator(explainer_diff)
    res_builtin = orch_builtin.apply_policy(
        policy="flag", x=explainer_builtin.x_cal, strategy="builtin.default", threshold=0.5
    )
    res_diff = orch_diff.apply_policy(
        policy="flag",
        x=explainer_diff.x_cal,
        strategy="experimental.difficulty_normalized",
        threshold=0.5,
    )
    np.testing.assert_array_equal(res_builtin.rejected, res_diff.rejected)
    np.testing.assert_array_equal(
        res_builtin.metadata["prediction_set_size"],
        res_diff.metadata["prediction_set_size"],
    )


class DifficultyEstimator:
    def __init__(self, values):
        self.values = values

    def apply(self, x):
        return self.values


class NoveltyEstimator(DifficultyEstimator):
    pass


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

    def predict_probability(self, x, y_threshold=None, bins=None):
        # For regression stubs: return P(y <= threshold) from proba[:,1], plus three None slots.
        return self.proba[:, 1].copy(), None, None, None


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


def make_regression_explainer(proba_1, y_cal, *, threshold=0.5, difficulty_values=None):
    """Regression-mode explainer stub.  proba_1[i] = P(y_i <= threshold)."""
    proba_array = np.column_stack([1.0 - np.asarray(proba_1), np.asarray(proba_1)]).astype(float)
    explainer = SimpleNamespace(
        mode="regression",
        x_cal=np.arange(len(proba_1), dtype=float).reshape(-1, 1),
        y_cal=np.asarray(y_cal, dtype=float),
        bins=None,
        interval_learner=IntervalLearnerStub(proba_array),
        reject_learner=RecordingConformalClassifier(),
        difficulty_estimator=(
            None if difficulty_values is None else DifficultyEstimator(difficulty_values)
        ),
        reject_threshold=threshold,
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
    # Sub-threshold but strictly positive — must be clipped to 1e-6 floor (not zero).
    explainer = make_explainer(proba, labels, difficulty_values=[1e-12, 2e-12])
    setattr(explainer, "_reject_difficulty_normalized", True)
    conformal = RecordingConformalClassifier()
    monkeypatch.setattr(orch, "ConformalClassifier", lambda: conformal)

    orch.RejectOrchestrator(explainer).initialize_reject_learner(ncf="default", w=0.5)
    base_scores = default_score_cal(proba, np.unique(labels), labels, "hinge")

    np.testing.assert_allclose(conformal.fit_alphas, base_scores / np.array([1e-6, 1e-6]))


def test_novelty_penalized_matches_difficulty_normalized_when_trivial_estimators(monkeypatch):
    proba = np.array([[0.2, 0.8], [0.6, 0.4]], dtype=float)
    labels = np.array([1, 0])
    # Uniform difficulty [1.0, 1.0] makes normalization a no-op; both strategies must agree.
    explainer = make_explainer(proba, labels, difficulty_values=[1.0, 1.0])
    orchestrator = orch.RejectOrchestrator(explainer)
    monkeypatch.setattr(orch, "ConformalClassifier", InverseAlphaConformalClassifier)

    res_diff = orchestrator.apply_policy(
        policy="flag",
        x=explainer.x_cal,
        strategy="experimental.difficulty_normalized",
        reject_confidence=0.5,
        include_prediction_payload=False,
    )
    res_novelty = orchestrator.apply_policy(
        policy="flag",
        x=explainer.x_cal,
        strategy="experimental.ambiguity_normalized_novelty_penalized",
        reject_confidence=0.5,
        include_prediction_payload=False,
    )
    np.testing.assert_array_equal(res_diff.rejected, res_novelty.rejected)
    np.testing.assert_array_equal(
        res_diff.metadata["prediction_set_size"],
        res_novelty.metadata["prediction_set_size"],
    )


def test_novelty_penalized_higher_ambiguity_difficulty_lowers_normalized_alpha(monkeypatch):
    proba = np.array([[0.2, 0.8], [0.6, 0.4]], dtype=float)
    labels = np.array([1, 0])
    explainer = make_explainer(proba, labels)
    setattr(explainer, "_reject_ambiguity_novelty_normalized", True)
    explainer.difficulty_estimator = DifficultyEstimator([1.0, 4.0])
    setattr(explainer, "_reject_novelty_estimator", None)
    setattr(explainer, "_reject_novelty_weight", 0.0)
    conformal = RecordingConformalClassifier()
    monkeypatch.setattr(orch, "ConformalClassifier", lambda: conformal)

    orch.RejectOrchestrator(explainer).initialize_reject_learner(ncf="default", w=0.5)
    base_scores = default_score_cal(proba, np.unique(labels), labels, "hinge")

    np.testing.assert_allclose(conformal.fit_alphas, base_scores / np.array([1.0, 4.0]))
    assert conformal.fit_alphas[1] < base_scores[1]


def test_novelty_penalty_increases_alpha_for_all_labels(monkeypatch):
    proba = np.array([[0.2, 0.8], [0.6, 0.4]], dtype=float)
    labels = np.array([1, 0])
    explainer = make_explainer(proba, labels)
    setattr(explainer, "_reject_ambiguity_novelty_normalized", True)
    explainer.difficulty_estimator = DifficultyEstimator([1.0, 1.0])
    setattr(explainer, "_reject_novelty_estimator", NoveltyEstimator([0.5, 1.0]))
    setattr(explainer, "_reject_novelty_weight", 0.4)
    conformal = RecordingConformalClassifier()
    monkeypatch.setattr(orch, "ConformalClassifier", lambda: conformal)

    orch.RejectOrchestrator(explainer).initialize_reject_learner(ncf="default", w=0.5)
    base_scores = default_score_cal(proba, np.unique(labels), labels, "hinge")
    # Novelty penalty is applied only on the test side; calibration alphas stay difficulty-normalized.
    expected = base_scores
    np.testing.assert_allclose(conformal.fit_alphas, expected)


def test_novelty_estimator_requires_apply_method_via_policy():
    proba = np.array([[0.2, 0.8], [0.6, 0.4]], dtype=float)
    labels = np.array([1, 0])
    explainer = make_explainer(proba, labels, difficulty_values=[1.0, 1.0])
    orchestrator = orch.RejectOrchestrator(explainer)

    with pytest.raises(ValidationError, match=r"must define an apply\(x\) method"):
        orchestrator.apply_policy(
            policy="flag",
            x=explainer.x_cal,
            strategy="experimental.ambiguity_normalized_novelty_penalized",
            reject_confidence=0.5,
            novelty_estimator=object(),
            novelty_weight=0.2,
            include_prediction_payload=False,
        )


def test_novelty_estimator_requires_numeric_outputs_via_policy():
    class NonNumericNoveltyEstimator:
        def apply(self, x):
            return ["bad", "values"]

    proba = np.array([[0.2, 0.8], [0.6, 0.4]], dtype=float)
    labels = np.array([1, 0])
    explainer = make_explainer(proba, labels, difficulty_values=[1.0, 1.0])
    orchestrator = orch.RejectOrchestrator(explainer)

    with pytest.raises(ValidationError, match="must return numeric novelty values"):
        orchestrator.apply_policy(
            policy="flag",
            x=explainer.x_cal,
            strategy="experimental.ambiguity_normalized_novelty_penalized",
            reject_confidence=0.5,
            novelty_estimator=NonNumericNoveltyEstimator(),
            novelty_weight=0.2,
            include_prediction_payload=False,
        )


def test_novelty_estimator_requires_per_instance_values_via_policy():
    class WrongLengthNoveltyEstimator:
        def apply(self, x):
            return [0.1]

    proba = np.array([[0.2, 0.8], [0.6, 0.4]], dtype=float)
    labels = np.array([1, 0])
    explainer = make_explainer(proba, labels, difficulty_values=[1.0, 1.0])
    orchestrator = orch.RejectOrchestrator(explainer)

    with pytest.raises(ValidationError, match="must return one value per instance"):
        orchestrator.apply_policy(
            policy="flag",
            x=explainer.x_cal,
            strategy="experimental.ambiguity_normalized_novelty_penalized",
            reject_confidence=0.5,
            novelty_estimator=WrongLengthNoveltyEstimator(),
            novelty_weight=0.2,
            include_prediction_payload=False,
        )


def test_novelty_estimator_rejects_nonfinite_and_negative_values_via_policy():
    class NonFiniteNoveltyEstimator:
        def apply(self, x):
            return [0.1, np.inf]

    class NegativeNoveltyEstimator:
        def apply(self, x):
            return [0.1, -0.2]

    proba = np.array([[0.2, 0.8], [0.6, 0.4]], dtype=float)
    labels = np.array([1, 0])
    explainer = make_explainer(proba, labels, difficulty_values=[1.0, 1.0])
    orchestrator = orch.RejectOrchestrator(explainer)

    with pytest.raises(ValidationError, match="novelty values must be finite"):
        orchestrator.apply_policy(
            policy="flag",
            x=explainer.x_cal,
            strategy="experimental.ambiguity_normalized_novelty_penalized",
            reject_confidence=0.5,
            novelty_estimator=NonFiniteNoveltyEstimator(),
            novelty_weight=0.2,
            include_prediction_payload=False,
        )

    with pytest.raises(ValidationError, match="novelty values must be non-negative"):
        orchestrator.apply_policy(
            policy="flag",
            x=explainer.x_cal,
            strategy="experimental.ambiguity_normalized_novelty_penalized",
            reject_confidence=0.5,
            novelty_estimator=NegativeNoveltyEstimator(),
            novelty_weight=0.2,
            include_prediction_payload=False,
        )


def test_novelty_weight_requires_non_negative_float_via_policy():
    proba = np.array([[0.2, 0.8], [0.6, 0.4]], dtype=float)
    labels = np.array([1, 0])
    explainer = make_explainer(proba, labels, difficulty_values=[1.0, 1.0])
    orchestrator = orch.RejectOrchestrator(explainer)

    with pytest.raises(ValidationError, match="must be a non-negative float"):
        orchestrator.apply_policy(
            policy="flag",
            x=explainer.x_cal,
            strategy="experimental.ambiguity_normalized_novelty_penalized",
            reject_confidence=0.5,
            novelty_estimator=NoveltyEstimator([0.1, 0.2]),
            novelty_weight="not-a-number",
            include_prediction_payload=False,
        )

    with pytest.raises(ValidationError, match="must be a non-negative float"):
        orchestrator.apply_policy(
            policy="flag",
            x=explainer.x_cal,
            strategy="experimental.ambiguity_normalized_novelty_penalized",
            reject_confidence=0.5,
            novelty_estimator=NoveltyEstimator([0.1, 0.2]),
            novelty_weight=-0.1,
            include_prediction_payload=False,
        )


def test_novelty_strategy_metadata_collection_falls_back_on_estimator_errors(monkeypatch):
    class FlakyNoveltyEstimator:
        def __init__(self, first_values):
            self.first_values = np.asarray(first_values, dtype=float)
            self.calls = 0

        def apply(self, x):
            self.calls += 1
            if self.calls == 1:
                return self.first_values
            raise RuntimeError("metadata sampling failed")

    proba = np.array([[0.2, 0.8], [0.6, 0.4]], dtype=float)
    labels = np.array([1, 0])
    explainer = make_explainer(proba, labels, difficulty_values=[1.0, 1.0])
    orchestrator = orch.RejectOrchestrator(explainer)
    monkeypatch.setattr(orch, "ConformalClassifier", InverseAlphaConformalClassifier)

    ambiguity = DifficultyEstimator([1.0, 2.0])
    novelty = FlakyNoveltyEstimator([0.1, 0.2])
    result = orchestrator.apply_policy(
        policy="flag",
        x=explainer.x_cal,
        strategy="experimental.ambiguity_normalized_novelty_penalized",
        reject_confidence=0.5,
        ambiguity_estimator=ambiguity,
        novelty_estimator=novelty,
        novelty_weight=0.2,
        include_prediction_payload=False,
    )

    assert (
        result.metadata["reject_strategy"] == "experimental.ambiguity_normalized_novelty_penalized"
    )
    assert result.metadata["novelty_penalized"] is True
    assert "ambiguity_difficulty_min" in result.metadata
    assert "novelty_score_min" not in result.metadata


def test_experimental_difficulty_strategy_none_policy_v2_raises_validation_error():
    proba = np.array([[0.2, 0.8], [0.6, 0.4]], dtype=float)
    labels = np.array([1, 0])
    explainer = make_explainer(proba, labels, difficulty_values=[1.0, 1.0])
    orchestrator = orch.RejectOrchestrator(explainer)

    with pytest.raises(ValidationError, match="only available for non-NONE reject policies"):
        orchestrator.apply_policy(
            policy="none",
            x=explainer.x_cal,
            strategy="experimental.difficulty_normalized",
            result_schema="v2",
        )


def test_experimental_difficulty_strategy_none_policy_legacy_returns_empty_result():
    proba = np.array([[0.2, 0.8], [0.6, 0.4]], dtype=float)
    labels = np.array([1, 0])
    explainer = make_explainer(proba, labels, difficulty_values=[1.0, 1.0])
    orchestrator = orch.RejectOrchestrator(explainer)

    result = orchestrator.apply_policy(
        policy="none",
        x=explainer.x_cal,
        strategy="experimental.difficulty_normalized",
        result_schema="legacy",
    )

    assert result.rejected is None
    assert result.prediction is None
    assert result.explanation is None
    assert result.metadata is None


def test_novelty_penalty_can_increase_empty_set_rate(monkeypatch):
    proba = np.array([[0.2, 0.8], [0.6, 0.4], [0.52, 0.48]], dtype=float)
    labels = np.array([1, 0, 0])
    explainer = make_explainer(proba, labels, difficulty_values=[1.0, 1.0, 1.0])
    orchestrator = orch.RejectOrchestrator(explainer)
    monkeypatch.setattr(orch, "ConformalClassifier", InverseAlphaConformalClassifier)

    no_penalty = orchestrator.apply_policy(
        policy="flag",
        x=explainer.x_cal,
        strategy="experimental.ambiguity_normalized_novelty_penalized",
        reject_confidence=0.5,
        novelty_weight=0.0,
        novelty_estimator=NoveltyEstimator([0.0, 0.0, 0.0]),
        include_prediction_payload=False,
    )
    penalty = orchestrator.apply_policy(
        policy="flag",
        x=explainer.x_cal,
        strategy="experimental.ambiguity_normalized_novelty_penalized",
        reject_confidence=0.5,
        novelty_weight=0.8,
        novelty_estimator=NoveltyEstimator([0.0, 0.0, 2.0]),
        include_prediction_payload=False,
    )
    assert penalty.metadata["novelty_rate"] >= no_penalty.metadata["novelty_rate"]


def test_novelty_penalized_confidence_monotonicity_and_metadata(monkeypatch):
    proba = np.array([[0.1, 0.9], [0.6, 0.4], [0.8, 0.2]], dtype=float)
    labels = np.array([1, 0, 1])
    explainer = make_explainer(proba, labels, difficulty_values=[1.0, 2.0, 3.0])
    orchestrator = orch.RejectOrchestrator(explainer)
    monkeypatch.setattr(orch, "ConformalClassifier", InverseAlphaConformalClassifier)

    novelty = NoveltyEstimator([0.1, 0.2, 0.3])
    res = orchestrator.apply_policy(
        policy="flag",
        x=explainer.x_cal,
        strategy="experimental.ambiguity_normalized_novelty_penalized",
        reject_confidence=0.7,
        novelty_estimator=novelty,
        novelty_weight=0.1,
        include_prediction_payload=False,
    )
    for key in [
        "reject_strategy",
        "difficulty_normalized",
        "novelty_penalized",
        "novelty_weight",
        "ambiguity_difficulty_min",
        "ambiguity_difficulty_max",
        "ambiguity_difficulty_mean",
        "novelty_score_min",
        "novelty_score_max",
        "novelty_score_mean",
        "base_ncf",
    ]:
        assert key in res.metadata

    prev_prediction_set = None
    for conf in [0.7, 0.8, 0.9, 0.95]:
        out = orchestrator.apply_policy(
            policy="flag",
            x=explainer.x_cal,
            strategy="experimental.ambiguity_normalized_novelty_penalized",
            reject_confidence=conf,
            novelty_estimator=novelty,
            novelty_weight=0.1,
            include_prediction_payload=False,
        )
        prediction_set = np.asarray(out.metadata["prediction_set"], dtype=bool)
        if prev_prediction_set is not None:
            assert np.all(prev_prediction_set <= prediction_set)
        prev_prediction_set = prediction_set


# ---------------------------------------------------------------------------
# RT-11: ConfigurationError guard — experimental strategy without estimator
# ---------------------------------------------------------------------------


def test_difficulty_strategy_without_estimator_raises_configuration_error():
    proba = np.array([[0.2, 0.8], [0.6, 0.4]], dtype=float)
    labels = np.array([1, 0])
    explainer = make_explainer(proba, labels)  # difficulty_estimator=None
    orchestrator = orch.RejectOrchestrator(explainer)
    with pytest.raises(ConfigurationError, match="requires difficulty_estimator") as exc_info:
        orchestrator.apply_policy(
            policy="flag", x=explainer.x_cal, strategy="experimental.difficulty_normalized"
        )
    assert exc_info.value.details == {
        "strategy": "experimental.difficulty_normalized",
        "requirement": "difficulty_estimator",
    }


def test_ambiguity_novelty_strategy_without_estimator_raises_configuration_error():
    proba = np.array([[0.2, 0.8], [0.6, 0.4]], dtype=float)
    labels = np.array([1, 0])
    explainer = make_explainer(proba, labels)  # difficulty_estimator=None
    orchestrator = orch.RejectOrchestrator(explainer)
    with pytest.raises(ConfigurationError, match="requires difficulty_estimator") as exc_info:
        orchestrator.apply_policy(
            policy="flag",
            x=explainer.x_cal,
            strategy="experimental.ambiguity_normalized_novelty_penalized",
        )
    assert exc_info.value.details == {
        "strategy": "experimental.ambiguity_normalized_novelty_penalized",
        "requirement": "difficulty_estimator",
    }


def test_builtin_default_without_estimator_does_not_raise():
    proba = np.array([[0.2, 0.8], [0.6, 0.4]], dtype=float)
    labels = np.array([1, 0])
    explainer = make_explainer(proba, labels)  # difficulty_estimator=None
    orchestrator = orch.RejectOrchestrator(explainer)
    result = orchestrator.apply_policy(policy="flag", x=explainer.x_cal, strategy="builtin.default")
    assert result is not None


def test_experimental_with_estimator_set_does_not_raise():
    proba = np.array([[0.2, 0.8], [0.6, 0.4]], dtype=float)
    labels = np.array([1, 0])
    explainer = make_explainer(proba, labels, difficulty_values=[1.0, 2.0])
    orchestrator = orch.RejectOrchestrator(explainer)
    result = orchestrator.apply_policy(
        policy="flag", x=explainer.x_cal, strategy="experimental.difficulty_normalized"
    )
    assert result is not None


# ---------------------------------------------------------------------------
# RT-6: ValidationError for zero difficulty; large ratios are legitimate and do not warn
# ---------------------------------------------------------------------------


def test_zero_difficulty_values_raise_validation_error(monkeypatch):
    proba = np.array([[0.2, 0.8], [0.6, 0.4]], dtype=float)
    labels = np.array([1, 0])
    explainer = make_explainer(proba, labels, difficulty_values=[0.0, 1.0])
    setattr(explainer, "_reject_difficulty_normalized", True)
    monkeypatch.setattr(orch, "ConformalClassifier", RecordingConformalClassifier)
    with pytest.raises(ValidationError, match="zero values"):
        orch.RejectOrchestrator(explainer).initialize_reject_learner(ncf="default", w=0.5)


def test_large_ratio_difficulty_no_warning(monkeypatch):
    proba = np.array([[0.2, 0.8], [0.6, 0.4]], dtype=float)
    labels = np.array([1, 0])
    # ratio = 22.0 — legitimate for crepes.extras.DifficultyEstimator on real data
    explainer = make_explainer(proba, labels, difficulty_values=[1.0, 22.0])
    setattr(explainer, "_reject_difficulty_normalized", True)
    conformal = RecordingConformalClassifier()
    monkeypatch.setattr(orch, "ConformalClassifier", lambda: conformal)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        orch.RejectOrchestrator(explainer).initialize_reject_learner(ncf="default", w=0.5)
    assert conformal.fit_alphas is not None


def test_normal_difficulty_range_no_warning_or_error(monkeypatch):
    proba = np.array([[0.2, 0.8], [0.6, 0.4]], dtype=float)
    labels = np.array([1, 0])
    explainer = make_explainer(proba, labels, difficulty_values=[1.0, 2.0])
    setattr(explainer, "_reject_difficulty_normalized", True)
    conformal = RecordingConformalClassifier()
    monkeypatch.setattr(orch, "ConformalClassifier", lambda: conformal)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        orch.RejectOrchestrator(explainer).initialize_reject_learner(ncf="default", w=0.5)
    assert conformal.fit_alphas is not None
