"""Red-team tests for reject NCF implementations.

These tests exercise the public NCF contract ``{default, ensured}`` plus
legacy ``entropy`` input mapping to ``default``.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from calibrated_explanations.core.reject import orchestrator as orch


def test_reject_input_validators_reject_non_numeric_payloads():
    from calibrated_explanations.utils.exceptions import ValidationError  # noqa: PLC0415

    with pytest.raises(ValidationError, match="confidence must be a float"):
        orch.validate_reject_confidence("bad")

    with pytest.raises(ValidationError, match="w must be a float"):
        orch.validate_reject_w("bad")


def test_initialize_reject_learner_warns_on_low_w(monkeypatch):
    # Create a minimal explainer stub that the orchestrator expects.
    class StubIntervalLearner:
        def predict_proba(self, x, bins=None):
            # Two-class probability matrix
            proba = np.tile(np.array([[0.2, 0.8]]), (len(x), 1))
            # predicted_labels for multiclass branch isn't used in binary
            return proba

        def predict_probability(self, x, y_threshold=None, bins=None):
            # For regression path - not used in this test
            return np.zeros(len(x)), None, None, None

    class ExplainerStub(SimpleNamespace):
        def __init__(self):
            super().__init__()
            self.mode = "classification"
            self.x_cal = np.array([[0], [1], [2], [3]])
            self.y_cal = np.array([0, 1, 0, 1])
            self.bins = None
            self.interval_learner = StubIntervalLearner()
            self.reject_learner = None
            self.reject_threshold = None

        def is_multiclass(self):
            return False

    # Stub ConformalClassifier to avoid heavy fitting; ensure fit returns an object
    class DummyLearner:
        def fit(self, *args, **kwargs):
            return self

    monkeypatch.setattr(orch, "ConformalClassifier", lambda: DummyLearner())

    expl = ExplainerStub()
    orchestrator = orch.RejectOrchestrator(expl)

    # Non-hinge NCF with very low w should emit a UserWarning
    with pytest.warns(UserWarning, match="ncf='ensured' with w=0.05"):
        learner = orchestrator.initialize_reject_learner(ncf="ensured", w=0.05)

    # Attributes should be set on the explainer
    assert expl.reject_ncf == "ensured"
    assert abs(expl.reject_ncf_w - 0.05) < 1e-8
    assert learner is not None


def test_predict_reject_breakdown_runs_for_all_ncfs(monkeypatch):
    """Ensure the public APIs run for each supported NCF and return expected keys."""

    class StubIntervalLearner:
        def predict_proba(self, x, bins=None):
            # simple two-class predictions for each instance
            return np.tile(np.array([[0.3, 0.7]]), (len(x), 1))

        def predict_probability(self, x, y_threshold=None, bins=None):
            return np.zeros(len(x)), None, None, None

    class ExplainerStub(SimpleNamespace):
        def __init__(self):
            super().__init__()
            self.mode = "classification"
            self.x_cal = np.array([[0], [1], [2]])
            self.y_cal = np.array([0, 1, 0])
            self.bins = None
            self.interval_learner = StubIntervalLearner()
            self.reject_learner = None

        def is_multiclass(self):
            return False

    # Dummy conformal that produces deterministic p-values and predict_set
    class DummyConformal:
        def fit(self, *args, **kwargs):
            return self

        def predict_p(self, alphas, **kwargs):
            # return high p-values so sets are non-empty
            n = np.atleast_2d(alphas).shape[0]
            k = 2
            return np.full((n, k), 0.9)

        def predict_set(self, alphas, **kwargs):
            n = np.atleast_2d(alphas).shape[0]
            k = 2
            return np.ones((n, k), dtype=bool)

    monkeypatch.setattr(orch, "ConformalClassifier", lambda: DummyConformal())

    expl = ExplainerStub()
    orchestrator = orch.RejectOrchestrator(expl)

    for ncf in ("default", "ensured", "entropy"):
        # Include legacy 'entropy' mapping alongside public values.
        orchestrator.initialize_reject_learner(ncf=ncf, w=0.5)
        breakdown = orchestrator.predict_reject_breakdown([[0], [1], [2]], confidence=0.95)

        # Basic sanity checks on returned structure
        assert isinstance(breakdown, dict)
        assert "rejected" in breakdown
        assert "prediction_set" in breakdown
        ps = breakdown["prediction_set"]
        assert ps.ndim == 2
        assert ps.shape[0] == 3


def test_multiclass_default_ncf_predict_breakdown_uses_public_path(monkeypatch):
    class StubIntervalLearner:
        def predict_proba(self, x, bins=None):
            proba = np.tile(np.array([[0.1, 0.3, 0.6]]), (len(x), 1))
            predicted_labels = np.argmax(proba, axis=1)
            return proba, predicted_labels

        def predict_probability(self, x, y_threshold=None, bins=None):
            return np.zeros(len(x)), None, None, None

    class ExplainerStub(SimpleNamespace):
        def __init__(self):
            super().__init__()
            self.mode = "classification"
            self.x_cal = np.array([[0], [1], [2], [3], [4], [5]])
            self.y_cal = np.array([0, 1, 2, 0, 1, 2])
            self.bins = None
            self.interval_learner = StubIntervalLearner()
            self.reject_learner = None

        def is_multiclass(self):
            return True

    class DummyConformal:
        def fit(self, *args, **kwargs):
            return self

        def predict_p(self, alphas, **kwargs):
            n = np.atleast_2d(alphas).shape[0]
            return np.tile(np.array([[0.8, 0.7, 0.6]]), (n, 1))

    monkeypatch.setattr(orch, "ConformalClassifier", lambda: DummyConformal())
    expl = ExplainerStub()
    orchestrator = orch.RejectOrchestrator(expl)
    orchestrator.initialize_reject_learner(ncf="default", w=0.5)

    breakdown = orchestrator.predict_reject_breakdown([[0], [1], [2]], confidence=0.95)
    assert breakdown["prediction_set"].shape == (3, 3)
    assert breakdown["rejected"].shape == (3,)


def test_predict_reject_breakdown_legacy_non_numeric_error_rate_is_normalized():
    class LegacyRejectOrchestrator(orch.RejectOrchestrator):
        def predict_reject(self, x, bins=None, confidence=0.95, threshold=None):
            return np.array([True, False]), "not-a-number", 0.5

    orchestrator = LegacyRejectOrchestrator(SimpleNamespace())
    breakdown = orchestrator.predict_reject_breakdown([[0], [1]], confidence=0.95)
    assert breakdown["error_rate"] == 0.0
    assert breakdown["error_rate_defined"] is False


# ---------------------------------------------------------------------------
# Shared stub factory
# ---------------------------------------------------------------------------


def make_stub(monkeypatch, *, all_empty: bool = False, singletons: bool = False):
    """Return (ExplainerStub, RejectOrchestrator) with a deterministic DummyConformal.

    Modes (mutually exclusive; all_empty takes priority):
    - all_empty=True : every p-value is 0.0 → all prediction sets are empty.
    - singletons=True: class-0 p-value is high, class-1 p-value is below threshold
                       → every prediction set is a singleton {0} (accepted).
    - default        : both p-values are 0.9 → ambiguous sets (size 2), all rejected.
    """

    class StubIntervalLearner:
        def predict_proba(self, x, bins=None):
            return np.tile(np.array([[0.5, 0.5]]), (len(x), 1))

        def predict_probability(self, x, y_threshold=None, bins=None):
            return np.zeros(len(x)), None, None, None

    class ExplainerStub(SimpleNamespace):
        def __init__(self):
            super().__init__()
            self.mode = "classification"
            self.x_cal = np.array([[0], [1], [2], [3]])
            self.y_cal = np.array([0, 1, 0, 1])
            self.bins = None
            self.interval_learner = StubIntervalLearner()
            self.reject_learner = None

        def is_multiclass(self):
            return False

    if all_empty:
        p_row = [0.0, 0.0]  # both fail → empty sets
    elif singletons:
        p_row = [0.9, 0.01]  # only class-0 passes → singleton {0}
    else:
        p_row = [0.9, 0.9]  # both pass → ambiguous sets (size 2)

    class DummyConformal:
        def fit(self, *args, **kwargs):
            return self

        def predict_p(self, alphas, **kwargs):
            n = np.atleast_2d(alphas).shape[0]
            return np.tile(np.array(p_row), (n, 1))

    monkeypatch.setattr(orch, "ConformalClassifier", lambda: DummyConformal())
    expl = ExplainerStub()
    orchestrator = orch.RejectOrchestrator(expl)
    orchestrator.initialize_reject_learner(ncf="default", w=1.0)
    return expl, orchestrator


# ---------------------------------------------------------------------------
# MT-3 — error_rate is always >= 0 and error_rate_defined reflects validity
# ---------------------------------------------------------------------------


def test_error_rate_nonnegative_all_empty(monkeypatch):
    """error_rate must be >= 0 even when all prediction sets are empty."""
    _, orchestrator = make_stub(monkeypatch, all_empty=True)
    breakdown = orchestrator.predict_reject_breakdown([[0], [1], [2]], confidence=0.99)
    assert breakdown["error_rate"] >= 0.0


def test_error_rate_defined_false_when_no_singletons(monkeypatch):
    """error_rate_defined is False when there are no singleton prediction sets."""
    _, orchestrator = make_stub(monkeypatch, all_empty=True)
    breakdown = orchestrator.predict_reject_breakdown([[0], [1], [2]], confidence=0.99)
    assert breakdown["error_rate_defined"] is False


def test_error_rate_defined_true_when_singletons_exist(monkeypatch):
    """error_rate_defined is True when at least one singleton prediction set exists."""
    # singletons=True → only class-0 passes threshold → every set is {0} (singleton)
    _, orchestrator = make_stub(monkeypatch, singletons=True)
    breakdown = orchestrator.predict_reject_breakdown([[0], [1], [2]], confidence=0.95)
    assert breakdown["error_rate_defined"] is True


# ---------------------------------------------------------------------------
# MT-9 — auto-selected NCF is recorded on the explainer
# ---------------------------------------------------------------------------


def test_ncf_auto_selected_true_when_implicit(monkeypatch):
    """reject_ncf_auto_selected is True when ncf is not specified explicitly."""
    expl, orchestrator = make_stub(monkeypatch)
    expl.reject_learner = None
    orchestrator.initialize_reject_learner()
    assert expl.reject_ncf_auto_selected is True


def test_ncf_auto_selected_false_when_explicit(monkeypatch):
    """reject_ncf_auto_selected is False when ncf is specified explicitly."""
    expl, orchestrator = make_stub(monkeypatch)
    orchestrator.initialize_reject_learner(ncf="default")
    assert expl.reject_ncf_auto_selected is False


# ---------------------------------------------------------------------------
# MT-10 — w=0.0 raises ValidationError only for ensured NCF
# ---------------------------------------------------------------------------


def test_w_zero_raises_for_ensured_ncf(monkeypatch):
    """w=0.0 with ensured must raise ValidationError."""
    from calibrated_explanations.utils.exceptions import ValidationError  # noqa: PLC0415

    _, orchestrator = make_stub(monkeypatch)
    with pytest.raises(ValidationError, match="w=0.0"):
        orchestrator.initialize_reject_learner(ncf="ensured", w=0.0)


def test_w_zero_allowed_for_default_and_entropy(monkeypatch):
    """w is accepted/ignored for non-ensured default NCF mode."""
    _, orchestrator = make_stub(monkeypatch)
    orchestrator.initialize_reject_learner(ncf="default", w=0.0)
    orchestrator.initialize_reject_learner(ncf="entropy", w=0.0)
    assert orchestrator.explainer.reject_ncf == "default"
    assert abs(orchestrator.explainer.reject_ncf_w - 0.0) < 1e-8


@pytest.mark.parametrize("bad_w", [-0.01, 1.01])
def test_w_out_of_bounds_raises_validation_error(monkeypatch, bad_w):
    """w outside [0,1] must fail fast before any NCF-specific handling."""
    from calibrated_explanations.utils.exceptions import ValidationError  # noqa: PLC0415

    _, orchestrator = make_stub(monkeypatch)
    with pytest.raises(ValidationError, match="w must be a float in the closed interval"):
        orchestrator.initialize_reject_learner(ncf="default", w=bad_w)


@pytest.mark.parametrize("ncf", ["hinge", "margin"])
def test_removed_explicit_ncf_inputs_raise_validation_error(monkeypatch, ncf):
    """Explicit hinge/margin user inputs are removed and must fail fast."""
    from calibrated_explanations.utils.exceptions import ValidationError  # noqa: PLC0415

    _, orchestrator = make_stub(monkeypatch)
    with pytest.raises(ValidationError, match="no longer supported"):
        orchestrator.initialize_reject_learner(ncf=ncf, w=0.5)


def test_default_entropy_breakdown_invariant_to_w(monkeypatch):
    """default/entropy public reject breakdown must be invariant to w."""

    class StubIntervalLearner:
        def predict_proba(self, x, bins=None):
            return np.array(
                [
                    [0.1, 0.9],
                    [0.3, 0.7],
                    [0.45, 0.55],
                    [0.6, 0.4],
                ],
                dtype=float,
            )[: len(x)]

        def predict_probability(self, x, y_threshold=None, bins=None):
            return np.zeros(len(x)), None, None, None

    class ExplainerStub(SimpleNamespace):
        def __init__(self):
            super().__init__()
            self.mode = "classification"
            self.x_cal = np.array([[0], [1], [2], [3]])
            self.y_cal = np.array([0, 1, 0, 1])
            self.bins = None
            self.interval_learner = StubIntervalLearner()
            self.reject_learner = None

        def is_multiclass(self):
            return False

    class DummyConformal:
        def fit(self, *args, **kwargs):
            return self

        def predict_p(self, alphas, **kwargs):
            # propagate alpha differences into p-values so NCF score changes
            # would affect the public breakdown path.
            arr = np.asarray(alphas, dtype=float)
            return np.clip(arr, 0.0, 1.0)

    monkeypatch.setattr(orch, "ConformalClassifier", lambda: DummyConformal())

    expl = ExplainerStub()
    orchestrator = orch.RejectOrchestrator(expl)
    x = np.array([[0], [1], [2], [3]])

    orchestrator.initialize_reject_learner(ncf="default", w=0.1)
    default_low = orchestrator.predict_reject_breakdown(x, confidence=0.95)
    orchestrator.initialize_reject_learner(ncf="default", w=0.9)
    default_high = orchestrator.predict_reject_breakdown(x, confidence=0.95)
    assert np.array_equal(default_low["prediction_set_size"], default_high["prediction_set_size"])

    orchestrator.initialize_reject_learner(ncf="entropy", w=0.1)
    entropy_low = orchestrator.predict_reject_breakdown(x, confidence=0.95)
    orchestrator.initialize_reject_learner(ncf="entropy", w=0.9)
    entropy_high = orchestrator.predict_reject_breakdown(x, confidence=0.95)
    assert np.array_equal(entropy_low["prediction_set_size"], entropy_high["prediction_set_size"])


@pytest.mark.parametrize("bad_confidence", [0.0, 1.0, -0.1, 1.1])
def test_predict_reject_breakdown_rejects_invalid_confidence(monkeypatch, bad_confidence):
    """confidence outside (0,1) must raise ValidationError."""
    from calibrated_explanations.utils.exceptions import ValidationError  # noqa: PLC0415

    _, orchestrator = make_stub(monkeypatch, singletons=True)
    with pytest.raises(ValidationError, match="confidence must be a float in the open interval"):
        orchestrator.predict_reject_breakdown([[0], [1], [2]], confidence=bad_confidence)


# ---------------------------------------------------------------------------
# MT-5 — plain RejectPolicy reuses existing NCF without reinit
# ---------------------------------------------------------------------------


def test_plain_policy_reuses_existing_ncf_no_reinit(monkeypatch):
    """Passing a plain RejectPolicy enum returns it unchanged; reject_ncf unchanged."""
    from calibrated_explanations.core.reject.orchestrator import resolve_policy_spec
    from calibrated_explanations.core.reject.policy import RejectPolicy

    expl, _ = make_stub(monkeypatch)
    expl.reject_ncf = "default"
    expl.reject_ncf_w = 0.5

    result = resolve_policy_spec(RejectPolicy.FLAG, expl)
    assert result is RejectPolicy.FLAG
    assert expl.reject_ncf == "default"


# ---------------------------------------------------------------------------
# MT-6 — ONLY_REJECTED with 0 matches: explanation=None, matched_count=0
# ---------------------------------------------------------------------------


def test_only_rejected_empty_subset_returns_none_and_matched_count_zero(monkeypatch):
    """When no instances are rejected, explanation is None and matched_count==0."""
    _, orchestrator = make_stub(monkeypatch, singletons=True)  # singleton sets → all accepted
    from calibrated_explanations.core.reject.policy import RejectPolicy

    explain_called = []

    def fake_explain_fn(x, **kwargs):
        explain_called.append(len(x))
        return object()

    result = orchestrator.apply_policy(
        RejectPolicy.ONLY_REJECTED,
        np.array([[0], [1], [2]]),
        explain_fn=fake_explain_fn,
        confidence=0.95,
    )
    assert result.explanation is None
    assert result.metadata["matched_count"] == 0
    assert not explain_called


def test_apply_policy_metadata_includes_effective_confidence_and_w(monkeypatch):
    """Non-NONE policy metadata includes effective confidence and w."""
    _, orchestrator = make_stub(monkeypatch, singletons=True)
    from calibrated_explanations.core.reject.policy import RejectPolicy

    result = orchestrator.apply_policy(
        RejectPolicy.FLAG,
        np.array([[0], [1], [2]]),
        explain_fn=None,
        confidence=0.77,
    )

    assert result.metadata is not None
    assert result.metadata["effective_confidence"] == pytest.approx(0.77)
    assert result.metadata["effective_w"] == pytest.approx(
        float(getattr(orchestrator.explainer, "reject_ncf_w", 0.0))
    )


def test_apply_policy_metadata_contains_required_contract_keys(monkeypatch):
    """Non-NONE policy output always exposes required WP5 metadata keys."""
    from calibrated_explanations.core.reject.policy import RejectPolicy

    expl, orchestrator = make_stub(monkeypatch, singletons=True)
    expl.prediction_orchestrator = SimpleNamespace(
        predict=lambda x, **kwargs: (np.zeros(len(x)), np.zeros(len(x)), np.ones(len(x)), None)
    )
    result = orchestrator.apply_policy(
        RejectPolicy.FLAG,
        np.array([[0], [1], [2]]),
        explain_fn=None,
        confidence=0.77,
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
    assert result.metadata is not None
    assert required.issubset(result.metadata.keys())
    assert result.metadata["policy"] == RejectPolicy.FLAG.value
    assert result.metadata["init_ok"] is True
    assert result.metadata["init_error"] is False
    assert isinstance(result.metadata["degraded_mode"], tuple)


def test_apply_policy_metadata_contract_on_init_failure(monkeypatch):
    """Init failures still return full contract metadata with explicit failure flags."""
    from calibrated_explanations.core.reject.policy import RejectPolicy

    expl = SimpleNamespace(
        mode="classification",
        x_cal=np.array([[0], [1]]),
        y_cal=np.array([0, 1]),
        bins=None,
        interval_learner=SimpleNamespace(),
        reject_learner=None,
        reject_threshold=None,
        reject_ncf="default",
        reject_ncf_w=0.5,
        reject_ncf_auto_selected=False,
        prediction_orchestrator=SimpleNamespace(predict=lambda x, **kwargs: None),
        is_multiclass=lambda: False,
    )
    orchestrator = orch.RejectOrchestrator(expl)

    def _fail_init(*args, **kwargs):
        raise RuntimeError("init boom")

    monkeypatch.setattr(orchestrator, "initialize_reject_learner", _fail_init)
    result = orchestrator.apply_policy(RejectPolicy.FLAG, np.array([[0], [1]]))
    assert result.metadata is not None
    assert result.metadata["init_ok"] is False
    assert result.metadata["init_error"] is True
    assert result.metadata["fallback_used"] is True
    assert "reject_init_failure" in result.metadata["degraded_mode"]
    assert result.metadata["policy"] == RejectPolicy.FLAG.value
    assert result.metadata["original_count"] == 2


def test_apply_policy_prediction_fallback_sets_degraded_mode(monkeypatch):
    """Prediction payload fallback must be visible in metadata."""
    from calibrated_explanations.core.reject.policy import RejectPolicy

    expl, orchestrator = make_stub(monkeypatch, singletons=True)
    expl.prediction_orchestrator = SimpleNamespace(
        predict=lambda x, **kwargs: (_ for _ in ()).throw(RuntimeError("predict boom"))
    )
    result = orchestrator.apply_policy(
        RejectPolicy.FLAG,
        np.array([[0], [1], [2]]),
        explain_fn=None,
    )
    assert result.metadata["fallback_used"] is True
    assert "prediction_payload_failed" in result.metadata["degraded_mode"]


def test_apply_policy_metadata_includes_source_indices_and_original_count(monkeypatch):
    """Non-NONE policies expose deterministic source mapping metadata."""
    _, orchestrator = make_stub(monkeypatch)
    from calibrated_explanations.core.reject.policy import RejectPolicy

    x = np.array([[0], [1], [2]], dtype=float)
    res_flag = orchestrator.apply_policy(RejectPolicy.FLAG, x, explain_fn=lambda arr, **_: arr)
    assert res_flag.metadata["source_indices"] == [0, 1, 2]
    assert res_flag.metadata["original_count"] == 3

    res_only_rej = orchestrator.apply_policy(
        RejectPolicy.ONLY_REJECTED,
        x,
        explain_fn=lambda arr, **_: arr,
    )
    assert res_only_rej.metadata["source_indices"] == [0, 1, 2]
    assert res_only_rej.metadata["original_count"] == 3

    orchestrator_singletons = make_stub(monkeypatch, singletons=True)[1]
    res_only_acc = orchestrator_singletons.apply_policy(
        RejectPolicy.ONLY_ACCEPTED,
        x,
        explain_fn=lambda arr, **_: arr,
    )
    assert res_only_acc.metadata["source_indices"] == [0, 1, 2]
    assert res_only_acc.metadata["original_count"] == 3


def test_apply_policy_source_indices_follow_reject_mask_for_subset_policies(monkeypatch):
    """Subset policies must expose source indices that match the full rejected mask."""
    _, orchestrator = make_stub(monkeypatch, singletons=True)
    from calibrated_explanations.core.reject.policy import RejectPolicy

    orchestrator.predict_reject_breakdown = lambda *args, **kwargs: {
        "rejected": np.array([True, False, True, False]),
        "error_rate": 0.0,
        "reject_rate": 0.5,
        "ambiguity_rate": 0.5,
        "novelty_rate": 0.0,
        "ambiguity": np.array([True, False, True, False]),
        "novelty": np.array([False, False, False, False]),
        "prediction_set_size": np.array([2, 1, 2, 1]),
        "prediction_set": np.array([[1, 1], [1, 0], [1, 1], [1, 0]]),
        "epsilon": 0.05,
        "raw_total_examples": 4,
        "raw_reject_counts": {"rejected": 2},
        "error_rate_defined": True,
    }

    x = np.array([[0], [1], [2], [3]], dtype=float)
    only_rej = orchestrator.apply_policy(
        RejectPolicy.ONLY_REJECTED, x, explain_fn=lambda arr, **_: arr
    )
    only_acc = orchestrator.apply_policy(
        RejectPolicy.ONLY_ACCEPTED, x, explain_fn=lambda arr, **_: arr
    )
    assert only_rej.metadata["source_indices"] == [0, 2]
    assert only_acc.metadata["source_indices"] == [1, 3]


def test_regression_apply_policy_requires_call_threshold():
    """Regression reject orchestration must fail fast when threshold is absent."""
    from calibrated_explanations.core.reject.policy import RejectPolicy
    from calibrated_explanations.utils.exceptions import ValidationError

    class StubIntervalLearner:
        def predict_probability(self, x, y_threshold=None, bins=None):
            proba = np.full(len(x), 0.5)
            return proba, np.zeros(len(x)), np.ones(len(x)), None

    class DummyPredictionOrchestrator:
        def predict(self, x, **kwargs):
            return np.zeros(len(x)), np.zeros(len(x)), np.ones(len(x)), None

    expl = SimpleNamespace(
        mode="regression",
        x_cal=np.array([[0.0], [1.0], [2.0]]),
        y_cal=np.array([0.1, 0.5, 0.9]),
        bins=None,
        interval_learner=StubIntervalLearner(),
        prediction_orchestrator=DummyPredictionOrchestrator(),
        reject_learner=object(),
        reject_threshold=0.5,
        reject_ncf="default",
        reject_ncf_w=0.5,
        reject_ncf_auto_selected=False,
        is_multiclass=lambda: False,
    )
    orchestrator = orch.RejectOrchestrator(expl)
    with pytest.raises(
        ValidationError, match="reject learner unavailable for regression without threshold"
    ):
        orchestrator.apply_policy(RejectPolicy.FLAG, np.array([[0.0], [1.0]]), explain_fn=None)


def test_regression_apply_policy_uses_call_threshold_and_reinitializes_on_mismatch(monkeypatch):
    """Regression reject decision and prediction payload share one call threshold."""
    from calibrated_explanations.core.reject.policy import RejectPolicy

    class DummyPredictionOrchestrator:
        def __init__(self):
            self.seen_thresholds = []

        def predict(self, x, **kwargs):
            self.seen_thresholds.append(kwargs.get("threshold"))
            n = len(x)
            return np.zeros(n), np.zeros(n), np.ones(n), None

    expl = SimpleNamespace(
        mode="regression",
        x_cal=np.array([[0.0], [1.0], [2.0]]),
        y_cal=np.array([0.0, 1.0, 2.0]),
        bins=None,
        interval_learner=SimpleNamespace(),
        prediction_orchestrator=DummyPredictionOrchestrator(),
        reject_learner=object(),
        reject_threshold=0.1,
        reject_ncf="default",
        reject_ncf_w=0.5,
        reject_ncf_auto_selected=False,
        is_multiclass=lambda: False,
    )
    orchestrator = orch.RejectOrchestrator(expl)

    init_calls = []
    seen_breakdown_thresholds = []

    def fake_init(calibration_set=None, threshold=None, ncf=None, w=0.5):
        init_calls.append(threshold)
        expl.reject_threshold = threshold
        expl.reject_learner = object()
        return expl.reject_learner

    def fake_breakdown(x, bins=None, confidence=0.95, threshold=None):
        seen_breakdown_thresholds.append(threshold)
        n = len(x)
        rejected = np.array([False] * n, dtype=bool)
        return {
            "rejected": rejected,
            "error_rate": 0.0,
            "reject_rate": 0.0,
            "ambiguity_rate": 0.0,
            "novelty_rate": 0.0,
            "ambiguity": rejected.copy(),
            "novelty": rejected.copy(),
            "prediction_set_size": np.ones(n, dtype=int),
            "prediction_set": np.ones((n, 2), dtype=bool),
            "epsilon": 0.05,
            "raw_total_examples": n,
            "raw_reject_counts": {"rejected": 0},
            "error_rate_defined": True,
        }

    monkeypatch.setattr(orchestrator, "initialize_reject_learner", fake_init)
    monkeypatch.setattr(orchestrator, "predict_reject_breakdown", fake_breakdown)

    with pytest.warns(UserWarning, match="threshold mismatch"):
        result = orchestrator.apply_policy(
            RejectPolicy.FLAG,
            np.array([[0.0], [1.0], [2.0]]),
            explain_fn=None,
            threshold=0.8,
        )

    assert init_calls == [0.8]
    assert seen_breakdown_thresholds == [0.8]
    assert expl.prediction_orchestrator.seen_thresholds == [0.8]
    assert result.metadata["effective_threshold"] == pytest.approx(0.8)
    assert result.metadata["threshold_source"] == "call_reinitialized"


def test_regression_apply_policy_same_threshold_does_not_reinitialize(monkeypatch):
    from calibrated_explanations.core.reject.policy import RejectPolicy

    class DummyPredictionOrchestrator:
        def predict(self, x, **kwargs):
            n = len(x)
            return np.zeros(n), np.zeros(n), np.ones(n), None

    expl = SimpleNamespace(
        mode="regression",
        x_cal=np.array([[0.0], [1.0], [2.0]]),
        y_cal=np.array([0.0, 1.0, 2.0]),
        bins=None,
        interval_learner=SimpleNamespace(),
        prediction_orchestrator=DummyPredictionOrchestrator(),
        reject_learner=object(),
        reject_threshold=np.array([0.8, 0.8, 0.8]),
        reject_ncf="default",
        reject_ncf_w=0.5,
        reject_ncf_auto_selected=False,
        is_multiclass=lambda: False,
    )
    orchestrator = orch.RejectOrchestrator(expl)

    init_calls = []
    monkeypatch.setattr(
        orchestrator,
        "initialize_reject_learner",
        lambda *args, **kwargs: init_calls.append(kwargs.get("threshold")),
    )
    monkeypatch.setattr(
        orchestrator,
        "predict_reject_breakdown",
        lambda *args, **kwargs: {
            "rejected": np.array([False, False, False]),
            "error_rate": 0.0,
            "reject_rate": 0.0,
            "ambiguity_rate": 0.0,
            "novelty_rate": 0.0,
            "ambiguity": np.array([False, False, False]),
            "novelty": np.array([False, False, False]),
            "prediction_set_size": np.array([1, 1, 1]),
            "prediction_set": np.ones((3, 2), dtype=bool),
            "epsilon": 0.05,
            "raw_total_examples": 3,
            "raw_reject_counts": {"rejected": 0},
            "error_rate_defined": True,
        },
    )

    result = orchestrator.apply_policy(
        RejectPolicy.FLAG,
        np.array([[0.0], [1.0], [2.0]]),
        threshold=np.array([0.8, 0.8, 0.8]),
    )
    assert init_calls == []
    assert result.metadata["threshold_source"] == "call"


def test_predict_reject_breakdown_legacy_override_without_threshold_arg_fallback():
    class LegacyOverrideOrchestrator(orch.RejectOrchestrator):
        def predict_reject(self, x, bins=None, confidence=0.95):  # threshold intentionally omitted
            rejected = np.array([False] * len(x), dtype=bool)
            return rejected, 0.0, 0.0

    expl = SimpleNamespace(
        mode="classification",
        x_cal=np.array([[0], [1]]),
        y_cal=np.array([0, 1]),
        bins=None,
        interval_learner=SimpleNamespace(),
        reject_learner=object(),
        is_multiclass=lambda: False,
    )
    orchestrator = LegacyOverrideOrchestrator(expl)
    breakdown = orchestrator.predict_reject_breakdown(np.array([[0], [1]]), confidence=0.95)
    assert breakdown["reject_rate"] == 0.0
    assert np.array_equal(breakdown["rejected"], np.array([False, False]))


def test_regression_apply_policy_reinitializes_on_threshold_shape_mismatch(monkeypatch):
    from calibrated_explanations.core.reject.policy import RejectPolicy

    class DummyPredictionOrchestrator:
        def predict(self, x, **kwargs):
            n = len(x)
            return np.zeros(n), np.zeros(n), np.ones(n), None

    expl = SimpleNamespace(
        mode="regression",
        x_cal=np.array([[0.0], [1.0], [2.0]]),
        y_cal=np.array([0.0, 1.0, 2.0]),
        bins=None,
        interval_learner=SimpleNamespace(),
        prediction_orchestrator=DummyPredictionOrchestrator(),
        reject_learner=object(),
        reject_threshold=np.array([0.4, 0.4]),
        reject_ncf="default",
        reject_ncf_w=0.5,
        reject_ncf_auto_selected=False,
        is_multiclass=lambda: False,
    )
    orchestrator = orch.RejectOrchestrator(expl)
    init_calls = []
    monkeypatch.setattr(
        orchestrator,
        "initialize_reject_learner",
        lambda *args, **kwargs: init_calls.append(kwargs.get("threshold")),
    )
    monkeypatch.setattr(
        orchestrator,
        "predict_reject_breakdown",
        lambda *args, **kwargs: {
            "rejected": np.array([False, False]),
            "error_rate": 0.0,
            "reject_rate": 0.0,
            "ambiguity_rate": 0.0,
            "novelty_rate": 0.0,
            "ambiguity": np.array([False, False]),
            "novelty": np.array([False, False]),
            "prediction_set_size": np.array([1, 1]),
            "prediction_set": np.ones((2, 2), dtype=bool),
            "epsilon": 0.05,
            "raw_total_examples": 2,
            "raw_reject_counts": {"rejected": 0},
            "error_rate_defined": True,
        },
    )
    with pytest.warns(UserWarning, match="threshold mismatch"):
        orchestrator.apply_policy(
            RejectPolicy.FLAG,
            np.array([[0.0], [1.0]]),
            threshold=0.4,
        )
    assert init_calls == [0.4]


def test_regression_apply_policy_reinitializes_when_threshold_comparison_falls_back(monkeypatch):
    from calibrated_explanations.core.reject.policy import RejectPolicy

    class Unarrayable:
        def __array__(self, dtype=None):  # pragma: no cover - invoked by numpy
            raise TypeError("cannot convert")

        def __eq__(self, other):
            return False

    class DummyPredictionOrchestrator:
        def predict(self, x, **kwargs):
            n = len(x)
            return np.zeros(n), np.zeros(n), np.ones(n), None

    expl = SimpleNamespace(
        mode="regression",
        x_cal=np.array([[0.0], [1.0]]),
        y_cal=np.array([0.0, 1.0]),
        bins=None,
        interval_learner=SimpleNamespace(),
        prediction_orchestrator=DummyPredictionOrchestrator(),
        reject_learner=object(),
        reject_threshold=Unarrayable(),
        reject_ncf="default",
        reject_ncf_w=0.5,
        reject_ncf_auto_selected=False,
        is_multiclass=lambda: False,
    )
    orchestrator = orch.RejectOrchestrator(expl)
    init_calls = []
    monkeypatch.setattr(
        orchestrator,
        "initialize_reject_learner",
        lambda *args, **kwargs: init_calls.append(kwargs.get("threshold")),
    )
    monkeypatch.setattr(
        orchestrator,
        "predict_reject_breakdown",
        lambda *args, **kwargs: {
            "rejected": np.array([False, False]),
            "error_rate": 0.0,
            "reject_rate": 0.0,
            "ambiguity_rate": 0.0,
            "novelty_rate": 0.0,
            "ambiguity": np.array([False, False]),
            "novelty": np.array([False, False]),
            "prediction_set_size": np.array([1, 1]),
            "prediction_set": np.ones((2, 2), dtype=bool),
            "epsilon": 0.05,
            "raw_total_examples": 2,
            "raw_reject_counts": {"rejected": 0},
            "error_rate_defined": True,
        },
    )
    with pytest.warns(UserWarning, match="threshold mismatch"):
        orchestrator.apply_policy(
            RejectPolicy.FLAG,
            np.array([[0.0], [1.0]]),
            threshold=0.4,
        )
    assert init_calls == [0.4]


# ---------------------------------------------------------------------------
# MT-4 — sliced RejectCalibratedExplanations recomputes aggregate rates
# ---------------------------------------------------------------------------


def test_sliced_reject_contract_preserves_original_batch_rates():
    """Sliced wrappers keep contract rates on original-batch semantics."""
    from dataclasses import dataclass

    from calibrated_explanations.explanations.reject import (
        RejectCalibratedExplanations,
        RejectPolicy,
    )

    @dataclass
    class FakeExpl:
        index: int

    class FakeBase:
        def __init__(self):
            self.explanations = [FakeExpl(i) for i in range(4)]
            self.x_test = np.zeros((4, 2))
            self.y_threshold = None
            self.bins = None
            self.calibrated_explainer = SimpleNamespace()

    # 4 instances: 0=rejected/ambiguous, 1=accepted, 2=rejected/novel, 3=accepted
    rejected_full = np.array([True, False, True, False])
    ambiguity_full = np.array([True, False, False, False])
    novelty_full = np.array([False, False, True, False])
    metadata_full = {
        "reject_rate": 0.5,
        "ambiguity_rate": 0.25,
        "novelty_rate": 0.25,
        "error_rate": 0.1,
        "error_rate_defined": True,
        "ambiguity_mask": ambiguity_full,
        "novelty_mask": novelty_full,
        "prediction_set_size": np.array([2, 1, 0, 1]),
    }

    base = FakeBase()
    rce = RejectCalibratedExplanations.from_collection(
        base, metadata_full, RejectPolicy.FLAG, rejected=rejected_full
    )

    # Slice to rejected-only subset via public __getitem__
    sliced = rce[[0, 2]]
    meta = sliced.metadata

    # Contract rates/counts remain original-batch values
    assert abs(meta["reject_rate"] - 0.5) < 1e-9
    assert meta["rejected_count"] == 2
    assert meta["accepted_count"] == 2
    # Payload stats are exposed separately for sliced objects
    assert abs(meta["payload_reject_rate"] - 1.0) < 1e-9
    assert meta["payload_rejected_count"] == 2


def test_reject_wrapper_raises_on_malformed_source_indices():
    """Malformed source_indices must fail fast to avoid silent index drift."""
    from dataclasses import dataclass

    from calibrated_explanations.explanations.reject import (
        RejectCalibratedExplanations,
        RejectPolicy,
    )
    from calibrated_explanations.utils.exceptions import DataShapeError

    @dataclass
    class FakeExpl:
        index: int

    class FakeBase:
        def __init__(self):
            self.explanations = [FakeExpl(0), FakeExpl(1)]
            self.x_test = np.zeros((2, 2))
            self.y_threshold = None
            self.bins = None
            self.calibrated_explainer = SimpleNamespace()

    with pytest.raises(DataShapeError, match="source_indices"):
        RejectCalibratedExplanations.from_collection(
            FakeBase(),
            {
                "source_indices": [2, 1],  # invalid ordering + out of range
                "original_count": 2,
                "ambiguity_mask": np.array([True, False, True, False]),
            },
            RejectPolicy.ONLY_ACCEPTED,
            rejected=np.array([True, False, True, False]),
        )


def test_reject_wrapper_derives_source_indices_when_missing():
    """Missing source_indices falls back deterministically for subset policies."""
    from dataclasses import dataclass

    from calibrated_explanations.explanations.reject import (
        RejectCalibratedExplanations,
        RejectPolicy,
    )

    @dataclass
    class FakeExpl:
        index: int

    class FakeBase:
        def __init__(self):
            self.explanations = [FakeExpl(1), FakeExpl(3)]
            self.x_test = np.zeros((2, 2))
            self.y_threshold = None
            self.bins = None
            self.calibrated_explainer = SimpleNamespace()

    with pytest.warns(UserWarning, match="missing source_indices"):
        wrapped = RejectCalibratedExplanations.from_collection(
            FakeBase(),
            {
                "ambiguity_mask": np.array([True, False, True, False]),
                "novelty_mask": np.array([False, False, False, False]),
                "prediction_set_size": np.array([2, 1, 2, 1]),
            },
            RejectPolicy.ONLY_ACCEPTED,
            rejected=np.array([True, False, True, False]),
        )
    assert wrapped.metadata["source_indices"] == [1, 3]
    assert len(wrapped.rejected) == len(wrapped.explanations) == 2


def test_apply_policy_result_schema_v2_returns_strict_artifacts(monkeypatch):
    """`result_schema='v2'` returns strict split decision/payload artifacts."""
    from calibrated_explanations.core.reject.policy import RejectPolicy
    from calibrated_explanations.explanations.reject import (
        RejectDecisionArtifact,
        RejectPayloadArtifact,
        RejectResultV2,
    )

    _, orchestrator = make_stub(monkeypatch, singletons=True)
    result = orchestrator.apply_policy(
        RejectPolicy.FLAG,
        np.array([[0.0], [1.0], [2.0]]),
        explain_fn=lambda arr, **_: arr,
        result_schema="v2",
    )
    assert isinstance(result, RejectResultV2)
    assert result.schema_version == "2.0"
    assert isinstance(result.decision, RejectDecisionArtifact)
    assert isinstance(result.payload, RejectPayloadArtifact)
    assert result.metadata["schema_version"] == "2.0"

    with pytest.warns(DeprecationWarning):
        legacy = result.to_legacy()
    assert legacy.policy is RejectPolicy.FLAG
    np.testing.assert_array_equal(legacy.rejected, result.decision.rejected)
    assert legacy.metadata["schema_version"] == "2.0"


def test_reject_result_v2_round_trip_from_legacy(monkeypatch):
    """Legacy envelopes can be upgraded to strict v2 and downgraded back."""
    from calibrated_explanations.core.reject.policy import RejectPolicy
    from calibrated_explanations.explanations.reject import RejectResultV2

    _, orchestrator = make_stub(monkeypatch, singletons=True)
    legacy = orchestrator.apply_policy(
        RejectPolicy.FLAG,
        np.array([[0.0], [1.0], [2.0]]),
        explain_fn=lambda arr, **_: arr,
    )
    upgraded = RejectResultV2.from_legacy(legacy)
    with pytest.warns(DeprecationWarning):
        downgraded = upgraded.to_legacy()
    np.testing.assert_array_equal(downgraded.rejected, legacy.rejected)
    assert downgraded.policy is legacy.policy
    assert downgraded.metadata["policy"] == legacy.metadata["policy"]


def test_apply_policy_skips_prediction_payload_when_explain_fn_is_present(monkeypatch):
    """Explanation reject paths should skip expensive prediction payload computation by default."""
    from calibrated_explanations.core.reject.policy import RejectPolicy

    _, orchestrator = make_stub(monkeypatch, singletons=True)
    calls = {"predict": 0}

    def _predict(x, **kwargs):
        calls["predict"] += 1
        return np.zeros(len(x)), np.zeros(len(x)), np.ones(len(x)), None

    orchestrator.explainer.prediction_orchestrator = SimpleNamespace(predict=_predict)
    result = orchestrator.apply_policy(
        RejectPolicy.FLAG,
        np.array([[0.0], [1.0], [2.0]]),
        explain_fn=lambda arr, **_: arr,
    )
    assert calls["predict"] == 0
    assert result.prediction is None


def test_apply_policy_prediction_payload_opt_in_for_explain_paths(monkeypatch):
    """Prediction payload computation can be re-enabled explicitly when needed."""
    from calibrated_explanations.core.reject.policy import RejectPolicy

    _, orchestrator = make_stub(monkeypatch, singletons=True)
    calls = {"predict": 0}

    def _predict(x, **kwargs):
        calls["predict"] += 1
        return np.zeros(len(x)), np.zeros(len(x)), np.ones(len(x)), None

    orchestrator.explainer.prediction_orchestrator = SimpleNamespace(predict=_predict)
    result = orchestrator.apply_policy(
        RejectPolicy.FLAG,
        np.array([[0.0], [1.0], [2.0]]),
        explain_fn=lambda arr, **_: arr,
        include_prediction_payload=True,
    )
    assert calls["predict"] == 1
    assert result.prediction is not None


def test_subset_policy_gates_full_prediction_set_payload_by_default(monkeypatch):
    """Subset policies omit heavy `prediction_set` metadata unless explicitly requested."""
    from calibrated_explanations.core.reject.policy import RejectPolicy

    _, orchestrator = make_stub(monkeypatch, singletons=True)
    orchestrator.predict_reject_breakdown = lambda *args, **kwargs: {
        "rejected": np.array([True, False, True, False]),
        "error_rate": 0.0,
        "reject_rate": 0.5,
        "ambiguity_rate": 0.5,
        "novelty_rate": 0.0,
        "ambiguity": np.array([True, False, True, False]),
        "novelty": np.array([False, False, False, False]),
        "prediction_set_size": np.array([2, 1, 2, 1]),
        "prediction_set": np.array([[1, 1], [1, 0], [1, 1], [1, 0]]),
        "epsilon": 0.05,
        "raw_total_examples": 4,
        "raw_reject_counts": {"rejected": 2},
        "error_rate_defined": True,
    }

    default_result = orchestrator.apply_policy(
        RejectPolicy.ONLY_REJECTED,
        np.array([[0.0], [1.0], [2.0], [3.0]]),
        explain_fn=lambda arr, **_: arr,
    )
    assert default_result.metadata["prediction_set"] is None

    full_result = orchestrator.apply_policy(
        RejectPolicy.ONLY_REJECTED,
        np.array([[0.0], [1.0], [2.0], [3.0]]),
        explain_fn=lambda arr, **_: arr,
        include_prediction_set=True,
    )
    assert full_result.metadata["prediction_set"] is not None
