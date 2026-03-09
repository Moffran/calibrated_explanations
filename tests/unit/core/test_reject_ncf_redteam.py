"""Red-team tests for reject NCF implementations.

These tests exercise the various NCF variants and the orchestration
paths that warn on low hinge-weight (`w`) for non-hinge NCFs. The goal
is to validate shapes, ranges, and warning semantics across NCF types.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from calibrated_explanations.core.reject import orchestrator as orch


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

    for ncf in ("hinge", "ensured", "entropy", "margin"):
        # Initialize with a non-trivial w for non-hinge
        orchestrator.initialize_reject_learner(ncf=ncf, w=0.5)
        breakdown = orchestrator.predict_reject_breakdown([[0], [1], [2]], confidence=0.95)

        # Basic sanity checks on returned structure
        assert isinstance(breakdown, dict)
        assert "rejected" in breakdown
        assert "prediction_set" in breakdown
        ps = breakdown["prediction_set"]
        assert ps.ndim == 2
        assert ps.shape[0] == 3


# ---------------------------------------------------------------------------
# Shared stub factory
# ---------------------------------------------------------------------------

def _make_stub(monkeypatch, *, all_empty: bool = False, singletons: bool = False):
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
        p_row = [0.0, 0.0]          # both fail → empty sets
    elif singletons:
        p_row = [0.9, 0.01]         # only class-0 passes → singleton {0}
    else:
        p_row = [0.9, 0.9]          # both pass → ambiguous sets (size 2)

    class DummyConformal:
        def fit(self, *args, **kwargs):
            return self

        def predict_p(self, alphas, **kwargs):
            n = np.atleast_2d(alphas).shape[0]
            return np.tile(np.array(p_row), (n, 1))

    monkeypatch.setattr(orch, "ConformalClassifier", lambda: DummyConformal())
    expl = ExplainerStub()
    orchestrator = orch.RejectOrchestrator(expl)
    orchestrator.initialize_reject_learner(ncf="hinge", w=1.0)
    return expl, orchestrator


# ---------------------------------------------------------------------------
# MT-3 — error_rate is always >= 0 and error_rate_defined reflects validity
# ---------------------------------------------------------------------------

def test_error_rate_nonnegative_all_empty(monkeypatch):
    """error_rate must be >= 0 even when all prediction sets are empty."""
    _, orchestrator = _make_stub(monkeypatch, all_empty=True)
    breakdown = orchestrator.predict_reject_breakdown([[0], [1], [2]], confidence=0.99)
    assert breakdown["error_rate"] >= 0.0


def test_error_rate_defined_false_when_no_singletons(monkeypatch):
    """error_rate_defined is False when there are no singleton prediction sets."""
    _, orchestrator = _make_stub(monkeypatch, all_empty=True)
    breakdown = orchestrator.predict_reject_breakdown([[0], [1], [2]], confidence=0.99)
    assert breakdown["error_rate_defined"] is False


def test_error_rate_defined_true_when_singletons_exist(monkeypatch):
    """error_rate_defined is True when at least one singleton prediction set exists."""
    # singletons=True → only class-0 passes threshold → every set is {0} (singleton)
    _, orchestrator = _make_stub(monkeypatch, singletons=True)
    breakdown = orchestrator.predict_reject_breakdown([[0], [1], [2]], confidence=0.95)
    assert breakdown["error_rate_defined"] is True


# ---------------------------------------------------------------------------
# MT-9 — auto-selected NCF is recorded on the explainer
# ---------------------------------------------------------------------------

def test_ncf_auto_selected_true_when_implicit(monkeypatch):
    """reject_ncf_auto_selected is True when ncf is not specified explicitly."""
    expl, orchestrator = _make_stub(monkeypatch)
    expl.reject_learner = None
    orchestrator.initialize_reject_learner()
    assert expl.reject_ncf_auto_selected is True


def test_ncf_auto_selected_false_when_explicit(monkeypatch):
    """reject_ncf_auto_selected is False when ncf is specified explicitly."""
    expl, orchestrator = _make_stub(monkeypatch)
    orchestrator.initialize_reject_learner(ncf="hinge")
    assert expl.reject_ncf_auto_selected is False


# ---------------------------------------------------------------------------
# MT-10 — w=0.0 raises ValidationError for non-hinge NCFs
# ---------------------------------------------------------------------------

def test_w_zero_raises_for_nohinge_ncf(monkeypatch):
    """w=0.0 with a non-hinge NCF must raise ValidationError."""
    from calibrated_explanations.utils.exceptions import ValidationError  # noqa: PLC0415

    _, orchestrator = _make_stub(monkeypatch)
    with pytest.raises(ValidationError, match="w=0.0"):
        orchestrator.initialize_reject_learner(ncf="entropy", w=0.0)


# ---------------------------------------------------------------------------
# MT-5 — plain RejectPolicy reuses existing NCF without reinit
# ---------------------------------------------------------------------------

def test_plain_policy_reuses_existing_ncf_no_reinit(monkeypatch):
    """Passing a plain RejectPolicy enum returns it unchanged; reject_ncf unchanged."""
    from calibrated_explanations.core.reject.orchestrator import resolve_policy_spec
    from calibrated_explanations.core.reject.policy import RejectPolicy

    expl, _ = _make_stub(monkeypatch)
    expl.reject_ncf = "entropy"
    expl.reject_ncf_w = 0.5

    result = resolve_policy_spec(RejectPolicy.FLAG, expl)
    assert result is RejectPolicy.FLAG
    assert expl.reject_ncf == "entropy"


# ---------------------------------------------------------------------------
# MT-6 — ONLY_REJECTED with 0 matches: explanation=None, matched_count=0
# ---------------------------------------------------------------------------

def test_only_rejected_empty_subset_returns_none_and_matched_count_zero(monkeypatch):
    """When no instances are rejected, explanation is None and matched_count==0."""
    _, orchestrator = _make_stub(monkeypatch, singletons=True)  # singleton sets → all accepted
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


# ---------------------------------------------------------------------------
# MT-4 — sliced RejectCalibratedExplanations recomputes aggregate rates
# ---------------------------------------------------------------------------

def test_sliced_reject_rates_recomputed():
    """After slicing, aggregate rates must reflect the sliced masks, not originals.

    Uses the public __getitem__ slicing API and the public .metadata property.
    """
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

    # Slice to first 2 instances via public __getitem__
    sliced = rce[0:2]
    meta = sliced.metadata

    # reject_rate for [True, False] = 0.5
    assert abs(meta["reject_rate"] - 0.5) < 1e-9
    # ambiguity_rate for [True, False] = 0.5
    assert abs(meta["ambiguity_rate"] - 0.5) < 1e-9
    # novelty_rate for [False, False] = 0.0
    assert abs(meta["novelty_rate"] - 0.0) < 1e-9
    # error_rate_defined must be False after slicing
    assert meta["error_rate_defined"] is False
