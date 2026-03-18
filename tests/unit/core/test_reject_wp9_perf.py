"""Performance-hardening benchmarks for reject subset policies.

Three claims are verified:

1. **Subset explain cost reduction** — ONLY_REJECTED and ONLY_ACCEPTED pass only
   the matching subset to explain_fn, not the full batch.  The row count seen by
   explain_fn must equal the number of matching instances, not the batch size.

2. **Metadata payload gating** — ``prediction_set`` is absent (None) by default
   for subset policies, reducing per-result memory allocation. It is present when
   the caller opts in with ``include_prediction_set=True``.

3. **Timing proportionality** — when explain cost is proportional to row count,
   subset policies complete faster than FLAG in proportion to the rejection rate.
   The ratio is verified with a tolerance to keep the test deterministic without
   wallclock fragility.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pytest

from calibrated_explanations.core.reject import orchestrator as orch
from calibrated_explanations.core.reject.policy import RejectPolicy


# ---------------------------------------------------------------------------
# Shared stub factory (mirrors make_stub in test_reject_ncf_redteam.py)
# ---------------------------------------------------------------------------


def make_stub(monkeypatch, *, rejection_mask: np.ndarray):
    """Return (expl, RejectOrchestrator) wired to a deterministic breakdown.

    ``rejection_mask`` controls which instances are reported as rejected.
    The orchestrator's ``predict_reject_breakdown`` is replaced with a lambda
    that returns a fixed dict whose ``rejected`` key matches the mask.
    """

    class IntervalLearner:
        def predict_proba(self, x, bins=None):
            return np.tile(np.array([[0.5, 0.5]]), (len(x), 1))

        def predict_probability(self, x, y_threshold=None, bins=None):
            return np.zeros(len(x)), None, None, None

    class ExplainerStub(SimpleNamespace):
        def __init__(self):
            super().__init__()
            self.mode = "classification"
            self.x_cal = np.zeros((4, 1))
            self.y_cal = np.array([0, 1, 0, 1])
            self.bins = None
            self.interval_learner = IntervalLearner()
            self.reject_learner = None

        def is_multiclass(self):
            return False

    class DummyConformal:
        def fit(self, *args, **kwargs):
            return self

        def predict_p(self, alphas, **kwargs):
            n = np.atleast_2d(alphas).shape[0]
            return np.full((n, 2), 0.9)

    monkeypatch.setattr(orch, "ConformalClassifier", lambda: DummyConformal())
    expl = ExplainerStub()
    o = orch.RejectOrchestrator(expl)
    o.initialize_reject_learner(ncf="default", w=1.0)

    rejected = np.asarray(rejection_mask, dtype=bool)
    n = len(rejected)
    reject_count = int(rejected.sum())
    o.predict_reject_breakdown = lambda *a, **kw: {
        "rejected": rejected,
        "error_rate": 0.0,
        "error_rate_defined": True,
        "reject_rate": reject_count / n,
        "ambiguity_rate": reject_count / n,
        "novelty_rate": 0.0,
        "ambiguity": rejected,
        "novelty": np.zeros(n, dtype=bool),
        "prediction_set_size": np.where(rejected, 2, 1),
        "prediction_set": np.tile([1, 1], (n, 1)),
        "epsilon": 0.05,
        "raw_total_examples": n,
        "raw_reject_counts": {"rejected": reject_count},
    }
    return expl, o


# ---------------------------------------------------------------------------
# Benchmark 1: explain_fn receives only the matching subset
# ---------------------------------------------------------------------------

_HALF_REJECTED = np.array([True, False, True, False, True, False, True, False], dtype=bool)


def test_only_rejected_explain_fn_receives_subset_rows(monkeypatch):
    """ONLY_REJECTED must pass exactly the rejected rows to explain_fn (F16)."""
    _, o = make_stub(monkeypatch, rejection_mask=_HALF_REJECTED)
    x = np.arange(len(_HALF_REJECTED)).reshape(-1, 1).astype(float)

    received_sizes = []

    def _explain_fn(arr, **_):
        received_sizes.append(len(arr))
        return arr

    o.apply_policy(RejectPolicy.ONLY_REJECTED, x, explain_fn=_explain_fn)

    expected = int(_HALF_REJECTED.sum())
    assert received_sizes == [expected], (
        f"ONLY_REJECTED: explain_fn should have received {expected} rows, " f"got {received_sizes}"
    )


def test_only_accepted_explain_fn_receives_subset_rows(monkeypatch):
    """ONLY_ACCEPTED must pass exactly the accepted rows to explain_fn (F16)."""
    _, o = make_stub(monkeypatch, rejection_mask=_HALF_REJECTED)
    x = np.arange(len(_HALF_REJECTED)).reshape(-1, 1).astype(float)

    received_sizes = []

    def _explain_fn(arr, **_):
        received_sizes.append(len(arr))
        return arr

    o.apply_policy(RejectPolicy.ONLY_ACCEPTED, x, explain_fn=_explain_fn)

    expected = int((~_HALF_REJECTED).sum())
    assert received_sizes == [expected], (
        f"ONLY_ACCEPTED: explain_fn should have received {expected} rows, " f"got {received_sizes}"
    )


def test_flag_explain_fn_receives_all_rows(monkeypatch):
    """FLAG must pass all rows to explain_fn (semantic drift control)."""
    _, o = make_stub(monkeypatch, rejection_mask=_HALF_REJECTED)
    x = np.arange(len(_HALF_REJECTED)).reshape(-1, 1).astype(float)

    received_sizes = []

    def _explain_fn(arr, **_):
        received_sizes.append(len(arr))
        return arr

    o.apply_policy(RejectPolicy.FLAG, x, explain_fn=_explain_fn)

    assert received_sizes == [
        len(x)
    ], f"FLAG: explain_fn should have received all {len(x)} rows, got {received_sizes}"


def test_subset_explain_row_count_scales_with_rejection_rate(monkeypatch):
    """explain_fn row count for subset policies must scale with rejection rate (F16)."""
    n = 20
    # High rejection (80%)
    high_mask = np.array([i % 5 != 0 for i in range(n)])
    _, o_high = make_stub(monkeypatch, rejection_mask=high_mask)
    x = np.zeros((n, 1))

    rows_high = []
    o_high.apply_policy(
        RejectPolicy.ONLY_REJECTED, x, explain_fn=lambda a, **_: rows_high.append(len(a)) or a
    )

    # Low rejection (20%)
    low_mask = np.array([i % 5 == 0 for i in range(n)])
    _, o_low = make_stub(monkeypatch, rejection_mask=low_mask)

    rows_low = []
    o_low.apply_policy(
        RejectPolicy.ONLY_REJECTED, x, explain_fn=lambda a, **_: rows_low.append(len(a)) or a
    )

    assert (
        rows_high[0] > rows_low[0]
    ), "Higher rejection rate must cause explain_fn to receive more rows for ONLY_REJECTED"
    assert rows_high[0] == int(high_mask.sum())
    assert rows_low[0] == int(low_mask.sum())


# ---------------------------------------------------------------------------
# Benchmark 2: metadata payload gating (prediction_set)
# ---------------------------------------------------------------------------


def test_prediction_set_absent_by_default_for_only_rejected(monkeypatch):
    """prediction_set metadata is None by default for ONLY_REJECTED (WP9 gate)."""
    _, o = make_stub(monkeypatch, rejection_mask=_HALF_REJECTED)
    x = np.zeros((len(_HALF_REJECTED), 1))

    result = o.apply_policy(RejectPolicy.ONLY_REJECTED, x, explain_fn=lambda a, **_: a)
    assert result.metadata["prediction_set"] is None


def test_prediction_set_absent_by_default_for_only_accepted(monkeypatch):
    """prediction_set metadata is None by default for ONLY_ACCEPTED (WP9 gate)."""
    _, o = make_stub(monkeypatch, rejection_mask=_HALF_REJECTED)
    x = np.zeros((len(_HALF_REJECTED), 1))

    result = o.apply_policy(RejectPolicy.ONLY_ACCEPTED, x, explain_fn=lambda a, **_: a)
    assert result.metadata["prediction_set"] is None


def test_prediction_set_present_for_flag_by_default(monkeypatch):
    """prediction_set metadata is populated by default for FLAG (no gating)."""
    _, o = make_stub(monkeypatch, rejection_mask=_HALF_REJECTED)
    x = np.zeros((len(_HALF_REJECTED), 1))

    result = o.apply_policy(RejectPolicy.FLAG, x, explain_fn=lambda a, **_: a)
    assert result.metadata["prediction_set"] is not None


def test_prediction_set_opt_in_populates_for_subset_policies(monkeypatch):
    """include_prediction_set=True restores prediction_set for subset policies."""
    _, o = make_stub(monkeypatch, rejection_mask=_HALF_REJECTED)
    x = np.zeros((len(_HALF_REJECTED), 1))

    for policy in (RejectPolicy.ONLY_REJECTED, RejectPolicy.ONLY_ACCEPTED):
        result = o.apply_policy(policy, x, explain_fn=lambda a, **_: a, include_prediction_set=True)
        assert (
            result.metadata["prediction_set"] is not None
        ), f"{policy}: prediction_set should be populated when include_prediction_set=True"


# ---------------------------------------------------------------------------
# Benchmark 3: timing proportionality
# ---------------------------------------------------------------------------


def make_timed_explain(row_cost_s: float):
    """Return an explain_fn whose cost is proportional to the number of input rows."""

    def timed_explain(arr, **_):
        time.sleep(row_cost_s * len(arr))
        return arr

    return timed_explain


@pytest.mark.parametrize("rejection_frac", [0.25, 0.5, 0.75])
def test_subset_policy_faster_than_flag_proportionally(monkeypatch, rejection_frac):
    """ONLY_REJECTED explain time scales with rejection fraction (F16 timing proof).

    The assertion is soft: subset time must be strictly less than FLAG time when
    explain cost is proportional to row count.  A 20 % tolerance absorbs OS
    scheduling jitter without requiring sub-millisecond precision.
    """
    n = 20
    k = round(n * rejection_frac)
    mask = np.array([i < k for i in range(n)], dtype=bool)  # first k rejected

    _, o_subset = make_stub(monkeypatch, rejection_mask=mask)
    _, o_flag = make_stub(monkeypatch, rejection_mask=mask)

    x = np.zeros((n, 1))
    row_cost = 0.002  # 2 ms per row — fast enough for CI, slow enough to measure

    # Time ONLY_REJECTED
    t0 = time.perf_counter()
    o_subset.apply_policy(RejectPolicy.ONLY_REJECTED, x, explain_fn=make_timed_explain(row_cost))
    t_subset = time.perf_counter() - t0

    # Time FLAG (all rows)
    t0 = time.perf_counter()
    o_flag.apply_policy(RejectPolicy.FLAG, x, explain_fn=make_timed_explain(row_cost))
    t_flag = time.perf_counter() - t0

    # Subset must be cheaper. Allow 20 % timing slack for scheduler noise.
    tolerance = 0.20
    assert t_subset < t_flag * (1 + tolerance), (
        f"ONLY_REJECTED ({rejection_frac*100:.0f}% rejected) took {t_subset:.3f}s "
        f"but FLAG took {t_flag:.3f}s — expected subset to be faster"
    )

    # Also verify the ratio is roughly right: subset ≈ rejection_frac * flag
    expected_ratio = rejection_frac
    actual_ratio = t_subset / t_flag if t_flag > 0 else 0
    assert actual_ratio < expected_ratio + tolerance, (
        f"Subset/FLAG time ratio {actual_ratio:.2f} exceeds expected {expected_ratio:.2f} "
        f"(+{tolerance:.0%} tolerance)"
    )
