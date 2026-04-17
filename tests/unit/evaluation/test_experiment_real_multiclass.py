"""Tests for the CE-first real multiclass experiment."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import numpy as np
from sklearn.metrics import brier_score_loss


MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "evaluation"
    / "multiclass"
    / "real-multiclass"
    / "experiment_real_multiclass.py"
)
MODULE_SPEC = importlib.util.spec_from_file_location("experiment_real_multiclass", MODULE_PATH)
experiment_real_multiclass = importlib.util.module_from_spec(MODULE_SPEC)
assert MODULE_SPEC.loader is not None
MODULE_SPEC.loader.exec_module(experiment_real_multiclass)


def test_should_not_import_venn_abers_in_ce_first_experiment():
    """The CE-first experiment must not import VennAbers directly."""
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "VennAbers" not in source


def test_should_not_import_matplotlib_at_module_import_time():
    """The experiment module should avoid hard matplotlib imports at import time."""
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    top_level_matplotlib_imports = [
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "matplotlib"
    ]
    assert not top_level_matplotlib_imports


def test_should_not_call_predict_proba_on_learner_or_model_symbols():
    """The experiment should predict only through wrapper public APIs."""
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    disallowed = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            owner = node.func.value
            disallowed_owners = {"learner", "model", "uncal_model", "va_model"}
            if isinstance(owner, ast.Name) and owner.id in disallowed_owners:
                if node.func.attr == "predict_proba":
                    disallowed.append(owner.id)
    assert not disallowed


def test_should_compute_per_class_brier_when_labels_are_non_contiguous():
    """Per-class Brier should follow class-label order, not numeric ranges."""
    y_true = np.array([0, 2, 0, 2])
    proba = np.array(
        [
            [0.8, 0.2],
            [0.1, 0.9],
            [0.7, 0.3],
            [0.2, 0.8],
        ]
    )
    class_labels = np.array([0, 2])
    expected = np.mean(
        [
            brier_score_loss((y_true == 0).astype(int), proba[:, 0]),
            brier_score_loss((y_true == 2).astype(int), proba[:, 1]),
        ]
    )
    score = experiment_real_multiclass.per_class_brier(y_true, proba, class_labels)
    assert np.isclose(score, expected)


def test_should_compute_multiclass_brier_when_labels_are_non_contiguous():
    """Multiclass Brier should score the full probability vector against one-hot targets."""
    y_true = np.array([0, 2])
    proba = np.array([[0.8, 0.2], [0.1, 0.9]])
    class_labels = np.array([0, 2])
    expected = ((0.8 - 1.0) ** 2 + (0.2 - 0.0) ** 2 + (0.1 - 0.0) ** 2 + (0.9 - 1.0) ** 2) / 2
    score = experiment_real_multiclass.multiclass_brier_score(y_true, proba, class_labels)
    assert np.isclose(score, expected)


def test_should_return_finite_ece_when_probabilities_hit_upper_bin_boundary():
    """ECE should remain finite when confidence values include exact 1.0 entries."""
    y_true_binary = np.array([1, 1, 0, 0])
    y_prob = np.array([1.0, 0.95, 0.05, 0.0])
    ece_value, fraction_of_positives, mean_predicted_value = (
        experiment_real_multiclass.expected_calibration_error(y_true_binary, y_prob)
    )
    assert np.isfinite(ece_value)
    assert len(fraction_of_positives) == len(mean_predicted_value)
    assert len(fraction_of_positives) > 0


def test_should_return_classwise_metric_rows_with_expected_schema():
    """Classwise diagnostics should expose one row per class with stable keys."""
    y_true = np.array([0, 2, 0, 2])
    proba = np.array(
        [
            [0.8, 0.2],
            [0.1, 0.9],
            [0.7, 0.3],
            [0.2, 0.8],
        ]
    )
    class_labels = np.array([0, 2])
    rows = experiment_real_multiclass.per_class_metric_rows(
        dataset_name="toy",
        method_name="CE",
        y_true=y_true,
        proba=proba,
        class_labels=class_labels,
        n_samples=len(y_true),
    )
    assert len(rows) == 2
    assert rows[0]["dataset"] == "toy"
    assert rows[0]["method"] == "CE"
    assert rows[0]["class_label"] == 0
    assert rows[1]["class_label"] == 2
    assert "class_ece" in rows[0]
    assert "class_brier" in rows[0]


def test_should_label_exactly_normalized_row_sums_as_no_issue():
    """Row-sum diagnostics should explicitly mark exact normalization as safe."""
    summary = experiment_real_multiclass.summarize_row_sum_quality(np.array([1.0, 1.0, 1.0]))
    assert summary["row_sum_quality"] == "exactly_normalized"
    assert summary["row_sum_issue_severity"] == "none"
    assert "No action needed" in summary["row_sum_suggestion"]


def test_should_label_large_row_sum_errors_as_high_severity():
    """Large deviation from one should produce a strong warning and repair suggestion."""
    summary = experiment_real_multiclass.summarize_row_sum_quality(np.array([1.10, 0.88, 1.07]))
    assert summary["row_sum_quality"] == "strongly_misnormalized"
    assert summary["row_sum_issue_severity"] == "high"
    assert "normalize rows" in summary["row_sum_suggestion"]


def test_should_count_wins_losses_and_ties_when_pairwise_vectors_are_compared():
    """Pairwise statistics should preserve win/loss/tie accounting for paired metrics."""
    left = np.array([0.10, 0.20, 0.30])
    right = np.array([0.20, 0.10, 0.30])
    stats = experiment_real_multiclass.compare_metric_vectors(left, right, lower_is_better=True)
    assert stats["wins"] == 1
    assert stats["losses"] == 1
    assert stats["ties"] == 1
    assert stats["n_valid_pairs"] == 3


# ---------------------------------------------------------------------------
# Red-team: sum-to-one invariant for multi_labels_enabled=True
# ---------------------------------------------------------------------------


def test_fold_runner_does_not_assert_row_sums_equal_one_for_ce_multi(monkeypatch):
    """run_fold_predictions has no guard against misnormalized CE_multi output.

    Red-team finding: a wrapper that returns proba rows summing to 1.2 passes
    through the fold runner without any error or quality check.  The experiment
    records row-sum quality only at the dataset level, *after* all folds are
    stacked, so per-fold misnormalization is never caught.
    """

    class MisnormalizedWrapper:
        """Wrapper that deliberately returns probabilities summing to 1.2."""

        def __init__(self, learner):
            self.learner = learner
            self.fitted = True
            self.calibrated = False
            self.multi_enabled = False

        def fit(self, _x_data, y_labels):
            """Fit stub — tracks class count from y_labels."""
            self.fitted = True
            self.n_classes = len(np.unique(y_labels))
            return self

        def calibrate(self, _x_data, y_labels, **kwargs):
            """Calibrate stub — stores multi_enabled flag."""
            self.calibrated = True
            self.multi_enabled = bool(kwargs.get("multi_labels_enabled", False))
            self.n_classes = len(np.unique(y_labels))
            return self

        def predict_proba(self, x_data, calibrated=True):
            """Return misnormalized proba (rows sum to 1.2 when multi_enabled)."""
            n_rows = len(x_data)
            if not calibrated:
                return np.full((n_rows, 3), 1.0 / 3)
            if self.multi_enabled:
                # Deliberately misnormalized: rows sum to 1.2, not 1.0
                return np.tile(np.array([[0.2, 0.4, 0.6]]), (n_rows, 1))
            return np.tile(np.array([[0.2, 0.3, 0.5]]), (n_rows, 1))

    class FakeLearner:
        """Minimal learner stub."""

        def __init__(self, random_state):
            self.random_state = random_state
            self.fitted = False

    monkeypatch.setattr(experiment_real_multiclass, "RandomForestClassifier", FakeLearner)
    monkeypatch.setattr(
        experiment_real_multiclass,
        "ensure_ce_first_wrapper",
        MisnormalizedWrapper,
    )

    x_prop = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    y_prop = np.array([0, 1, 2, 0, 1, 2])
    x_cal = np.array([[0.5], [1.5], [2.5], [3.5], [4.5], [5.5]])
    y_cal = np.array([0, 1, 2, 0, 1, 2])
    x_test = np.array([[9.0], [10.0]])

    result = experiment_real_multiclass.run_fold_predictions(
        x_prop=x_prop,
        y_prop=y_prop,
        x_cal=x_cal,
        y_cal=y_cal,
        x_test=x_test,
        split_seed=42,
    )

    ce_multi_row_sums = result["CE_multi"].sum(axis=1)
    # The fold runner completes successfully despite misnormalized output.
    # This asserts the GAP: there is no per-fold simplex check.
    assert result["CE_multi"].shape == (2, 3)
    assert not np.allclose(ce_multi_row_sums, 1.0), (
        "Expected misnormalized output to pass through the fold runner unchecked; "
        "if this assertion fails the fold runner now enforces the simplex constraint — "
        "which is a fix, not a failure."
    )


def test_ce_multi_row_sums_equal_ce_row_sums_when_predict_proba_unchanged(
    monkeypatch,
):
    """multi_labels_enabled=True in calibrate() does not change predict_proba output.

    Red-team finding: CalibratedExplainer.__init__ does not consume the
    multi_labels_enabled kwarg — it is silently absorbed into **kwargs.  As a
    result the VennAbers calibrator is configured identically for CE and CE_multi,
    and both wrappers return the same probability matrix.  The row-sum property
    therefore comes from the shared VennAbers normalisation, not from
    multi_labels_enabled.
    """

    class IdenticalWrapper:
        """Wrapper that returns the same proba regardless of multi_enabled."""

        def __init__(self, learner):
            self.learner = learner
            self.fitted = True
            self.calibrated = False
            self.multi_enabled = False

        def fit(self, _x_data, y_labels):
            """Fit stub — tracks class count from labels."""
            self.fitted = True
            self.n_classes = len(np.unique(y_labels))
            return self

        def calibrate(self, _x_data, y_labels, **kwargs):
            """Calibrate stub — records multi_enabled flag."""
            self.calibrated = True
            self.multi_enabled = bool(kwargs.get("multi_labels_enabled", False))
            self.n_classes = len(np.unique(y_labels))
            return self

        def predict_proba(self, x_data, calibrated=True):  # pylint: disable=unused-argument
            """Return identical proba regardless of multi_enabled — mirrors real CE."""
            n_rows = len(x_data)
            return np.tile(np.array([[0.2, 0.3, 0.5]]), (n_rows, 1))

    class FakeLearner:
        """Minimal learner stub."""

        def __init__(self, random_state):
            self.random_state = random_state
            self.fitted = False

    monkeypatch.setattr(experiment_real_multiclass, "RandomForestClassifier", FakeLearner)
    monkeypatch.setattr(
        experiment_real_multiclass,
        "ensure_ce_first_wrapper",
        IdenticalWrapper,
    )

    x_prop = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    y_prop = np.array([0, 1, 2, 0, 1, 2])
    x_cal = np.array([[0.5], [1.5], [2.5], [3.5], [4.5], [5.5]])
    y_cal = np.array([0, 1, 2, 0, 1, 2])
    x_test = np.array([[9.0], [10.0]])

    result = experiment_real_multiclass.run_fold_predictions(
        x_prop=x_prop,
        y_prop=y_prop,
        x_cal=x_cal,
        y_cal=y_cal,
        x_test=x_test,
        split_seed=3,
    )

    # CE_multi and CE produce the same probabilities — as in the real implementation.
    assert np.allclose(result["CE_multi"], result["CE"])
    # The fold runner flags this as non-distinct, but does not fail.
    assert result["metadata"]["ce_multi_prediction_distinct"] is False
    # Row sums are identical between CE and CE_multi (shared VennAbers normalisation).
    assert np.allclose(result["CE"].sum(axis=1), result["CE_multi"].sum(axis=1))


def test_summarize_row_sum_quality_near_normalized_boundary():
    """near_normalized requires BOTH mae <= 1e-3 AND max_abs_error <= 1e-2.

    Values just inside both bounds must be classified low-severity.
    Values where mae barely exceeds 1e-3 must fall through to mildly_misnormalized.
    """
    # Inside both bounds: classified as near_normalized
    near = np.array([1.001, 0.999, 1.0])  # mae = 0.000667, max = 0.001
    summary = experiment_real_multiclass.summarize_row_sum_quality(near)
    assert summary["row_sum_quality"] == "near_normalized"
    assert summary["row_sum_issue_severity"] == "low"

    # mae just over 1e-3: falls through to mildly_misnormalized
    mild = np.array([1.003, 0.997, 1.0])  # mae = 0.002, max = 0.003
    summary_mild = experiment_real_multiclass.summarize_row_sum_quality(mild)
    assert summary_mild["row_sum_quality"] == "mildly_misnormalized"
    assert summary_mild["row_sum_issue_severity"] == "medium"


def test_summarize_row_sum_quality_outlier_dominates_classification():
    """A single large-deviation row pushes quality to high-severity even if mae is small.

    Red-team: mae <= 1e-3 is NOT sufficient to guarantee low severity — if one
    instance has max_abs_error > 1e-2, the near_normalized gate fails and the
    batch is classified medium or high severity.  The experiment's row_sum_mean
    statistic alone cannot reveal this; per-instance max deviation matters.
    """
    # 999 perfect rows + 1 row with 5 % error: mae ≈ 5e-5 (good), max = 0.05 (borderline)
    row_sums = np.ones(1000)
    row_sums[0] = 1.05  # one outlier
    mae = float(np.mean(np.abs(row_sums - 1.0)))
    assert mae < 1e-3, "mae must be small so the near_normalized mae gate passes"
    # max_abs_error = 0.05 > 1e-2, so near_normalized gate fails
    summary = experiment_real_multiclass.summarize_row_sum_quality(row_sums)
    assert summary["row_sum_quality"] != "near_normalized", (
        "A single 5 % outlier should prevent near_normalized classification; "
        "mean-based statistics hide per-instance violations."
    )


def test_multiclass_brier_score_nonzero_for_perfect_argmax_with_unnormalized_rows():
    """multiclass_brier_score is sensitive to row-sum violations even when argmax is correct.

    Red-team: if the calibrator returns rows summing to 1.2 (pre-normalization),
    the full-vector Brier score is non-zero even for perfect top-1 accuracy.
    The experiment applies this metric to raw wrapper output; if that output is
    not simplex-valid the metric is biased.
    """
    class_labels = np.array([0, 1, 2])
    y_true = np.array([0, 1, 2])
    # Perfect argmax: each row puts all mass on the correct class, but unnormalized (sum=1.2)
    proba_unnorm = np.array([
        [1.2, 0.0, 0.0],
        [0.0, 1.2, 0.0],
        [0.0, 0.0, 1.2],
    ])
    score_unnorm = experiment_real_multiclass.multiclass_brier_score(
        y_true, proba_unnorm, class_labels
    )

    # Normalized version: same argmax, simplex-valid
    proba_norm = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    score_norm = experiment_real_multiclass.multiclass_brier_score(y_true, proba_norm, class_labels)

    assert score_norm == 0.0, "Perfect simplex-valid predictions must give zero Brier score."
    assert score_unnorm > 0.0, (
        "Unnormalized predictions (sum=1.2) must give non-zero Brier score even when "
        "argmax is correct; the metric is not invariant to row-sum violations."
    )


def test_multiclass_log_loss_artificially_low_when_correct_class_proba_exceeds_one():
    """multiclass_log_loss clips proba to [eps, 1.0], masking misnormalization.

    Red-team: when the correct-class probability exceeds 1.0 (unnormalized output),
    the clip in multiclass_log_loss sets it to 1.0 and log(1.0)=0, making the
    metric appear perfect.  This means log_loss cannot distinguish a properly
    normalized P(correct)=1.0 from an un-normalised P(correct)=1.5 — a silent
    bias that makes CE_multi look better than it is when rows are inflated.
    """
    class_labels = np.array([0, 1])
    y_true = np.array([0, 1])

    # Simplex-valid: log_loss should be finite and positive
    proba_valid = np.array([[0.9, 0.1], [0.1, 0.9]])
    loss_valid = experiment_real_multiclass.multiclass_log_loss(y_true, proba_valid, class_labels)

    # Unnormalized: correct-class proba > 1.0 — clipped to 1.0 → log_loss = 0
    proba_inflated = np.array([[1.5, 0.1], [0.1, 1.5]])
    loss_inflated = experiment_real_multiclass.multiclass_log_loss(
        y_true, proba_inflated, class_labels
    )

    assert loss_inflated == 0.0, (
        "Clipping proba to 1.0 when proba > 1 gives log(1)=0, so log_loss appears perfect."
    )
    assert loss_valid > loss_inflated, (
        "Valid probabilities < 1 produce higher log_loss than clipped proba=1.0; "
        "the metric counter-intuitively rewards misnormalization."
    )


# ---------------------------------------------------------------------------
# Deep red-team: Finding 2 at depth — two compounded bugs
#
# Bug A (experiment): multi_labels_enabled=True is passed to calibrate() when
#   it is an explain-time-only flag.  CalibratedExplainer.__init__ silently
#   absorbs it into **kwargs and never reads it.
#
# Bug B (library): calibrate() accepts the kwarg without emitting any warning,
#   so callers receive no feedback that the flag has no effect.
#
# Consequence: CE and CE_multi share the same underlying learner AND an
#   identically configured CalibratedExplainer, so predict_proba() returns
#   the same matrix for both — confirmed on real data below.
# ---------------------------------------------------------------------------


def make_three_class_data(n_per_class=40, n_features=4, seed=0):
    """Return a synthetic 3-class dataset with classes interleaved row-by-row.

    Interleaving ensures that any contiguous slice of the returned arrays
    contains all three classes, making naive train/cal/test splits safe.
    """
    rng = np.random.RandomState(seed)
    n = n_per_class
    x0 = rng.randn(n, n_features) + np.array([3, 0, 0, 0])
    x1 = rng.randn(n, n_features) + np.array([-3, 0, 0, 0])
    x2 = rng.randn(n, n_features) + np.array([0, 3, 0, 0])
    # Interleave: row 0 = class 0, row 1 = class 1, row 2 = class 2, row 3 = class 0, …
    x_out = np.empty((3 * n, n_features))
    x_out[0::3] = x0
    x_out[1::3] = x1
    x_out[2::3] = x2
    y_out = np.empty(3 * n, dtype=int)
    y_out[0::3] = 0
    y_out[1::3] = 1
    y_out[2::3] = 2
    return x_out, y_out


def test_calibrate_with_multi_labels_enabled_emits_no_warning():
    """calibrate(multi_labels_enabled=True) is silently accepted — library Bug B.

    The library should warn that multi_labels_enabled has no effect in
    calibrate(), but it does not.  This test documents the current behaviour
    (0 warnings) as evidence of the silent-failure gap.  When Bug B is fixed,
    this test must be updated to expect a UserWarning.
    """
    import warnings  # pylint: disable=import-outside-toplevel

    from sklearn.ensemble import RandomForestClassifier  # pylint: disable=import-outside-toplevel
    from calibrated_explanations import WrapCalibratedExplainer  # pylint: disable=import-outside-toplevel

    x_all, y_all = make_three_class_data()
    # Interleaved layout: X[:60] = 20 per class, X[60:90] = 10 per class
    x_prop, y_prop = x_all[:60], y_all[:60]
    x_cal, y_cal = x_all[60:90], y_all[60:90]

    wrapper = WrapCalibratedExplainer(RandomForestClassifier(n_estimators=10, random_state=0))
    wrapper.fit(x_prop, y_prop)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        wrapper.calibrate(x_cal, y_cal, mode="classification", multi_labels_enabled=True)

    param_warnings = [w for w in caught if "multi_labels_enabled" in str(w.message)]
    assert len(param_warnings) == 0, (
        "Library Bug B confirmed: calibrate() emits no warning about "
        "multi_labels_enabled having no effect on calibration or predict_proba."
    )


def test_predict_proba_identical_for_ce_and_ce_multi_on_real_three_class_data():
    """predict_proba() is numerically identical for CE and CE_multi — Bugs A+B combined.

    Since multi_labels_enabled=True is silently ignored by calibrate() (Bug A),
    CE and CE_multi share the same underlying learner and an identically
    configured CalibratedExplainer.  The VennAbers calibrators fitted on the
    same data with the same learner are deterministic, so predict_proba()
    must return the same matrix.
    """
    from sklearn.ensemble import RandomForestClassifier  # pylint: disable=import-outside-toplevel
    from calibrated_explanations import WrapCalibratedExplainer  # pylint: disable=import-outside-toplevel
    from calibrated_explanations.ce_agent_utils import (  # pylint: disable=import-outside-toplevel
        ensure_ce_first_wrapper,
    )

    x_all, y_all = make_three_class_data()
    # Interleaved layout: X[:60] = 20 per class, X[60:90] = 10 per class, X[90:] = 10 per class
    x_prop, y_prop = x_all[:60], y_all[:60]
    x_cal, y_cal = x_all[60:90], y_all[60:90]
    x_test = x_all[90:]

    # Replicate exactly the experiment's _build_shared_wrappers pattern
    learner = RandomForestClassifier(n_estimators=20, random_state=7)
    fit_wrapper = ensure_ce_first_wrapper(learner)
    fit_wrapper.fit(x_prop, y_prop)
    shared_learner = fit_wrapper.learner

    ce_wrapper = ensure_ce_first_wrapper(shared_learner)
    ce_multi_wrapper = ensure_ce_first_wrapper(shared_learner)

    ce_wrapper.calibrate(x_cal, y_cal, mode="classification")
    ce_multi_wrapper.calibrate(x_cal, y_cal, mode="classification", multi_labels_enabled=True)

    ce_proba = np.asarray(ce_wrapper.predict_proba(x_test), dtype=float)
    ce_multi_proba = np.asarray(ce_multi_wrapper.predict_proba(x_test), dtype=float)

    assert np.allclose(ce_proba, ce_multi_proba), (
        "Bug A confirmed on real data: predict_proba() is identical for CE and "
        "CE_multi because multi_labels_enabled=True in calibrate() has no effect. "
        "The experiment's CE_multi arm does not demonstrate any distinct behaviour."
    )

    # Both satisfy the simplex constraint — but this comes from VennAbers normalization
    # (venn_abers.py lines 236-242), not from multi_labels_enabled.
    row_sums = ce_multi_proba.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-12), (
        "Row sums are 1.0 for CE_multi — but this is the VennAbers FIXME normalization, "
        "not a consequence of multi_labels_enabled=True."
    )


def test_correct_usage_explain_factual_with_multi_labels_enabled_yields_per_class_probabilities():
    """The correct usage of multi_labels_enabled is with explain_factual(), not calibrate().

    When explain_factual(x_test, multi_labels_enabled=True) is called, each
    instance gets one FactualExplanation per class stored in a
    MultiClassCalibratedExplanations object.  The per-class probability for
    instance i and class c is expl.explanations[i][c].prediction['predict'].

    These values come from the same VennAbers-normalised probability matrix as
    predict_proba(), so:
      (a) they sum to 1.0 per instance, and
      (b) they are identical to predict_proba()[:, c] for each class c.

    This test documents the correct pattern and proves that the sum-to-1
    property belongs to the VennAbers normalisation — it is equally present in
    the standard CE predict_proba() path and is not a feature of
    multi_labels_enabled.
    """
    from sklearn.ensemble import RandomForestClassifier  # pylint: disable=import-outside-toplevel
    from calibrated_explanations import WrapCalibratedExplainer  # pylint: disable=import-outside-toplevel

    # make_three_class_data interleaves classes, so X[:60] has 20 per class,
    # X[60:90] has 10 per class, X[90:105] has 5 per class.
    x_all, y_all = make_three_class_data()
    x_prop, y_prop = x_all[:60], y_all[:60]
    x_cal, y_cal = x_all[60:90], y_all[60:90]
    x_test = x_all[90:105]  # 15 instances, 5 per class

    assert len(np.unique(y_cal)) == 3, "calibration set must contain all 3 classes"

    wrapper = WrapCalibratedExplainer(RandomForestClassifier(n_estimators=20, random_state=0))
    wrapper.fit(x_prop, y_prop)
    # NOTE: multi_labels_enabled belongs here — in explain_factual(), not in calibrate()
    wrapper.calibrate(x_cal, y_cal, mode="classification")

    mc_expl = wrapper.explain_factual(x_test, multi_labels_enabled=True)

    n_classes = 3
    # (a) per-class probabilities for each instance must sum to 1.0
    for inst_idx in range(len(x_test)):
        per_class_proba = np.array([
            mc_expl.explanations[inst_idx][cls].prediction["predict"]
            for cls in range(n_classes)
        ])
        row_sum = per_class_proba.sum()
        assert np.isclose(row_sum, 1.0, atol=1e-10), (
            f"Instance {inst_idx}: per-class proba from explain_factual() sums to "
            f"{row_sum:.8f}, not 1.0.  The sum-to-1 property comes from VennAbers "
            "normalisation and is equally present in standard predict_proba()."
        )

    # (b) per-class probabilities are identical to predict_proba() values
    full_proba = np.asarray(wrapper.predict_proba(x_test), dtype=float)
    for inst_idx in range(len(x_test)):
        for cls in range(n_classes):
            expl_p = mc_expl.explanations[inst_idx][cls].prediction["predict"]
            proba_p = full_proba[inst_idx, cls]
            assert np.isclose(expl_p, proba_p, atol=1e-10), (
                f"Instance {inst_idx}, class {cls}: explain_factual() gives "
                f"{expl_p:.8f} but predict_proba() gives {proba_p:.8f}. "
                "Both derive from the same VennAbers-normalised matrix."
            )


def test_should_fit_one_learner_and_share_it_between_ce_wrappers(monkeypatch):
    """Fold runner should create two wrappers that share one fitted learner instance."""

    class FakeLearner:
        """Minimal learner carrying shared state."""

        def __init__(self, random_state):
            self.random_state = random_state
            self.fitted = False

    class FakeWrapper:
        """Simple wrapper stub that mimics the CE wrapper surface."""

        calibration_calls = []

        def __init__(self, learner):
            self.learner = learner
            self.fitted = bool(getattr(learner, "fitted", False))
            self.calibrated = False
            self.multi_enabled = False

        def fit(self, x, y):
            self.learner.fitted = True
            self.fitted = True
            self.n_classes = len(np.unique(y))
            return self

        def calibrate(self, x, y, **kwargs):
            self.calibrated = True
            self.fitted = True
            self.multi_enabled = bool(kwargs.get("multi_labels_enabled", False))
            self.__class__.calibration_calls.append(
                (self.learner, self.multi_enabled, kwargs.get("mode"))
            )
            self.n_classes = len(np.unique(y))
            return self

        def predict_proba(self, x, calibrated=True):
            n_rows = len(x)
            n_classes = getattr(self, "n_classes", 3)
            if not calibrated:
                return np.full((n_rows, n_classes), 1.0 / n_classes)
            if self.multi_enabled:
                return np.tile(np.array([[0.1, 0.3, 0.6]]), (n_rows, 1))
            return np.tile(np.array([[0.2, 0.3, 0.5]]), (n_rows, 1))

    monkeypatch.setattr(experiment_real_multiclass, "RandomForestClassifier", FakeLearner)
    monkeypatch.setattr(experiment_real_multiclass, "ensure_ce_first_wrapper", FakeWrapper)
    FakeWrapper.calibration_calls.clear()

    x_prop = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    y_prop = np.array([0, 1, 2, 0, 1, 2])
    x_cal = np.array([[0.5], [1.5], [2.5], [3.5], [4.5], [5.5]])
    y_cal = np.array([0, 1, 2, 0, 1, 2])
    x_test = np.array([[9.0], [10.0]])

    result = experiment_real_multiclass.run_fold_predictions(
        x_prop=x_prop,
        y_prop=y_prop,
        x_cal=x_cal,
        y_cal=y_cal,
        x_test=x_test,
        split_seed=7,
    )

    assert result["Uncal"].shape == (2, 3)
    assert result["CE"].shape == (2, 3)
    assert result["CE_multi"].shape == (2, 3)
    assert result["metadata"]["shared_learner_verified"] is True
    assert result["metadata"]["ce_multi_prediction_distinct"] is True

    calibration_calls = FakeWrapper.calibration_calls
    assert len(calibration_calls) == 2
    assert calibration_calls[0][0] is calibration_calls[1][0]
    assert calibration_calls[0][1] is False
    assert calibration_calls[1][1] is True
    assert calibration_calls[0][2] == "classification"
    assert calibration_calls[1][2] == "classification"
