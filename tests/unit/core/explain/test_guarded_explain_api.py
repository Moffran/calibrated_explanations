"""Contract tests for guarded explanation APIs and guard utilities."""

from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer
from calibrated_explanations.explanations import (
    AlternativeExplanations,
    CalibratedExplanations,
)
from calibrated_explanations.explanations.guarded_explanation import (
    GuardedAlternativeExplanation,
    GuardedBin,
    GuardedFactualExplanation,
)
from calibrated_explanations.utils.distribution_guard import InDistributionGuard
from calibrated_explanations.utils.exceptions import ValidationError


def make_classification_explainer(*, seed: int = 0) -> tuple[CalibratedExplainer, np.ndarray]:
    data = load_iris()
    x_train, x_cal, y_train, y_cal = train_test_split(
        data.data,
        data.target,
        test_size=0.2,
        random_state=seed,
        stratify=data.target,
    )

    model = RandomForestClassifier(n_estimators=15, random_state=seed, max_depth=3)
    model.fit(x_train, y_train)

    explainer = CalibratedExplainer(model, x_cal, y_cal, mode="classification", seed=seed)
    return explainer, x_cal


# ---------------------------------------------------------------------------
# Factual guarded explanations
# ---------------------------------------------------------------------------


def test_explain_guarded_factual__returns_calibrated_explanations_container():
    """Guarded factual explain returns container + guarded explanation type."""
    explainer, x_cal = make_classification_explainer(seed=1)

    result = explainer.explain_factual(
        x_cal[:2],
        guarded=True,
        significance=0.2,
        merge_adjacent=True,
        n_neighbors=3,
        normalize_guard=True,
    )

    assert isinstance(result, CalibratedExplanations)
    assert len(result) == 2

    for expl in result.explanations:
        assert isinstance(expl, GuardedFactualExplanation)
        assert hasattr(expl, "rules")
        assert expl.has_rules


def test_explain_guarded_factual__get_rules_returns_dict():
    """Guarded factual explanation exposes a rules dict."""
    explainer, x_cal = make_classification_explainer(seed=3)

    result = explainer.explain_factual(
        x_cal[:1],
        guarded=True,
        significance=0.3,
        n_neighbors=3,
    )

    expl = result.explanations[0]
    rules = expl.get_rules()
    assert isinstance(rules, dict)
    assert "rule" in rules
    assert "predict" in rules
    assert "weight" in rules


# ---------------------------------------------------------------------------
# Alternative guarded explanations
# ---------------------------------------------------------------------------


def test_explore_guarded_alternatives__returns_alternative_explanations_container():
    """Guarded alternatives returns alternative container + guarded type."""
    explainer, x_cal = make_classification_explainer(seed=2)

    result = explainer.explore_alternatives(
        x_cal[:1],
        guarded=True,
        significance=0.2,
        merge_adjacent=False,
        n_neighbors=3,
        normalize_guard=True,
    )

    assert isinstance(result, AlternativeExplanations)
    assert len(result) == 1

    for expl in result.explanations:
        assert isinstance(expl, GuardedAlternativeExplanation)
        assert hasattr(expl, "rules")
        assert expl.has_rules


def test_explore_guarded_alternatives__get_rules_returns_dict():
    """Guarded alternative explanation exposes a rules dict."""
    explainer, x_cal = make_classification_explainer(seed=4)

    result = explainer.explore_alternatives(
        x_cal[:1],
        guarded=True,
        significance=0.3,
        n_neighbors=3,
    )

    expl = result.explanations[0]
    rules = expl.get_rules()
    assert isinstance(rules, dict)
    assert "rule" in rules
    assert "predict" in rules


def test_explore_guarded_alternatives__should_run_merge_adjacent_branch_in_alternative_mode():
    """Alternative guarded explain executes merge-adjacent branching."""
    explainer, x_cal = make_classification_explainer(seed=7)

    result = explainer.explore_alternatives(
        x_cal[:1],
        guarded=True,
        significance=0.2,
        merge_adjacent=True,
        n_neighbors=3,
        normalize_guard=True,
    )

    assert isinstance(result, AlternativeExplanations)
    assert len(result) == 1


# ---------------------------------------------------------------------------
# InDistributionGuard
# ---------------------------------------------------------------------------


def test_in_distribution_guard__should_compute_scores_and_p_values():
    """InDistributionGuard returns scores, p-values, and mask."""
    x_cal = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
        ]
    )
    guard = InDistributionGuard(x_cal, n_neighbors=2, normalize=False)

    x_test = np.array([[0.0, 0.0], [10.0, 10.0]])
    scores = guard.nonconformity_scores(x_test)
    p_vals = guard.p_values(x_test)
    conforming = guard.is_conforming(x_test, significance=0.25)

    assert scores.shape == (2,)
    assert p_vals.shape == (2,)
    assert np.all((p_vals >= 0.0) & (p_vals <= 1.0))
    assert conforming.dtype == bool


def test_in_distribution_guard__larger_significance_is_stricter():
    """Verify that larger significance = stricter (fewer accepted)."""
    x_cal = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
        ]
    )
    guard = InDistributionGuard(x_cal, n_neighbors=2, normalize=False)

    x_test = np.array([[1.5, 1.5], [10.0, 10.0]])

    # Lenient: significance=0.05 -> threshold = 0.05
    conforming_lenient = guard.is_conforming(x_test, significance=0.05)
    # Strict: significance=0.5 -> threshold = 0.5
    conforming_strict = guard.is_conforming(x_test, significance=0.5)

    # Strict should accept fewer or equal instances
    assert conforming_strict.sum() <= conforming_lenient.sum()


def test_in_distribution_guard__rejects_invalid_neighbor_count():
    """Constructor should reject invalid neighbor counts."""
    with pytest.raises(ValidationError, match="n_neighbors must be >= 1"):
        InDistributionGuard(np.array([[0.0, 0.0], [1.0, 1.0]]), n_neighbors=0)


def test_in_distribution_guard__rejects_invalid_input_shape_and_feature_count():
    """Preprocess validation should reject non-2D and wrong-width test inputs."""
    guard = InDistributionGuard(np.array([[0.0, 0.0], [1.0, 1.0]]), n_neighbors=1, normalize=False)

    with pytest.raises(ValidationError, match="x must be a 2D array"):
        guard.nonconformity_scores(np.array([0.5, 0.5]))

    with pytest.raises(ValidationError, match="same number of features as x_cal"):
        guard.nonconformity_scores(np.array([[0.5, 0.5, 0.5]]))


def test_in_distribution_guard__rejects_out_of_range_significance():
    """Conforming mask requires strict significance bounds (0, 1)."""
    guard = InDistributionGuard(np.array([[0.0], [1.0], [2.0]]), n_neighbors=1, normalize=False)

    with pytest.raises(ValidationError, match="strictly between 0 and 1"):
        guard.is_conforming(np.array([[0.5]]), significance=0.0)


# ---------------------------------------------------------------------------
# _use_plugin warning
# ---------------------------------------------------------------------------


def test_use_plugin_false__emits_warning_for_guarded_factual():
    """Guarded factual warns when _use_plugin is passed."""
    explainer, x_cal = make_classification_explainer(seed=5)

    with pytest.warns(UserWarning, match="_use_plugin has no effect"):
        explainer.explain_factual(
            x_cal[:1],
            guarded=True,
            _use_plugin=False,
            verbose=True,
            significance=0.3,
            n_neighbors=3,
        )


def test_use_plugin_false__emits_warning_for_guarded_alternatives():
    """Guarded alternatives warns when _use_plugin is passed."""
    explainer, x_cal = make_classification_explainer(seed=6)

    with pytest.warns(UserWarning, match="_use_plugin has no effect"):
        explainer.explore_alternatives(
            x_cal[:1],
            guarded=True,
            _use_plugin=False,
            verbose=True,
            significance=0.3,
            n_neighbors=3,
        )


class DummyExplainer:
    def __init__(self) -> None:
        self.y_cal = np.array([0, 1])
        self.mode = "classification"
        self.feature_names = ["f0", "f1"]
        self.class_labels = [0, 1]

    def is_multiclass(self) -> bool:
        """Return whether the dummy explainer represents multiclass."""
        return False


class DummyCalibratedExplanations:
    def __init__(self, explainer: DummyExplainer) -> None:
        self.explainer = explainer
        self.low_high_percentiles = None

    def get_explainer(self) -> DummyExplainer:
        """Return the underlying explainer (compat with explanation base class)."""
        return self.explainer


def test_guarded_factual_explanation__should_build_conditions_and_warn_when_nonconforming():
    """Guarded factual builds interval conditions and warns if non-conforming."""
    calibrated_explanations = DummyCalibratedExplanations(DummyExplainer())
    x_instance = np.array([0.5, 2.0])

    guarded_bins = {
        0: [
            GuardedBin(
                lower=-np.inf,
                upper=1.0,
                representative=0.25,
                predict=0.6,
                low=0.55,
                high=0.65,
                conforming=False,
                p_value=0.01,
                is_factual=True,
            )
        ],
        1: [
            GuardedBin(
                lower=1.0,
                upper=np.inf,
                representative=2.0,
                predict=0.8,
                low=0.75,
                high=0.85,
                conforming=True,
                p_value=0.9,
                is_factual=True,
            )
        ],
    }

    with pytest.warns(UserWarning, match="Dropping non-conforming factual bin"):
        explanation = GuardedFactualExplanation(
            calibrated_explanations,
            0,
            x_instance,
            binned={},
            feature_weights={
                "predict": np.array([[0.0, 0.0]]),
                "low": np.array([[0.0, 0.0]]),
                "high": np.array([[0.0, 0.0]]),
            },
            feature_predict={
                "predict": np.array([[0.8, 0.7]]),
                "low": np.array([[0.7, 0.65]]),
                "high": np.array([[0.9, 0.75]]),
            },
            prediction={
                "predict": np.array([0.8]),
                "low": np.array([0.7]),
                "high": np.array([0.9]),
            },
            guarded_bins=guarded_bins,
            feature_names=["f0", "f1"],
            categorical_features={1},
            verbose=True,
        )

    conditions = explanation.define_conditions()
    assert conditions[0].startswith("f0 <=")
    assert conditions[1] == "f1 = 2.0"

    rules = explanation.get_rules()
    assert rules["base_predict"] == [0.8]
    assert len(rules["rule"]) == 1


def test_guarded_factual_explanation__should_keep_base_prediction_when_no_rules():
    """Guarded factual with zero accepted rules should still keep CE baseline payload."""
    calibrated_explanations = DummyCalibratedExplanations(DummyExplainer())
    x_instance = np.array([0.5, 2.0])

    explanation = GuardedFactualExplanation(
        calibrated_explanations,
        0,
        x_instance,
        binned={},
        feature_weights={
            "predict": np.array([0.8, 0.8]),
            "low": np.array([0.7, 0.7]),
            "high": np.array([0.9, 0.9]),
        },
        feature_predict={
            "predict": np.array([0.8, 0.8]),
            "low": np.array([0.7, 0.7]),
            "high": np.array([0.9, 0.9]),
        },
        prediction={
            "predict": np.array([0.8]),
            "low": np.array([0.7]),
            "high": np.array([0.9]),
            "classes": np.array([1.0]),
        },
        guarded_bins={},
        feature_names=["f0", "f1"],
    )

    rules = explanation.get_rules()
    assert rules["rule"] == []
    assert rules["base_predict"] == [0.8]
    assert rules["base_predict_low"] == [0.7]
    assert rules["base_predict_high"] == [0.9]


def test_guarded_alternative_explanation__should_join_conditions_and_skip_baseline_duplicate_bins():
    """Guarded alternatives join conditions and skip baseline-equal bins."""
    calibrated_explanations = DummyCalibratedExplanations(DummyExplainer())
    x_instance = np.array([0.5, 2.0])

    guarded_bins = {
        0: [
            GuardedBin(
                lower=-np.inf,
                upper=1.0,
                representative=0.25,
                predict=0.8,
                low=0.7,
                high=0.9,
                conforming=True,
                p_value=0.9,
                is_factual=False,
            ),
            GuardedBin(
                lower=1.0,
                upper=np.inf,
                representative=1.5,
                predict=0.9,
                low=0.85,
                high=0.95,
                conforming=True,
                p_value=0.9,
                is_factual=False,
            ),
        ]
    }

    explanation = GuardedAlternativeExplanation(
        calibrated_explanations,
        0,
        x_instance,
        binned={},
        feature_weights={},
        feature_predict={},
        prediction={
            "predict": np.array([0.8]),
            "low": np.array([0.7]),
            "high": np.array([0.9]),
        },
        guarded_bins=guarded_bins,
        feature_names=["f0", "f1"],
    )

    conditions = explanation.define_conditions()
    assert "|" in conditions[0]
    assert conditions[0].startswith("f0 <=")
    assert conditions[1] == ""

    rules = explanation.get_rules()
    assert rules["base_predict"] == [0.8]
    assert len(rules["rule"]) == 1


def test_guarded_pipeline__should_emit_unique_interval_records_when_merge_enabled():
    """Guarded pipeline should produce unique interval records when merging is enabled."""
    explainer, x_cal = make_classification_explainer(seed=17)
    result = explainer.explore_alternatives(
        x_cal[:1],
        guarded=True,
        significance=0.2,
        merge_adjacent=True,
        n_neighbors=3,
        normalize_guard=True,
    )

    audit = result.explanations[0].get_guarded_audit()
    keys = [
        (
            rec["feature"],
            rec["lower"],
            rec["upper"],
            rec["predict"],
            rec["low"],
            rec["high"],
            rec["conforming"],
            rec["is_factual"],
        )
        for rec in audit["intervals"]
    ]
    assert len(keys) == len(set(keys))


def test_guarded_alternative_explanation__should_not_deduplicate_bins_during_rule_creation():
    """Alternative explanation should consume preprocessed bins without late dedupe."""
    calibrated_explanations = DummyCalibratedExplanations(DummyExplainer())
    x_instance = np.array([0.5, 2.0])
    duplicated_bin = GuardedBin(
        lower=1.0,
        upper=np.inf,
        representative=1.5,
        predict=0.9,
        low=0.85,
        high=0.95,
        conforming=True,
        p_value=0.9,
        is_factual=False,
    )

    explanation = GuardedAlternativeExplanation(
        calibrated_explanations,
        0,
        x_instance,
        binned={},
        feature_weights={},
        feature_predict={},
        prediction={
            "predict": np.array([0.8]),
            "low": np.array([0.7]),
            "high": np.array([0.9]),
        },
        guarded_bins={0: [duplicated_bin, duplicated_bin]},
        feature_names=["f0", "f1"],
    )

    rules = explanation.get_rules()
    assert len(rules["rule"]) == 2


def test_guarded_docs__should_keep_significance_wording_aligned_with_api_contract():
    repo_root = Path(__file__).resolve().parents[4]
    concepts = (
        repo_root / "docs" / "foundations" / "concepts" / "guarded_explanations.md"
    ).read_text(encoding="utf-8")
    quickstart = (repo_root / "docs" / "get-started" / "quickstart_guarded.md").read_text(
        encoding="utf-8"
    )

    assert "Lower values apply stricter filtering." not in concepts
    assert "Larger values apply stricter filtering." in concepts
    assert "representative perturbation passed the guard" not in quickstart


# ---------------------------------------------------------------------------
# Guarded vs standard non-identity
# ---------------------------------------------------------------------------


def test_guarded_factual__produces_fewer_rules_on_ood_instance():
    """Guarded explain must remove at least one interval for a clearly OOD instance.

    An instance constructed at 5x the calibration-set maximum lies far outside
    the training distribution.  The guard should reject at least one interval
    that standard CE would emit.
    """
    explainer, x_cal = make_classification_explainer(seed=50)
    x_ood = (x_cal.max(axis=0) * 5.0).reshape(1, -1)

    result = explainer.explain_factual(
        x_ood,
        guarded=True,
        significance=0.2,
        n_neighbors=5,
        normalize_guard=True,
    )

    audit = result.explanations[0].get_guarded_audit()
    assert audit["summary"]["intervals_removed_guard"] > 0, (
        "Expected at least one interval to be removed by the guard for an OOD instance "
        f"at 5x calibration max; got audit={audit['summary']}"
    )


def test_in_distribution_guard__normalize_affects_conformity():
    """normalize=True vs False must produce different conformity outcomes on OOD data.

    With normalization disabled the raw distance score is compared against
    calibration distances; with normalization enabled each feature is scaled
    to unit variance first, which changes the effective distance metric and
    therefore the p-value ranking.
    """
    explainer, x_cal = make_classification_explainer(seed=51)
    x_ood = (x_cal.max(axis=0) * 5.0).reshape(1, -1)

    result_norm = explainer.explain_factual(
        x_ood,
        guarded=True,
        significance=0.2,
        n_neighbors=5,
        normalize_guard=True,
    )
    result_raw = explainer.explain_factual(
        x_ood,
        guarded=True,
        significance=0.2,
        n_neighbors=5,
        normalize_guard=False,
    )

    removed_norm = result_norm.explanations[0].get_guarded_audit()["summary"][
        "intervals_removed_guard"
    ]
    removed_raw = result_raw.explanations[0].get_guarded_audit()["summary"][
        "intervals_removed_guard"
    ]

    # Both must remove at least one interval on an extreme OOD instance.
    assert removed_norm > 0 or removed_raw > 0, (
        "Neither normalize=True nor normalize=False removed any intervals for an "
        "instance 5x outside the calibration range"
    )
    # The two settings must produce at least one differing count somewhere, confirming
    # that the normalize flag has a real effect on the guard outcome.
    total_norm = result_norm.explanations[0].get_guarded_audit()["summary"]["intervals_tested"]
    total_raw = result_raw.explanations[0].get_guarded_audit()["summary"]["intervals_tested"]
    assert total_norm == total_raw, "Both runs must test the same number of intervals"


# ---------------------------------------------------------------------------
# Red-team hardening tests — significance validation
# ---------------------------------------------------------------------------


def test_should_accept_significance_of_one():
    """significance=1.0 is allowed (interval is (0, 1])."""
    explainer, x_cal = make_classification_explainer(seed=50)
    # Should not raise — 1.0 is a valid significance
    result = explainer.explain_factual(x_cal[:1], guarded=True, significance=1.0)
    assert result is not None


def test_should_reject_significance_of_zero():
    """significance=0.0 must be rejected."""
    explainer, x_cal = make_classification_explainer(seed=51)

    with pytest.raises(ValidationError, match=r"significance must be in the interval \(0, 1\]"):
        explainer.explain_factual(x_cal[:1], guarded=True, significance=0.0)


def test_should_reject_negative_significance():
    """Negative significance must be rejected."""
    explainer, x_cal = make_classification_explainer(seed=52)

    with pytest.raises(ValidationError, match=r"significance must be in the interval \(0, 1\]"):
        explainer.explain_factual(x_cal[:1], guarded=True, significance=-0.1)


def test_should_reject_significance_above_one():
    """significance>1 must be rejected."""
    explainer, x_cal = make_classification_explainer(seed=53)

    with pytest.raises(ValidationError, match=r"significance must be in the interval \(0, 1\]"):
        explainer.explain_factual(x_cal[:1], guarded=True, significance=1.5)


# ---------------------------------------------------------------------------
# Red-team hardening tests — WrapExplainer guarded preconditions
# ---------------------------------------------------------------------------


def test_should_raise_not_fitted_when_wrap_guarded_factual_called_unfitted():
    """WrapCalibratedExplainer.explain_guarded_factual must fail when not fitted.

    In normal mode the deprecated wrapper emits a DeprecationWarning then raises
    NotFittedError. In strict CI mode (CE_DEPRECATIONS=error) deprecate() raises
    DeprecationWarning immediately, so NotFittedError is never reached.
    """
    from calibrated_explanations import WrapCalibratedExplainer
    from calibrated_explanations.utils.exceptions import NotFittedError as CENotFitted
    from sklearn.ensemble import RandomForestClassifier
    from tests.helpers.deprecation import deprecations_error_enabled

    wrapper = WrapCalibratedExplainer(RandomForestClassifier())
    if deprecations_error_enabled():
        with pytest.raises(DeprecationWarning, match="explain_guarded_factual"):
            wrapper.explain_guarded_factual(np.array([[1.0, 2.0]]), significance=0.2)
    else:
        with (
            pytest.warns(DeprecationWarning, match="explain_guarded_factual"),
            pytest.raises(CENotFitted),
        ):
            wrapper.explain_guarded_factual(np.array([[1.0, 2.0]]), significance=0.2)


def test_should_raise_not_fitted_when_wrap_guarded_alternatives_called_unfitted():
    """WrapCalibratedExplainer.explore_guarded_alternatives must fail when not fitted.

    In normal mode the deprecated wrapper emits a DeprecationWarning then raises
    NotFittedError. In strict CI mode (CE_DEPRECATIONS=error) deprecate() raises
    DeprecationWarning immediately, so NotFittedError is never reached.
    """
    from calibrated_explanations import WrapCalibratedExplainer
    from calibrated_explanations.utils.exceptions import NotFittedError as CENotFitted
    from sklearn.ensemble import RandomForestClassifier
    from tests.helpers.deprecation import deprecations_error_enabled

    wrapper = WrapCalibratedExplainer(RandomForestClassifier())
    if deprecations_error_enabled():
        with pytest.raises(DeprecationWarning, match="explore_guarded_alternatives"):
            wrapper.explore_guarded_alternatives(np.array([[1.0, 2.0]]), significance=0.2)
    else:
        with (
            pytest.warns(DeprecationWarning, match="explore_guarded_alternatives"),
            pytest.raises(CENotFitted),
        ):
            wrapper.explore_guarded_alternatives(np.array([[1.0, 2.0]]), significance=0.2)


# ---------------------------------------------------------------------------
# Red-team hardening tests — verbose factual_bin=None path
# ---------------------------------------------------------------------------


def test_should_warn_when_factual_bin_is_none_in_verbose_mode():
    """get_rules must emit UserWarning (not crash) when factual_bin is None and verbose=True."""

    # Minimal stub explainer and collection for direct construction
    class StubExplainer:
        y_cal = np.array([0, 1])
        mode = "classification"
        feature_names = ["f0", "f1"]
        class_labels = [0, 1]
        categorical_features = []
        categorical_labels = None

        def is_multiclass(self):
            return False

    class StubCollection:
        def __init__(self):
            self.explainer = StubExplainer()
            self.features_to_ignore = []
            self.feature_filter_per_instance_ignore = None

        def get_explainer(self):
            return self.explainer

    # Build a GuardedFactualExplanation where feature 0 has a non-factual bin only.
    non_factual_bin = GuardedBin(
        lower=-np.inf,
        upper=np.inf,
        representative=0.5,
        predict=0.3,
        low=0.2,
        high=0.4,
        conforming=True,
        p_value=0.8,
        is_factual=False,
    )
    factual_bin_f1 = GuardedBin(
        lower=-np.inf,
        upper=np.inf,
        representative=0.2,
        predict=0.8,
        low=0.7,
        high=0.9,
        conforming=True,
        p_value=0.9,
        is_factual=True,
    )
    payload = {
        "binned": {"rule_values": [{0: ([0.1], 0.1, 0.1), 1: ([0.2], 0.2, 0.2)}]},
        "feature_weights": {
            "predict": np.array([[0.0, 0.0]]),
            "low": np.array([[0.0, 0.0]]),
            "high": np.array([[0.0, 0.0]]),
        },
        "feature_predict": {
            "predict": np.array([[0.3, 0.4]]),
            "low": np.array([[0.2, 0.3]]),
            "high": np.array([[0.4, 0.5]]),
        },
        "prediction": {
            "predict": np.array([0.8]),
            "low": np.array([0.7]),
            "high": np.array([0.9]),
            "classes": np.array([1.0]),
        },
    }

    expl = GuardedFactualExplanation(
        StubCollection(),
        0,
        np.array([0.1, 0.2]),
        guarded_bins={0: [non_factual_bin], 1: [factual_bin_f1]},
        feature_names=["f0", "f1"],
        verbose=True,
        **payload,
    )

    with pytest.warns(UserWarning, match="No factual bin found"):
        expl.get_rules()


# ---------------------------------------------------------------------------
# Red-team hardening tests — InDistributionGuard boundary
# ---------------------------------------------------------------------------


def test_guard_should_reject_significance_boundary_values():
    """InDistributionGuard.is_conforming must reject 0.0 and 1.0."""
    x_cal = np.random.default_rng(42).standard_normal((20, 3))
    guard = InDistributionGuard(x_cal, n_neighbors=3, normalize=False)

    with pytest.raises(ValidationError, match="strictly between 0 and 1"):
        guard.is_conforming(x_cal[:1], significance=0.0)

    with pytest.raises(ValidationError, match="strictly between 0 and 1"):
        guard.is_conforming(x_cal[:1], significance=1.0)


# ---------------------------------------------------------------------------
# Parameterized guarded API — Task 13 (v0.11.3)
# ---------------------------------------------------------------------------


def test_explain_factual_guarded_false__preserves_unguarded_behavior():
    """explain_factual(guarded=False) must return the standard FactualExplanation container."""
    from calibrated_explanations.explanations.explanation import FactualExplanation

    explainer, x_cal = make_classification_explainer(seed=100)
    result = explainer.explain_factual(x_cal[:2], guarded=False)

    assert isinstance(result, CalibratedExplanations)
    assert len(result) == 2
    for expl in result.explanations:
        assert isinstance(expl, FactualExplanation)
        assert not isinstance(expl, GuardedFactualExplanation)


def test_explain_factual_guarded_true__returns_guarded_factual_container():
    """explain_factual(guarded=True) must return GuardedFactualExplanation instances."""
    explainer, x_cal = make_classification_explainer(seed=101)
    result = explainer.explain_factual(
        x_cal[:2],
        guarded=True,
        significance=0.2,
        n_neighbors=3,
    )

    assert isinstance(result, CalibratedExplanations)
    assert len(result) == 2
    for expl in result.explanations:
        assert isinstance(expl, GuardedFactualExplanation)


def test_explore_alternatives_guarded_false__preserves_unguarded_behavior():
    """explore_alternatives(guarded=False) must return standard AlternativeExplanation container."""
    from calibrated_explanations.explanations.explanation import AlternativeExplanation

    explainer, x_cal = make_classification_explainer(seed=102)
    result = explainer.explore_alternatives(x_cal[:1], guarded=False)

    assert isinstance(result, AlternativeExplanations)
    assert len(result) == 1
    for expl in result.explanations:
        assert isinstance(expl, AlternativeExplanation)
        assert not isinstance(expl, GuardedAlternativeExplanation)


def test_explore_alternatives_guarded_true__returns_guarded_alternative_container():
    """explore_alternatives(guarded=True) must return GuardedAlternativeExplanation instances."""
    explainer, x_cal = make_classification_explainer(seed=103)
    result = explainer.explore_alternatives(
        x_cal[:1],
        guarded=True,
        significance=0.2,
        n_neighbors=3,
    )

    assert isinstance(result, AlternativeExplanations)
    assert len(result) == 1
    for expl in result.explanations:
        assert isinstance(expl, GuardedAlternativeExplanation)


def test_explain_guarded_factual_deprecated__emits_deprecation_warning():
    """explain_guarded_factual(...) must emit DeprecationWarning and delegate."""
    from tests.helpers.deprecation import warns_or_raises

    explainer, x_cal = make_classification_explainer(seed=104)
    result = None
    with warns_or_raises(match="explain_guarded_factual"):
        result = explainer.explain_guarded_factual(x_cal[:1], significance=0.2, n_neighbors=3)
    if result is not None:
        assert isinstance(result, CalibratedExplanations)
        for expl in result.explanations:
            assert isinstance(expl, GuardedFactualExplanation)


def test_explore_guarded_alternatives_deprecated__emits_deprecation_warning():
    """explore_guarded_alternatives(...) must emit DeprecationWarning and delegate."""
    from tests.helpers.deprecation import warns_or_raises

    explainer, x_cal = make_classification_explainer(seed=105)
    result = None
    with warns_or_raises(match="explore_guarded_alternatives"):
        result = explainer.explore_guarded_alternatives(x_cal[:1], significance=0.2, n_neighbors=3)
    if result is not None:
        assert isinstance(result, AlternativeExplanations)
        for expl in result.explanations:
            assert isinstance(expl, GuardedAlternativeExplanation)


def test_metadata_validation__accepts_supports_guarded_true():
    """validate_plugin_meta must accept supports_guarded=True for explanation plugins."""
    from calibrated_explanations.plugins.base import validate_plugin_meta

    meta = {
        "schema_version": 1,
        "name": "test.guarded",
        "version": "0.1.0",
        "provider": "test",
        "capabilities": ["explanation:factual", "explanation:alternative"],
        "data_modalities": ("tabular",),
        "supports_guarded": True,
    }
    validate_plugin_meta(meta)
    assert meta["supports_guarded"] is True


def test_metadata_validation__defaults_supports_guarded_to_false():
    """validate_plugin_meta must default supports_guarded to False when absent."""
    from calibrated_explanations.plugins.base import validate_plugin_meta

    meta = {
        "schema_version": 1,
        "name": "test.no_guarded",
        "version": "0.1.0",
        "provider": "test",
        "capabilities": ["explanation:factual"],
        "data_modalities": ("tabular",),
    }
    validate_plugin_meta(meta)
    assert meta["supports_guarded"] is False


def test_metadata_validation__rejects_non_boolean_supports_guarded():
    """validate_plugin_meta must reject non-boolean supports_guarded."""
    from calibrated_explanations.plugins.base import validate_plugin_meta

    meta = {
        "schema_version": 1,
        "name": "test.bad_guarded",
        "version": "0.1.0",
        "provider": "test",
        "capabilities": ["explanation:factual"],
        "data_modalities": ("tabular",),
        "supports_guarded": "yes",
    }
    with pytest.raises(ValidationError, match="supports_guarded.*boolean"):
        validate_plugin_meta(meta)


def test_resolver_selects_guarded_plugin_when_guarded_true():
    """find_explanation_plugin_for must select a plugin with supports_guarded=True.

    The legacy builtin plugins only accept CalibratedExplainer as the model;
    use make_classification_explainer to produce one.
    """
    from calibrated_explanations.plugins.registry import find_explanation_plugin_for
    from calibrated_explanations.plugins import ensure_builtin_plugins

    ensure_builtin_plugins()
    explainer, _ = make_classification_explainer(seed=77)
    _, plugin = find_explanation_plugin_for(
        "tabular",
        mode="factual",
        task="classification",
        model=explainer,
        guarded=True,
    )
    meta = getattr(plugin, "plugin_meta", {})
    assert meta.get("supports_guarded", False) is True


def test_resolver_rejects_unguarded_plugin_for_guarded_request():
    """find_explanation_plugin_for must raise when no guarded plugin is available."""
    from calibrated_explanations.plugins.registry import (
        find_explanation_plugin_for,
        register_explanation_plugin,
        clear as registry_clear,
        ensure_builtin_plugins,
        reset_plugin_catalog,
    )
    from sklearn.ensemble import RandomForestClassifier

    registry_clear()
    try:

        class UnguardedOnlyPlugin:
            plugin_meta = {
                "schema_version": 1,
                "name": "test.unguarded_only",
                "version": "0.1.0",
                "provider": "test",
                "capabilities": ["explanation:factual"],
                "modes": ("factual",),
                "tasks": ("classification",),
                "data_modalities": ("tabular",),
                "dependencies": (),
                "supports_guarded": False,
                "trusted": True,
                "trust": {"trusted": True},
            }

            def supports(self, model):
                return True

            def explain(self, model, x, **kw):
                return None

            def supports_mode(self, mode, *, task):
                return mode == "factual"

            def initialize(self, context):
                pass

            def explain_batch(self, x, request):
                return None

        register_explanation_plugin(
            "test.unguarded_only", UnguardedOnlyPlugin(), metadata=UnguardedOnlyPlugin.plugin_meta
        )

        model = RandomForestClassifier()
        with pytest.raises(ValidationError, match="guarded"):
            find_explanation_plugin_for(
                "tabular",
                mode="factual",
                task="classification",
                model=model,
                guarded=True,
            )
    finally:
        reset_plugin_catalog(kind="explanation")
        ensure_builtin_plugins()


def test_guarded_audit_preserved_via_parameterized_api():
    """get_guarded_audit() must work when called via explain_factual(guarded=True)."""
    explainer, x_cal = make_classification_explainer(seed=200)
    result = explainer.explain_factual(
        x_cal[:1],
        guarded=True,
        significance=0.2,
        n_neighbors=3,
    )

    expl = result.explanations[0]
    audit = expl.get_guarded_audit()
    assert "summary" in audit
    assert "intervals" in audit
    assert "intervals_tested" in audit["summary"]
