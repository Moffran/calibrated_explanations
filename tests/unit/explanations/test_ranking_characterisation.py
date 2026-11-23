"""Characterisation tests for ADR-042 canonical rule and feature ranking (v1.1.0).

v1.0.2 pinned the pre-ADR-042 behaviour here. #204 (v1.1.0 T9) implemented the
contract in ``development/adrs/ADR-042-canonical-rule-and-feature-ranking.md``,
so every test whose behaviour the ADR changes was rewritten in place, keeping
the v1.0.x fixtures. The ADR's Appendix A maps each case to its decision.
"""

from __future__ import annotations

import numpy as np
import pytest

from calibrated_explanations.explanations import AlternativeExplanation, FactualExplanation
from calibrated_explanations.explanations import explanation as explanation_module
from calibrated_explanations.utils.exceptions import ValidationError
from calibrated_explanations.viz import build_regression_bars_spec


class _ExplainerStub:
    def __init__(self, mode="regression"):
        self.y_cal = np.array([0.1, 0.9])
        self.mode = mode
        self.class_labels = None
        self.feature_names = ["f0", "f1", "f2", "f3"]
        self.categorical_features: list[int] = []
        self.categorical_labels = None
        self.sample_percentiles = [25, 75]
        self.num_features = 2
        self.x_cal = np.array([[0.1, 0.3], [0.2, 0.4]])

    def is_multiclass(self):
        return False


class _ContainerStub:
    def __init__(self, mode="regression"):
        self.low_high_percentiles = (5, 95)
        self.explainer = _ExplainerStub(mode)
        self.features_to_ignore = []
        self.explanations = []

    def get_explainer(self):
        return self.explainer


class _RankingFactual(FactualExplanation):
    """FactualExplanation whose rule table is injected directly."""

    def _check_preconditions(self):
        return None

    def get_rules(self):
        return getattr(self, "injected_rules", {"rule": []})


class _RankingAlternative(AlternativeExplanation):
    """AlternativeExplanation whose rule table is injected directly."""

    def _check_preconditions(self):
        return None

    def get_rules(self):
        return getattr(self, "injected_rules", {"rule": []})


def _build(cls, mode="regression"):
    rule_values = np.empty((2, 2), dtype=object)
    for i in range(2):
        for j in range(2):
            rule_values[i, j] = np.array([[0.1]])
    pair = np.array([[0.1, 0.2], [0.3, 0.4]])
    return cls(
        _ContainerStub(mode),
        index=1,
        x=pair,
        binned={"predict": pair, "low": pair, "high": pair, "rule_values": rule_values},
        feature_weights={"predict": pair, "low": pair, "high": pair},
        feature_predict={"predict": pair, "low": pair, "high": pair},
        prediction={
            "predict": np.array([0.2, 0.6]),
            "low": np.array([0.1, 0.5]),
            "high": np.array([0.4, 0.9]),
            "classes": np.array([1, 1]),
        },
        y_threshold=None,
        instance_bin=0,
    )


def _make_factual(weights, weight_low, weight_high, texts):
    explanation = _build(_RankingFactual)
    n = len(weights)
    explanation.injected_rules = {
        "rule": list(texts),
        "feature": list(range(n)),
        "value": ["v"] * n,
        "weight": list(weights),
        "weight_low": list(weight_low),
        "weight_high": list(weight_high),
        "predict": [0.5] * n,
        "predict_low": [0.4] * n,
        "predict_high": [0.6] * n,
        "base_predict": [0.5],
    }
    return explanation


def _make_alternative():
    """Classification alternative with base prediction 0.6 and three contrasting rules.

    r0 moves far from the base (0.6 -> 0.2) with a wide interval; r1 moves little
    (0.75) with a very narrow interval; r2 moves moderately (0.8) with a medium one.
    """
    explanation = _build(_RankingAlternative, mode="classification")
    base = 0.6
    predict, low, high = [0.2, 0.75, 0.8], [0.05, 0.74, 0.7], [0.45, 0.76, 0.9]
    explanation.injected_rules = {
        "rule": ["r0", "r1", "r2"],
        "feature": [0, 1, 2],
        "value": ["v"] * 3,
        "sampled_values": [0] * 3,
        "feature_value": [0] * 3,
        "weight": [p - base for p in predict],
        "weight_low": [lo - base for lo in low],
        "weight_high": [hi - base for hi in high],
        "predict": predict,
        "predict_low": low,
        "predict_high": high,
        "base_predict": [base],
        "base_predict_low": [0.5],
        "base_predict_high": [0.9],
        "is_conjunctive": [False] * 3,
        "classes": 1,
    }
    return explanation


@pytest.fixture
def recorded_alternative_plot(monkeypatch):
    """Record the rule order ``AlternativeExplanation.plot`` hands to the renderer."""
    calls = []

    def _record(explanation, *_args, **_kwargs):
        features_to_plot = _args[3]
        texts = explanation.get_rules()["rule"]
        calls.append([texts[i] for i in features_to_plot])

    monkeypatch.setattr(explanation_module, "plot_alternative", _record)
    return calls


def _repr_rule_order(explanation):
    """Return rule texts in the order ``FactualExplanation.__repr__`` prints them."""
    texts = explanation.get_rules()["rule"]
    body = repr(explanation).splitlines()[3:]
    return [next(t for t in texts if f": {t}" in line) for line in body]


@pytest.fixture
def ranker():
    explanation = _make_factual([0.1], [0.0], [0.2], ["r0"])
    return explanation.rank_features


# --- rank_features (explanation.py) ------------------------------------------------


def test_rank_features_returns_ascending_with_strongest_last(ranker):
    ranked = ranker(np.array([0.4, -0.1, 0.2]), width=np.array([0.3, 0.5, 0.1]))

    assert list(ranked) == [1, 2, 0]


def test_rank_features_ranks_by_magnitude_ignoring_sign(ranker):
    ranked = ranker(np.array([-0.5, 0.3]), width=np.array([0.1, 0.1]))

    assert list(ranked)[-1] == 0


def test_rank_features_breaks_magnitude_ties_in_favour_of_narrower_interval(ranker):
    ranked = ranker(np.array([0.2, -0.2]), width=np.array([0.3, 0.1]))

    assert list(ranked)[-1] == 1  # ADR-042 D4/D5: the narrower (more certain) rule wins


def test_rank_features_full_tie_ranks_lowest_index_strongest(ranker):
    ranked = ranker(np.array([0.2, 0.2, 0.2]), width=np.array([0.1, 0.1, 0.1]))

    assert list(ranked) == [2, 1, 0]  # D5: native index ascending, returned strongest last


def test_rank_features_ranks_nan_weight_last(ranker):
    ranked = ranker(np.array([np.nan, 0.1, -0.05]), width=np.array([0.1, 0.1, 0.1]))

    assert list(ranked) == [0, 2, 1]  # D6: NaN is weakest, so first in ascending output


def test_rank_features_collapses_infinities_then_prefers_narrower(ranker):
    ranked = ranker(np.array([0.1, -np.inf, np.inf]), width=np.array([0.1, 0.1, 0.2]))

    assert list(ranked) == [0, 2, 1]  # +/-inf share magnitude +inf; narrower width wins


def test_rank_features_treats_nan_width_as_maximally_uncertain(ranker):
    ranked = ranker(np.array([0.2, 0.2]), width=np.array([np.nan, 0.1]))

    assert list(ranked)[-1] == 1  # D6: NaN width -> +inf, loses the narrower-wins tie-break


def test_rank_features_width_only_ranks_by_width(ranker):
    ranked = ranker(width=np.array([0.3, 0.1, 0.5]))

    assert list(ranked) == [1, 0, 2]


def test_rank_features_clamps_num_to_show_above_available(ranker):
    ranked = ranker(np.array([0.3, 0.1]), num_to_show=10)

    assert list(ranked) == [1, 0]


def test_rank_features_num_to_show_zero_is_deprecated_and_returns_all(ranker):
    with pytest.warns(DeprecationWarning, match="ADR-042 D10"):
        ranked = ranker(np.array([0.3, 0.1, 0.2]), num_to_show=0)

    assert list(ranked) == [1, 2, 0]  # legacy result kept in v1.1.0


def test_rank_features_negative_num_to_show_is_deprecated_and_drops_weakest(ranker):
    with pytest.warns(DeprecationWarning, match="ADR-042 D10"):
        ranked = ranker(np.array([0.3, 0.1, 0.2]), num_to_show=-1)

    assert list(ranked) == [2, 0]  # legacy result kept in v1.1.0


def test_rank_features_selects_strongest_with_numpy_integer(ranker):
    ranked = ranker(np.array([0.3, 0.1, 0.2]), num_to_show=np.int64(2))

    assert list(ranked) == [2, 0]  # D8 prefix of the canonical order, strongest last


@pytest.mark.parametrize("num_to_show", [1.5, "2", True])
def test_rank_features_rejects_non_integer_num_to_show(ranker, num_to_show):
    with pytest.raises(ValidationError):
        ranker(np.array([0.3, 0.1, 0.2]), num_to_show=num_to_show)


# --- FactualExplanation rule ordering (_rules_with_impact via __repr__) -------------


def test_factual_repr_orders_strongest_first_and_keeps_zero_impact():
    explanation = _make_factual(
        [0.2, -0.3, 0.0], [0.1, -0.4, 0.0], [0.3, -0.2, 0.0], ["r0", "r1", "r2"]
    )

    assert _repr_rule_order(explanation) == ["r1", "r0", "r2"]


def test_factual_repr_breaks_ties_by_width_and_agrees_with_rank_features():
    # The narrower rule wins whatever its text sorts as (D5: text is never a key).
    texts = ["z rule", "a rule"]
    explanation = _make_factual([0.2, -0.2], [0.1, -0.5], [0.3, 0.1], texts)
    widths = np.array([0.2, 0.6])

    repr_order = _repr_rule_order(explanation)
    strongest_by_rank = int(
        list(explanation.rank_features(np.array([0.2, -0.2]), width=widths))[-1]
    )

    assert repr_order == ["z rule", "a rule"]
    assert texts[strongest_by_rank] == "z rule"


@pytest.mark.parametrize(
    ("weights", "expected_order"),
    [
        ([np.nan, 0.1, 0.3], ["r2", "r1", "r0"]),
        ([0.1, np.nan, 0.3], ["r2", "r0", "r1"]),
    ],
)
def test_factual_repr_ranks_nan_impact_last_whatever_its_position(weights, expected_order):
    explanation = _make_factual(weights, [0.0] * 3, [0.0] * 3, ["r0", "r1", "r2"])

    assert _repr_rule_order(explanation) == expected_order  # D6, deterministic


def test_factual_plot_selects_canonical_top_k(monkeypatch):
    calls = []
    monkeypatch.setattr(
        explanation_module,
        "plot_regression",
        lambda _explanation, *args, **_kwargs: calls.append(list(args[3])),
    )
    explanation = _make_factual(
        [0.2, -0.3, 0.2, 0.05], [0.1, -0.4, 0.0, 0.0], [0.3, -0.2, 0.4, 0.1], list("abcd")
    )

    explanation.plot(show=False, filter_top=2)

    assert calls == [[0, 1]]  # top 2 by |w|; r0 beats r2 on the narrower interval


def test_factual_plot_non_feature_weight_metric_is_deprecated(monkeypatch):
    monkeypatch.setattr(explanation_module, "plot_regression", lambda *_a, **_k: None)
    explanation = _make_factual([0.2, -0.3], [0.1, -0.4], [0.3, -0.2], ["r0", "r1"])

    with pytest.warns(DeprecationWarning, match="ADR-042 D11"):
        explanation.plot(show=False, rnk_metric="ensured")


def test_factual_add_conjunctions_rejects_alternative_metric_before_mutation():
    explanation = _make_factual([0.2, -0.3], [0.1, -0.4], [0.3, -0.2], ["r0", "r1"])

    with pytest.raises(ValidationError):
        explanation.add_conjunctions(rnk_metric="ensured")
    with pytest.raises(ValidationError):
        explanation.add_conjunctions(n_top_features=-1)

    assert not explanation.has_conjunctive_rules


# --- AlternativeExplanation ordering ------------------------------------------------


def test_alternative_repr_ranks_by_default_ensured_score():
    explanation = _make_alternative()

    assert _repr_rule_order(explanation) == ["r1", "r2", "r0"]  # D1.2, agrees with plot


def test_alternative_plot_default_ranks_by_ensured_score(recorded_alternative_plot):
    explanation = _make_alternative()

    explanation.plot(show=False)

    # ensured, w=0.5: r1 0.865 > r2 0.800 > r0 0.400, the same order as __repr__
    assert recorded_alternative_plot == [["r1", "r2", "r0"]]


def test_alternative_plot_selects_prefix_of_canonical_order(recorded_alternative_plot):
    explanation = _make_alternative()

    explanation.plot(show=False, filter_top=2)

    assert recorded_alternative_plot == [["r1", "r2"]]


def test_alternative_plot_feature_weight_metric_is_explicit_opt_in(recorded_alternative_plot):
    explanation = _make_alternative()

    explanation.plot(show=False, rnk_metric="feature_weight")

    assert recorded_alternative_plot == [["r0", "r2", "r1"]]  # the v1.0.x |delta| order


def test_alternative_plot_negative_weight_puts_counter_alternatives_first(
    recorded_alternative_plot,
):
    explanation = _make_alternative()

    explanation.plot(show=False, rnk_weight=-0.5)

    assert recorded_alternative_plot == [["r0", "r1", "r2"]]


def test_alternative_plot_uncertainty_alias_puts_most_certain_first(
    recorded_alternative_plot,
):
    explanation = _make_alternative()

    explanation.plot(show=False, rnk_metric="uncertainty")

    # D1.2: the alias is ensured with w=0.0, score 1 - u: r1 0.98 > r2 0.80 > r0 0.60.
    assert recorded_alternative_plot == [["r1", "r2", "r0"]]


@pytest.mark.parametrize(
    ("kwargs"), [{"rnk_weight": 2.0}, {"rnk_metric": "unknown"}, {"rnk_weight": True}]
)
def test_alternative_plot_rejects_invalid_ranking_arguments(recorded_alternative_plot, kwargs):
    with pytest.raises(ValidationError):
        _make_alternative().plot(show=False, **kwargs)

    assert recorded_alternative_plot == []


def test_alternative_add_conjunctions_rejects_invalid_ranking_before_mutation():
    explanation = _make_alternative()

    with pytest.raises(ValidationError):
        explanation.add_conjunctions(rnk_weight=1.5)

    assert not explanation.has_conjunctive_rules


# --- PlotSpec presentation sort (viz/builders.py) -----------------------------------


def test_plotspec_interval_sort_breaks_width_ties_by_absolute_value():
    spec = build_regression_bars_spec(
        title=None,
        predict={"predict": 0.5, "low": 0.4, "high": 0.6},
        feature_weights={
            "predict": [0.1, -0.3, 0.2],
            "low": [0.0, -0.4, 0.0],
            "high": [0.2, -0.2, 0.4],
        },
        features_to_plot=[0, 1, 2],
        column_names=["a", "b", "c"],
        instance=[1, 2, 3],
        y_minmax=(0, 1),
        interval=True,
        sort_by="interval",
    )

    assert [bar.label for bar in spec.body.bars] == ["c", "b", "a"]


def test_plotspec_default_sort_preserves_caller_order():
    spec = build_regression_bars_spec(
        title=None,
        predict={"predict": 0.5, "low": 0.4, "high": 0.6},
        feature_weights={
            "predict": [0.1, -0.3, 0.2],
            "low": [0.0, -0.4, 0.0],
            "high": [0.2, -0.2, 0.4],
        },
        features_to_plot=[2, 0, 1],
        column_names=["a", "b", "c"],
        instance=[1, 2, 3],
        y_minmax=(0, 1),
        interval=True,
    )

    assert [bar.label for bar in spec.body.bars] == ["c", "a", "b"]
