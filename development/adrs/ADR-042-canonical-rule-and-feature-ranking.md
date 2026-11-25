> **Status note (2026-11-02):** Last edited 2026-11-25 (D17 public canonical rule selection added; D9, D13, D15 amended; #227, #204) · Archive after: Retain indefinitely as architectural record · Implementation window: decision recorded in v1.0.2 (#224); implementation and ordering changes in v1.1.0 (#204).

# ADR-042: Canonical Rule and Feature Ranking Semantics

Status: Accepted
Date: 2026-10-14
Deciders: Core maintainers
Reviewers: Core maintainers
Supersedes: None
Superseded-by: None
Related: ADR-002-validation-and-exception-design, ADR-005-explanation-json-schema-versioning, ADR-008-explanation-domain-model-and-compat, ADR-011-deprecation-and-migration-policy, ADR-020-legacy-user-api-stability, ADR-021-calibrated-interval-semantics, ADR-027-fast-feature-filtering, ADR-036-plot-spec-canonical-contract-and-validation-boundary, ADR-040-capability-verification-framework

## Context

CE orders rules in several places: `__repr__`, plot rule selection, narratives
and conjunction candidate selection. No ADR defines what "top rules" means;
ADR-008 mentions ranking only in passing. Each consumer picked its own
semantics, and they have drifted (#204).

The two explanation types carry different quantities under the same field
names:

- **Factual** `weight` is `prediction − feature_predict[f]`: the calibrated
  contribution of the observed feature value relative to a perturbed baseline.
- **Alternative** `weight` is `alt_predict − base_predict`, and
  `weight_low/high` is `alt_low/high − base_predict`. Because `base_predict` is
  constant within one explanation, `|weight|` only measures how far an
  alternative's point prediction lands from the current one, in either
  direction. Its "envelope" is the alternative's own prediction interval
  shifted by a constant. ADR-008 treats this delta as auxiliary; the
  alternative's primary payload is its prediction and interval.

Current ordering by consumer:

| Consumer | Factual | Alternative |
|---|---|---|
| Plot (single and collection) | `rnk_metric="feature_weight"` default: `rank_features(\|w\|, width)`; `rnk_metric="ensured"` accepted | **`rnk_metric="ensured"` default**: `rank_features(width=calculate_metrics(...))`; `"feature_weight"` accepted |
| `__repr__` | `_rules_with_impact`: `-\|w\|`, ties by rule text | `rank_features(\|delta\|, width)`, wider wins ties |
| Narrative canonical path | `_rules_with_impact` (as `__repr__`) | `_rules_with_impact`: `-\|delta\|`, ties by rule text |
| Narrative fallback (`core/narrative_generator.py`) | hand-copied `(\|w\|, width)` key, wider wins ties (its comment states the opposite intent) | same |
| Conjunction candidates (`explanation.py`, `legacy_conjunctions.py`) | `rank_features(\|w\|, width)` | `rank_features(\|delta\|, width)` |
| PlotSpec `sort_by` (`viz/builders.py`) | presentation re-sort; default `None` keeps input order; `"interval"` ties by `abs(value)` | same |

For an alternative explanation, the plot therefore disagrees with `__repr__`,
the narrative and conjunction selection on the ranked quantity itself, not only
on ties. Appendix A has a three-rule example in which the plot order is the
reverse of `__repr__` for the strongest rule.

The `ensured` score used by alternative plots
(`utils/helper.py::calculate_metrics`) is

`score = (1 − w)·(1 − u) + w·p̃`

- `u` is the alternative's prediction-interval width,
  `predict_high − predict_low`.
- `p̃` is the alternative's prediction:
  - For classification and thresholded regression it is oriented to the
    predicted class: `predict` if the base prediction is `> 0.5`, otherwise
    `1 − predict`.
  - For plain regression it is the raw prediction.
- `w` is the plot's `rnk_weight` (default `0.5`). A negative `w` uses `|w|`
  and negates `p̃`, which puts counter-alternatives first.

Characterisation also exposes defects:

- **`_rules_with_impact` with `NaN`:** a `NaN` impact makes the sort key
  incomparable, so the order depends on the `NaN`'s input position. A rule
  with impact `0.1` can rank above one with `0.3`.
- **`rank_features` and `NaN` width:** a `NaN` width becomes `0.0`, which
  treats unknown uncertainty as the most certain.
- **`rank_features(num_to_show=0)`:** returns every index, because `seq[-0:]`
  is the whole sequence. A negative value silently drops the weakest rules.
- **Weight-only `rank_features`:** uses the default `np.argsort`, which is not
  stable, so the order of exact ties is implementation-defined.
- **`rnk_metric="uncertainty"` alias:** the alternative plot sets `w = 1.0`,
  which removes the uncertainty term and ranks by `p̃` alone. Its source
  comment ("Put the most uncertain rules at the top") is false.
- **Factual plot accepts `rnk_metric="ensured"`:** the `ensured` metric is
  defined for alternatives and is not applicable to factual rules.

A ranking-plugin abstraction, and the ADR-041 narrative-provider draft (which
would fix ordering inside `NarrativeSpec`), both need a single agreed contract
first.

## Decision

### D1. Ranked quantity depends on the explanation type

1. **Factual (and FAST) explanations:** the ranked quantity MUST be `|w|`, the
   magnitude of the rule's calibrated feature weight. It is the only
   canonical factual ranking. The `ensured` metric is not applicable to
   factual rules (D11).
2. **Alternative explanations:** the default ranked quantity MUST be the
   `ensured` score defined in Context, with `w = 0.5`. This is exactly what
   alternative plots do today, and it becomes the default for every
   alternative consumer.
   - Callers MAY pass a different `rnk_weight` in `[-1, 1]`.
   - Callers MAY select `rnk_metric="feature_weight"` explicitly. The ranked
     quantity is then `|delta|`, treated as an explicit opt-in.
   - `rnk_metric="uncertainty"` MUST mean `ensured` with `w = 0.0`. The score
     is then `1 − u`, so the most certain alternative ranks first.
3. **Plain regression:** `ensured` keeps its current meaning, with `p̃` the raw
   prediction and `u` the interval width in target units. CE does not
   normalise either term. Choosing a `w` that fits the task, including the
   direction and the scale trade-off, is the user's responsibility. The
   documentation MUST say so (see Adoption).
4. A higher primary key always ranks stronger.

### D2. Direction

Canonical ranking output MUST be strongest-first: position 0 holds the
strongest rule.

### D3. Sign and partitions

1. For factual rules, sign MUST NOT affect rank.
2. For alternatives, direction enters only through `p̃` and the sign of `w`
   (D1.2).
3. The canonical result MUST carry each rule's sign or direction unchanged.
4. A consumer that groups rules MUST obtain each group by filtering the
   canonical order. Examples are the narrative positive/negative and
   certain/uncertain groups. Each group MUST keep the canonical order and
   MUST NOT be re-sorted. Grouping is presentation (D16): it MUST NOT change
   which rules are selected.

### D4. Uncertainty

1. **Factual:** the width `abs(weight_high − weight_low)` is only a tie-break.
   When two rules tie on `|w|`, the **narrower** interval ranks first.
2. **Alternative:** uncertainty is part of the ranked quantity through `u` in
   the `ensured` score. The width tie-break in D5 applies only to exact score
   ties.

### D5. Deterministic total order

The canonical sort key MUST be this total order:

1. the primary key (D1), descending;
2. the width, ascending (narrower first):
   - for factual rules, the weight-envelope width;
   - for alternatives, `u`;
3. the rule's index in the explanation's native rule table (`get_rules()`
   order), ascending.

Rule text, feature labels and display strings MUST NOT be ranking keys.
Implementations MUST order by these explicit keys. They MUST NOT rely on the
default `np.argsort` kind or on input order for anything the keys leave
undetermined.

### D6. Non-finite values

Ranking MUST NOT rewrite reported values; payloads stay as computed. For
ordering only:

- **Factual `|w|`:** `±inf` has magnitude `+inf` and ranks ahead of every
  finite value.
- **Alternative score:** the order is signed. `+inf` ranks first and `−inf`
  ranks after every finite score. For example, a one-sided regression
  interval gives `u = inf`, so the score is `−inf` and the rule ranks after
  every finite score.
- **`NaN` primary key:** ranks after every other value and MUST NOT raise.
- **`NaN` or non-finite width:** treated as `+inf` (maximally uncertain), so
  that rule loses width ties.

### D7. Zero impact and filtering

Ranking MUST NOT drop rules. Filtering is a separate, documented step, and it
MUST keep the canonical order of the rules it retains. Examples are the
alternative plot's removal of rules identical to the base prediction, and FAST
filtering (ADR-027).

### D8. `top_k` semantics

- `None` returns all rules.
- `top_k >= n` returns all `n` rules, without padding, warning or error.
- `top_k == 0` returns an empty result.
- A negative or non-integer `top_k` raises `ValidationError` (ADR-002).
- A `top_k` selection MUST be a prefix of the canonical order.

### D9. One authoritative implementation and its consumers

Exactly one internal implementation MUST produce canonical rankings, for both
explanation types. It returns strongest-first indices or a typed ranking
result; its module and name are an implementation choice for #204. The
following consumers MUST delegate to it, using the canonical quantity of their
explanation type:

- both `_rules_with_impact` implementations and both `__repr__` methods;
- plot rule selection, both single-explanation and collection;
- the narrative canonical and fallback paths, following plot logic (D16). The
  fallback's hand-copied `rank_key` MUST be removed.
- **conjunction candidate selection, in both `explanation.py` and
  `legacy_conjunctions.py`.** Candidates MUST be the canonical top
  `n_top_features` of the current rule set:
  - for factual explanations, by `|w|`;
  - for alternatives, by the `ensured` default.

  `add_conjunctions()` MUST accept `rnk_metric` and `rnk_weight` (see D11).
  This lets candidate selection use the same ranking the user later plots or
  narrates. `rnk_metric="feature_weight"` reproduces the v1.0.x alternative
  candidate sets.

  `|delta|` is not a justified default search heuristic for alternatives. It selects
  candidates by a quantity the alternative ranking does not use, and it
  systematically drops supportive, low-uncertainty rules when the base
  prediction is already confident. Those rules have a small delta and a high
  `ensured` score, and they are exactly the rules the default ranking puts
  first. With `|delta|` candidates, `n_top_features` would mean something
  different from the "top features" a user sees in the plot.

The public selection function (D17) is a further consumer and MUST delegate to
the same implementation.

A consumer that has to deviate MUST record a justified alternative as an
amendment to this ADR.

### D10. Public `rank_features`

`rank_features` is public API, but it is not part of the ADR-020 legacy
contract. Its signature MUST be kept. It MUST keep returning indices in
**ascending (strongest-last)** order and becomes a thin adapter over the
canonical implementation:

- **With `feature_weights`:** the factual key.
- **With `width` only:** a precomputed score ranked by D5-D6. This is how the
  alternative plots pass `ensured` today.

The legacy results for `num_to_show <= 0` are deprecated under ADR-011:

- v1.1.0 MUST emit `DeprecationWarning` and keep the legacy result.
- From the second minor release after v1.1.0, D8 applies: `0` returns an
  empty result and negative values raise.

### D11. Ranking knobs are type-specific

1. `rnk_metric` and `rnk_weight` are alternative-ranking parameters. From
   v1.1.0 they MUST be accepted, with the same defaults and the same meaning,
   by three entry points:
   - plots, as today;
   - `add_conjunctions()` (D9);
   - `to_narrative()` (D16).

   Both new entry points already take `**kwargs`, so the additions are
   additive.
2. For factual and FAST explanations, any `rnk_metric` other than
   `"feature_weight"` is not a canonical ranking. The plot ranking knobs are
   part of the ADR-020 legacy contract, so the factual handling MUST change in
   two steps:
   - From v1.1.0, a factual plot given such a value MUST emit a
     `DeprecationWarning` and keep its current behaviour.
   - The non-`feature_weight` factual path MAY be removed only in v2.0.0. This
     ADR is the decision record that ADR-020 requires for that removal.
3. The keywords themselves are kept, because collection `.plot` serves both
   explanation types.
4. On the new `add_conjunctions()` and `to_narrative()` entry points, a
   factual or FAST explanation given an `rnk_metric` other than
   `"feature_weight"` MUST raise `ValidationError` (ADR-002). These keywords
   have no defined meaning there today.

### D12. PlotSpec `sort_by` is presentation-only

PlotSpec builders' `sort_by` remains a presentation-only override.

- With the default (`None`), builders MUST preserve the canonical order they
  receive.
- `sort_by` MUST NOT affect which rules are selected.
- A builder-local secondary key is permitted, for example `abs(value)` under
  `sort_by="interval"`.
- Any ties that remain MUST fall back to canonical order through a stable
  sort.

### D13. Serialized payloads are not rankings

The order of rules in serialized explanation payloads (ADR-005) is the native
rule-table order. This ADR does not change it. A consumer that needs a ranked
view MUST compute one with D9; outside CE it does so through the public D17
function, which returns indices into the unchanged `rules` array.

### D14. No ranking guarantees

A ranking is a deterministic function of a single explanation's point
estimates and intervals. Docs, docstrings, capability claims (ADR-040) and
narrative templates MUST NOT claim any of the following:

- a distribution-free or other statistical guarantee on rank stability;
- that a top-`k` set is correct;
- that a ranking measures global feature importance.

Calibration guarantees (ADR-021) attach to predictions and intervals, not to
ranks. The `ensured` score combines two calibrated quantities, but the score
itself carries no guarantee.

### D15. Interaction with ADR-041 (narrative facts)

*Amended 2026-11-25 (#227):* ADR-041 defines no `NarrativeSpec` or other
separate narrative fact record. CE's own narration obtains rule selection and
order from D9 through D16. Any consumer outside CE, including one that writes
its own narrative text from a stored payload, obtains the same selection and
order through D17. A consumer MAY regroup the selected rules as presentation
(D3.4) but MUST NOT re-rank them, and any reordering it shows MUST be declared
as presentation (D12).

*History:* the 2026-11-02 text required `NarrativeSpec` to carry rules in D9
order with the ranking key, and forbade consumers of a `NarrativeSpec` to
re-rank (amending an earlier "provider" wording). The record itself is void
with the 2026-11-25 revision of ADR-041; its no-re-ranking rule now applies to
consumers of D17.

### D16. Narratives follow plot logic

For each explanation type, a narrative MUST describe the same rules, in the
same order, that the corresponding plot shows under the same ranking
arguments:

1. `to_narrative()` MUST accept `rnk_metric`, `rnk_weight` and `filter_top`.
   Each takes the default of the `plot` method at the same level:
   - `rnk_metric`/`rnk_weight`: `feature_weight` for factual explanations;
     `ensured` with `w = 0.5` for alternatives.
   - `filter_top`: all rules for a single explanation, because
     `CalibratedExplanation.plot` defaults to `filter_top=None`. For a
     collection it is `10`, because `CalibratedExplanations.plot` defaults to
     `filter_top=10`.
2. Rule selection MUST be the canonical `top_k` prefix (D8). The current
   per-sign cut, "top features (min 3)", is `max(3, len(group))` and therefore
   keeps every rule. It MUST be replaced by the canonical selection.
3. The narrative MUST apply the same filtering steps as the plot (D7). For
   example, alternative rules that are identical to the base prediction are
   removed in both.
4. Narrative grouping is presentation over that ordered selection (D3.4).
   Examples are positive/negative and, at the advanced level,
   certain/uncertain.

### D17. Public canonical rule selection

*Added 2026-11-25 (#227, implemented by #204).* External consumers MUST be
able to obtain exactly the rule selection and order that CE's own plots and
narratives use, from a stored explanation and without private API, the
explainer, the model or calibration data.

1. **Interface.** CE MUST expose one public function,
   `calibrated_explanations.explanations.canonical_rule_indices(payload, *,
   rnk_metric=None, rnk_weight=0.5, filter_top=None) -> list[int]`. `payload`
   is one ADR-005 explanation payload: an entry of
   `to_json()["explanations"]`, also after a JSON round trip. The result is
   the selected indices into that payload's `rules` array, strongest first.
2. **One implementation.** The function MUST delegate to the D9
   implementation, including its knob and `top_k` validation. It MUST NOT
   contain a second ranking, scoring or tie-breaking algorithm.
3. **Inputs.** It reads only `explanation_type`, `task`,
   `prediction.{predict, low, high}`, each rule's `rule_weight` and
   `rule_prediction` `{predict, low, high}` (with the serialized `"-inf"`/`"inf"`
   strings), and `metadata.narrative_context.problem_type`. The ranked quantity
   follows D1: `|rule_weight.predict|` with the weight-interval tie-break for
   factual and FAST explanations; for alternatives, `ensured` on
   `rule_prediction` with `u` its interval width, or `|rule_weight.predict|`
   under `feature_weight`. The probability orientation of D1.2 applies when
   `task` is `classification` or `problem_type` is `probabilistic_regression`,
   oriented by `prediction.predict`.
4. **Selection.** The result is the canonical order (D1-D6), then the
   `filter_top` prefix (D8; `None` selects all), then, for alternatives only,
   the removal of rules whose `rule_prediction` `predict`, `low` and `high`
   all equal the payload `prediction`'s, with the same tolerance as the plot
   (D7). The order of the retained rules is unchanged, so an alternative
   result can be shorter than `filter_top`. Factual and FAST results keep
   zero-weight rules (D7). This is exactly the selection of the
   single-explanation `plot()` of the same type with the same arguments, and
   therefore of `to_narrative()` (D16).
5. **Knobs.** D11 applies: factual and FAST payloads accept only `None` or
   `"feature_weight"`. The deprecated non-canonical factual plot metrics
   (D11.2) have no counterpart. Alternatives accept `"ensured"` (default,
   `rnk_weight` in `[-1, 1]`), `"uncertainty"` (`ensured` with `w = 0.0`) and
   `"feature_weight"`.
6. **Determinism, ties and unbounded intervals.** D5 and D6 apply with the
   rule's position in `rules` as the native index. A non-finite interval
   endpoint, including an open one-sided CPS endpoint serialized as `-inf` or
   `inf` (ADR-021), gives an infinite width. Identical payloads give identical
   results on every platform.
7. **Single explanations and collections.** The function ranks one
   explanation. Its default `filter_top=None` matches
   `CalibratedExplanation.plot`; a consumer reproducing
   `CalibratedExplanations.plot` or collection-level `to_narrative()` passes
   `filter_top=10` (D16.1). A multiclass collection serializes one payload per
   (instance, class); each is ranked on its own. Where a collection plot path
   does not apply this selection (the multiclass multi-label plot currently
   ranks each class over all rules and skips the base-identical filter), #204
   MUST either align it with this decision or record it here as a declared
   exception.
8. **Errors.** Invalid knobs or `filter_top`, an unknown `explanation_type`,
   and missing or non-numeric rule values raise `ValidationError` (ADR-002).
   The `ensured` ranking (including `"uncertainty"`) of a `regression`
   alternative payload without `metadata.narrative_context` raises
   `ValidationError`, because whether it is thresholded is unknown;
   `feature_weight` still works. No other payload validation is added:
   interval invariants are enforced when the payload is written (ADR-021).
9. **Result type and stability.** The result is a plain `list[int]`: no
   scores, ranking schema, version or hash. Serialized payloads keep their
   rule order (D13). The function is public API under ADR-011 and is not part
   of the ADR-020 legacy contract. D14 applies to its documentation.
10. **Payload fidelity precondition.** Parity requires each payload rule's
    `rule_weight` and `rule_prediction` to come from the same rule-table entry
    as its `rule` text (`conjunctive_rules` when present, otherwise
    `get_rules()`), which is what plots and narratives rank. Up to
    v1.1.0-dev, `to_json()` takes them from feature-indexed arrays, so for
    factual and alternative explanations, and for every conjunctive rule, the
    serialized values can belong to another feature (#262; the rule order and
    count were already aligned). #204 depends on that fix. For payloads
    written before it the function ranks the stored values and parity with
    the original plot is not guaranteed; the documentation MUST say so.
11. **Verification.** #204 MUST verify parity against the live plot selection
    for factual, FAST, alternative, conjunctive, multiclass, thresholded and
    one-sided regression explanations, with the D11 knobs, `filter_top`, ties
    and non-finite values, before and after a JSON round trip.

## Governed claims

This ADR constrains the rule order that these capabilities produce. They do
not currently claim any ranking behaviour; D14 governs how any future ranking
statement in them may be phrased. #204 decides whether a dedicated ranking
claim is added in v1.1.0, once there is canonical behaviour to verify.

- `CE-CAP-EXPL-001` — factual explanations; `__repr__`, narrative and plot order by `|w|` (D1.1, D2-D9).
- `CE-CAP-EXPL-002` — alternative explanations; `ensured` order in every consumer (D1.2, D2-D9).
- `CE-CAP-EXPL-CONJ-001` — conjunctive rules; candidate selection via the canonical ranking of each type (D9).
- `CE-CAP-NARR-001` — narratives; same rules and order as plots, grouping as presentation (D3, D9, D15, D16).

#204 decides whether a dedicated claim for the D17 public selection is added
under ADR-040; D17.11 lists the behaviour it would verify.

## Alternatives Considered

1. **`|w|` for both explanation types.** Rejected for the reasons below.
   - It reuses the factual quantity for a field that means something else for
     alternatives.
   - It ignores direction relative to the decision, so supportive and counter
     alternatives of equal size tie.
   - It ignores whether an alternative crosses the decision boundary.
   - It reduces the alternative's uncertainty to a tie-break.
   - It would reorder every alternative plot away from its long-standing
     default.
2. **Keep `|delta|` for alternative conjunction candidates as a search
   heuristic.** Rejected: see D9. It makes candidate selection disagree with
   the ranking users see and with the meaning of `n_top_features`.
3. **Wider interval wins factual ties (codify current `rank_features`).**
   Rejected: with equal contribution, it would put the less certain evidence
   first. It is also the opposite of the intent recorded in the narrative
   fallback's comment.
4. **Uncertainty as a default factual penalty, or `ensured` for factual
   explanations.** Rejected: `ensured` is not applicable to factual rules
   (D11).
5. **Rule text as the canonical tie-break.** Rejected: text is a display
   artefact. It changes with feature names, formatting and conjunction
   separators, and it is not a rule identity.
6. **Keep consumers independent and document each one.** Rejected: #204 and
   Appendix A show that they have already drifted.
7. **Record the contract as a Standard rather than an ADR.** Rejected: it is a
   cross-cutting behavioural contract with ADR-011 and ADR-020 consequences.
8. **Use public `rank_features` as the external contract (D17).** Rejected:
   it works on caller-built arrays, returns strongest last, and leaves the
   `ensured` score, its probability orientation and the base-identical filter
   to the caller, who would have to re-implement them.
9. **Publish a typed ranking result with scores, key and version, or a
   narrative fact record (D17).** Rejected: no consumer needs scores or a
   ranking schema, and ADR-041 D2 forbids a separate narrative record.
   Indices into the payload's own `rules` array are the smallest interface
   that reproduces CE's selection.
10. **Store rules in canonical order in the payload (D17).** Rejected: the
    order depends on caller-chosen knobs, and D13 keeps native order.

## Consequences

### Positive

- For each explanation type, `__repr__`, plots, narratives and conjunction
  selection agree on what "top rules" means.
- The alternative default is the one users already see in plots.
- The order is total and platform-independent, and `NaN`/`inf` have defined
  placement.
- It gives any future ranking-plugin family or narrative record (ADR-041 D2) a
  fixed contract to build on.
- External consumers obtain CE's exact rule selection from a stored payload
  through one public function, without private API or the model (D17).

### Negative / Risks

- **Alternative `__repr__` and narrative order change wholesale** in v1.1.0,
  from `|delta|` to `ensured`.
- **Alternative conjunction candidate sets change,** so `add_conjunctions()`
  on alternatives can generate different conjunctions.
- **Factual order changes only for exact ties, non-finite values and invalid
  `top_k`.**
- Golden and parity fixtures (`tests/golden_explanation_v1.json`,
  `tests/parity_reference/`) may need regeneration. Each diff must be reviewed
  as a deliberate behaviour change.
- Candidates are chosen when `add_conjunctions()` runs. A user who later
  plots or narrates with a non-default `rnk_weight` has to pass the same
  `rnk_metric`/`rnk_weight` to `add_conjunctions()`. Otherwise the conjunctions
  were selected under a different ranking, so the documentation must state
  this.
- Narrative content changes in v1.1.0 because of D16: the rule order, the rule
  selection (`filter_top`) and the base-identical filtering. In particular,
  collection-level `to_narrative()` will describe at most 10 rules per
  instance by default, matching the collection plot; today it describes all
  rules.
- `rnk_metric="uncertainty"` changes meaning (from `w = 1.0` to `w = 0.0`) for
  anyone who relied on the current behaviour.
- `rank_features` keeps the opposite orientation (strongest-last) to the
  canonical result, and its docstring must say so explicitly.
- D17 adds a public function that must be kept under ADR-011. Payloads written
  before the D17.10 fix (#262) give no plot parity, and the base-identical
  filter of D17.4 compares serialized floats with the plot's tolerance.

## Adoption & Migration

- **v1.0.2 (#224, this ADR):** decision only, with no production ranking
  change. Current behaviour is pinned by
  `tests/unit/explanations/test_ranking_characterisation.py` (Appendix A).
- **v1.1.0 (#204):**
  - Implement D9 and migrate every consumer.
  - Remove the narrative duplicate.
  - Add the D10 and D11 deprecation warnings.
  - Change `rnk_metric="uncertainty"` to `w = 0.0` (D1.2).
  - Add `rnk_metric`/`rnk_weight` to `add_conjunctions()` (D9, D11), and add
    `rnk_metric`/`rnk_weight`/`filter_top` to `to_narrative()` (D16).
  - Rewrite the Appendix A tests to match this contract.
  - Add the D17 public selection function and verify D17.11 parity, after the
    D17.10 payload fidelity fix (#262).
- **Compatibility classification:**
  - These are behaviour changes. No existing parameter is removed or renamed.
    The D9, D11 and D16 additions are new keyword arguments on methods that
    already accept `**kwargs`.
  - The ADR-020 legacy contract pins entry points and parameters (`filter_top`,
    the ranking knobs, `n_top_features`). It does not pin rule order.
  - The changes MAY land in the minor release v1.1.0 and MUST NOT land in a
    v1.0.x patch release.
  - v1.1.0 MUST include a CHANGELOG "Behaviour changes" entry. It MUST call out
    the alternative `__repr__`, narrative and conjunction changes and the
    `"uncertainty"` alias change. It MUST also say that
    `rnk_metric="feature_weight"` reproduces the v1.0.x alternative order in
    plots and narratives, and the v1.0.x candidate sets in
    `add_conjunctions()`.
  - Deprecations follow D10 (ADR-011) and D11 (ADR-020).
- **Documentation:** in the same release, update docs and docstrings
  (`rank_features`, plot `filter_top`/`rnk_metric`/`rnk_weight`,
  `add_conjunctions`, `to_narrative`, narrative how-tos) to describe the
  canonical order of each type and the D14 non-guarantee. The docs MUST also:
  - explain that in plain regression `ensured` combines the raw prediction
    with an interval width in target units, so the user chooses `w` for their
    task (D1.3);
  - tell users to pass the same ranking arguments to `add_conjunctions()` as
    to the plot or narrative they intend to read.

## Open Questions

None. *Resolved 2026-11-25:* the D9 implementation stays internal; its
public surface is the D17 index function only, with no typed ranking result.

## Appendix A: Current behaviour (v1.0.x) and its disposition

Executable evidence: `tests/unit/explanations/test_ranking_characterisation.py`.

| Current behaviour (test) | Disposition under this ADR |
|---|---|
| `rank_features` returns strongest last (`..._ascending_with_strongest_last`) | Kept for `rank_features` (D10); canonical output is strongest-first (D2) |
| Sign ignored (`..._ignoring_sign`) | Kept for factual explanations (D3) |
| Wider interval wins magnitude ties (`..._in_favour_of_wider_interval`) | Changes: narrower wins (D4, D5) |
| Full tie: highest index strongest (`..._highest_index_strongest`) | Changes: lowest index strongest (D5) |
| `NaN` weight treated as `0` (`..._nan_weight_to_zero`) | Changes: `NaN` ranks last (D6) |
| `+inf`/`-inf` share a magnitude (`..._positive_and_negative_infinity`) | Kept for factual `\|w\|` (D6) |
| `NaN` width treated as most certain (`..._nan_width_to_most_certain`) | Changes: treated as maximally uncertain (D6) |
| Width-only ranking (`..._width_only_ranks_by_width`) | Kept as the precomputed-score path (D10) |
| `num_to_show > n` clamps (`..._clamps_num_to_show_above_available`) | Kept (D8) |
| `num_to_show=0` returns all; negative drops weakest | Deprecated, then D8 (D10) |
| Factual `__repr__` strongest-first, zero impact kept | Kept (D2, D7) |
| Factual `__repr__` ties by rule text, disagreeing with `rank_features` | Changes: width, then index (D5) |
| Factual `__repr__` `NaN` order depends on input position | Changes: `NaN` last, deterministically (D6) |
| Alternative `__repr__` ranks by `\|delta\|`: `r0, r2, r1` (`test_alternative_repr_...`) | Changes: `ensured`, giving `r1, r2, r0` (D1.2, D9) |
| Alternative plot default `ensured`: `r1, r2, r0` (`test_alternative_plot_default_...`) | Kept; canonical alternative default (D1.2) |
| Alternative plot `rnk_metric="feature_weight"`: `r0, r2, r1` | Kept as explicit opt-in (D1.2) |
| Alternative plot `rnk_weight=-0.5` puts counter-alternatives first: `r0, r1, r2` | Kept (D1.2) |
| `rnk_metric="uncertainty"` ranks by `p̃` only: `r2, r1, r0` (`..._uncertainty_alias_...`) | Changes: `w = 0.0`, most certain first, giving `r1, r2, r0` (D1.2) |
| PlotSpec `sort_by="interval"` ties by `abs(value)` | Kept as presentation-only (D12) |
| PlotSpec default keeps caller order | Kept (D12) |

The following are characterised by inspection rather than by separate tests,
because no private-member test access is allowlisted:

- **`AlternativeExplanation._rules_with_impact`** uses the same `-|delta|`,
  rule-text key as the factual implementation.
- **The narrative fallback `rank_key`** reproduces the `rank_features` key.
- **Conjunction candidate selection** calls `rank_features` on `|w|` or
  `|delta|`, as described in Context.
- **The factual plot's acceptance of `rnk_metric="ensured"`** follows the
  `calculate_metrics` path, as described in Context.
