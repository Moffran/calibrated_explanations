# Guarded explanations (in-distribution filtering)

Guarded explanations answer "what feature changes are plausible given the
calibration data?" by filtering out perturbations that would move the instance
out of distribution. They are a strict subset of standard calibrated
explanations — same calibrated predictions and uncertainty intervals, but
restricted to perturbations that are statistically conforming to the
calibration set.

> **Audience:** Practitioner / Researcher

```{admonition} Guarantees & Assumptions
:class: important

- Guarded rules are a strict subset of standard CE rules — same math, fewer
  candidates. Every guarded rule would also appear in an unguarded explanation
  (modulo discretisation granularity).
- In-distribution filtering relies on KNN-based conformal p-values computed
  on the calibration set. Validity requires the same exchangeability
  assumption as the prediction intervals themselves.
- Bonferroni correction is optional (``bonferroni_correction``). When enabled,
  it controls the family-wise error rate across bins within each feature.
- Filtering reduces the risk of suggesting implausible interventions but does
  not guarantee causal actionability.

See ADR-032 for the semantic identity contract and ADR-021 for the formal
interval semantics.
```

## Why guard explanations?

Standard factual and alternative explanations perturb each feature to
representative values drawn from discretisation bins. Some perturbations may
create instances that are unlikely under the data-generating distribution —
for example, a 20-year-old with 40 years of work experience, or a tumour
radius outside any observed range.

Presenting such out-of-distribution perturbations to users erodes trust and
can lead to misguided decisions. A practitioner who sees "if feature X were
set to value Y, the prediction would change" needs confidence that the
suggested change is realistic.

Guarded explanations solve this by testing each perturbation against the
calibration set using conformal anomaly detection, keeping only those
perturbations whose conformal p-value exceeds a user-controlled significance
threshold. The result is fewer but more trustworthy rules.

## How it works

Guarded explanations follow four high-level steps:

1. **Multi-bin discretisation** — Each feature is split into multiple
   intervals (bins) rather than a single binary split. This yields richer
   interval-based conditions (``"30 < age <= 50"``) instead of simple
   threshold splits (``"age > 30"``).

2. **Perturbation and conformity testing** — For each bin, a representative
   value is computed (median of calibration samples in that bin). The instance
   is perturbed to that representative, and the perturbed instance is tested
   against the calibration distribution via a KNN-based conformal p-value.
   Bins whose perturbed instance is out-of-distribution (p-value below the
   significance threshold) are excluded. If ``bonferroni_correction=True``,
   the significance level is divided by the number of bins for each feature to
   control family-wise error rate.

3. **Calibrated prediction** — All conforming perturbed instances are scored
   through the same calibrated prediction pipeline used for standard
   explanations, producing point predictions and uncertainty intervals. The
   interval invariant (``low <= predict <= high``) is strictly enforced.

4. **Optional bin merging** — Adjacent conforming bins can optionally be
   merged into wider intervals. Merged representatives are re-tested via the
   guard; merges that fail the re-test are reverted to preserve validity.

## Factual vs alternative guarded explanations

### ``explain_guarded_factual()``

Reports how the prediction would change if the instance's feature values were
perturbed *within* their current in-distribution bin. The factual bin is the
one containing the actual observed value. Only conforming factual bins
produce rules.

Use this when you want to understand the feature attribution for the current
instance, with the guarantee that all reported perturbations are plausible.

### ``explore_guarded_alternatives()``

Reports what in-distribution feature value changes could shift the prediction.
Only non-factual conforming bins appear as alternatives. The factual bin acts
as a barrier during merging — left-of-factual and right-of-factual bins merge
independently but never across the factual bin. This preserves the semantic
distinction between "what the instance is" and "what could be changed."

Use this when you want actionable suggestions for changing the prediction,
with the guarantee that every suggested intervention is in-distribution.

## Parameters

| Parameter | Default | Meaning |
|---|---|---|
| ``significance`` | ``0.1`` | Confidence level for the conformity test. Lower values mean stricter filtering (fewer rules). |
| ``merge_adjacent`` | ``False`` | Whether to merge adjacent conforming bins into wider intervals. |
| ``n_neighbors`` | ``5`` | Number of neighbors for the KNN non-conformity measure. |
| ``normalize_guard`` | ``True`` | Per-feature min-max normalisation before computing distances. |

Standard parameters (``threshold``, ``low_high_percentiles``, ``bins``,
``features_to_ignore``) are also accepted and behave identically to their
counterparts in ``explain_factual`` and ``explore_alternatives``.

## Interaction with existing features

Because guarded explanations subclass the standard explanation types
(``GuardedFactualExplanation`` extends ``FactualExplanation``;
``GuardedAlternativeExplanation`` extends ``AlternativeExplanation``), all
existing features work identically:

- **Plotting** — ``plot()``, ``plot(style="triangular")``, global plots
- **Narratives** — ``to_narrative()``
- **Conjunctions** — ``add_conjunctions()``
- **Reject policies** — all reject policy configurations
- **Ensured framework** — ``ensured_explanations()``,
  ``super_explanations()``, ``semi_explanations()``,
  ``counter_explanations()``, ``pareto_explanations()``
- **Caching and parallel execution** — supported without modification

This semantic identity is guaranteed by ADR-032: the only permitted
differences are the bin conditions (interval rules from multi-bin
discretisation) and which perturbations are selected (conforming only).

## Guarded audit and interpretation

Guarded explanations expose a dedicated audit API that is separate from
``get_rules()``:

- Per explanation: ``guarded_explanations[0].get_guarded_audit()``
- Per collection: ``guarded_explanations.get_guarded_audit()``

The payload includes:

- Summary counts: tested, conforming, removed by guard, emitted.
- Full interval records: bounds, representative value, p-value, conformity,
  factual flag, merge flag, emission flag, emission reason, and calibrated
  ``predict/low/high`` for that interval.

Use ``calibrated_explanations.ce_agent_utils.print_guarded_audit_table(...)``
for a compact tabular notebook view.

### How to read counts

- ``intervals_removed_guard`` is defined strictly as ``conforming == False``.
- ``intervals_emitted`` is stricter than conformity. Conforming intervals can
  still be excluded by explanation semantics (for example factual-only or
  alternative-only eligibility).

### Emission reason semantics

- ``emitted``: interval became a rule.
- ``removed_guard``: interval failed conformity and was removed.
- ``design_excluded``: interval is not eligible for this explanation mode.
  In factual mode, non-factual bins are excluded. In alternative mode, factual
  bins are excluded.
- ``baseline_equal``: alternative interval is conforming but yields the same
  ``predict``, ``low``, and ``high`` as baseline.
- ``zero_impact``: factual interval is conforming but has no prediction impact.
- ``ignored_feature``: feature was explicitly ignored by configuration.

### ``baseline_equal`` vs ``zero_impact``

- ``zero_impact`` applies to guarded factual explanations.
- ``baseline_equal`` applies to guarded alternative explanations and checks the
  full triplet equality ``(predict, low, high)``.

## Minimal API examples

```python
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from calibrated_explanations import WrapCalibratedExplainer

# Setup
dataset = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, test_size=0.2,
    stratify=dataset.target, random_state=0,
)
x_proper, x_cal, y_proper, y_cal = train_test_split(
    x_train, y_train, test_size=0.25,
    stratify=y_train, random_state=0,
)

explainer = WrapCalibratedExplainer(RandomForestClassifier(random_state=0))
explainer.fit(x_proper, y_proper)
explainer.calibrate(x_cal, y_cal, feature_names=dataset.feature_names)

# Guarded factual explanations
guarded_factual = explainer.explain_guarded_factual(x_test[:1], significance=0.1)
print(guarded_factual[0])
guarded_factual[0].plot(show=False)

# Guarded alternative explanations
guarded_alts = explainer.explore_guarded_alternatives(x_test[:1], significance=0.1)
alt0 = guarded_alts[0]
print(alt0)
alt0.plot(show=False)

# All standard filters work on guarded alternatives
ensured = alt0.ensured_explanations()
counter = alt0.counter_explanations()
alt0.add_conjunctions(n_top_features=5, max_rule_size=2)
```

### Adjusting the significance level

```python
# Strict filtering — only the most clearly in-distribution perturbations
strict = explainer.explain_guarded_factual(x_test[:1], significance=0.01)

# Lenient filtering — more rules, closer to unguarded behaviour
lenient = explainer.explain_guarded_factual(x_test[:1], significance=0.2)
```

## Cross-references

- {doc}`alternatives` — Standard alternatives and the ensured framework
- {doc}`explanation_structures` — Internal data structures for explanations
- {doc}`../../get-started/quickstart_guarded` — Quickstart for guarded explanations
- {doc}`../../researcher/advanced/theory_and_literature` — Research publications
  underpinning calibrated explanations
