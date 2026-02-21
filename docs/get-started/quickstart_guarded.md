# Guarded explanations quickstart

Run guarded (in-distribution) explanations for classification and regression.
Guarded explanations filter out perturbations that fall outside the calibration
distribution, producing fewer but more trustworthy rules.

## Prerequisites

```bash
pip install calibrated-explanations scikit-learn
```

```{admonition} Guarantees & Assumptions
:class: important

* Guarded explanations require a held-out calibration set (the `x_cal`, `y_cal` split below).
* In-distribution filtering uses conformal p-values on the calibration set — validity requires the same exchangeability assumption as the prediction intervals.
* Bonferroni correction is optional (``bonferroni_correction``); when enabled,
  it controls family-wise error rate across bins within each feature.
* All standard CE features (plotting, narratives, conjunctions, ensured filters) work identically on guarded explanations.
```

## 1. Load data and split sets

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

dataset = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, test_size=0.2,
    stratify=dataset.target, random_state=0,
)
x_proper, x_cal, y_proper, y_cal = train_test_split(
    x_train, y_train, test_size=0.25,
    stratify=y_train, random_state=0,
)
```

## 2. Fit and calibrate the explainer

```python
from sklearn.ensemble import RandomForestClassifier
from calibrated_explanations import WrapCalibratedExplainer

explainer = WrapCalibratedExplainer(RandomForestClassifier(random_state=0))
explainer.fit(x_proper, y_proper)
explainer.calibrate(x_cal, y_cal, feature_names=dataset.feature_names)
```

## 3. Generate guarded factual explanations

```python
guarded_factual = explainer.explain_guarded_factual(x_test[:5], significance=0.1)
print(guarded_factual[0])
```

The output has the same structure as standard factual explanations — prediction
with uncertainty interval, followed by feature rules with calibrated weights.
The difference is that only in-distribution perturbations appear.

```python
# Plot the first guarded factual explanation
guarded_factual[0].plot(show=False)
```

## 4. Explore guarded alternatives

```python
guarded_alts = explainer.explore_guarded_alternatives(x_test[:2], significance=0.1)
alt0 = guarded_alts[0]
print(alt0)
```

Guarded alternatives suggest only feature value changes that are plausible
given the calibration distribution. All ensured framework filters work:

```python
# Ensured alternatives — narrower uncertainty than the base
ensured = alt0.ensured_explanations()

# Counter alternatives — cross the decision boundary
counter = alt0.counter_explanations()

# Add multi-feature conjunctions
alt0.add_conjunctions(n_top_features=5, max_rule_size=2)

# Plot
alt0.plot(show=False)
```

## 5. Adjust the significance level

The ``significance`` parameter controls how strict the in-distribution filter
is. Lower values produce fewer, more conservative rules:

```python
# Strict: only the most clearly in-distribution perturbations
strict = explainer.explain_guarded_factual(x_test[:1], significance=0.01)

# Lenient: more rules, closer to standard (unguarded) behaviour
lenient = explainer.explain_guarded_factual(x_test[:1], significance=0.2)
```

## 6. Regression example

Guarded explanations work identically for regression:

```python
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor

diabetes = load_diabetes()
x_train_r, x_test_r, y_train_r, y_test_r = train_test_split(
    diabetes.data, diabetes.target, test_size=0.2, random_state=0,
)
x_proper_r, x_cal_r, y_proper_r, y_cal_r = train_test_split(
    x_train_r, y_train_r, test_size=0.25, random_state=0,
)

reg_explainer = WrapCalibratedExplainer(RandomForestRegressor(random_state=0))
reg_explainer.fit(x_proper_r, y_proper_r)
reg_explainer.calibrate(x_cal_r, y_cal_r, feature_names=diabetes.feature_names)

# Guarded factual for regression
guarded_reg = reg_explainer.explain_guarded_factual(x_test_r[:1], significance=0.1)
print(guarded_reg[0])

# Guarded alternatives for regression
guarded_reg_alts = reg_explainer.explore_guarded_alternatives(
    x_test_r[:1], significance=0.1,
)
print(guarded_reg_alts[0])
```

## 7. Audit guarded filtering decisions

Use the guarded audit API to inspect every tested interval, its p-value, and
why it was emitted or excluded:

```python
from calibrated_explanations.ce_agent_utils import print_guarded_audit_table

# Collection-level audit payload (aggregated + per-instance records)
audit = guarded_factual.get_guarded_audit()

# Compact notebook table (rounded bounds/p-values + legend)
print_guarded_audit_table(audit, max_rows=40, bound_decimals=3, pvalue_decimals=3)
```

Interpret the summary fields as:

- ``intervals_tested``: total number of interval candidates evaluated.
- ``intervals_conforming``: candidates that passed the in-distribution guard.
- ``intervals_removed_guard``: candidates removed strictly because
  ``conforming == False``.
- ``intervals_emitted``: candidates that became explanation rules.

Interpret ``emission_reason`` as:

- ``emitted``: interval produced a rule.
- ``removed_guard``: non-conforming interval removed by the guard.
- ``design_excluded``: interval not eligible in the current mode.
  Example: non-factual bins in factual mode, or factual bins in alternative mode.
- ``baseline_equal``: alternative candidate did not change
  ``(predict, low, high)`` versus baseline.
- ``zero_impact``: factual candidate had no effect on the prediction.
- ``ignored_feature``: feature excluded via ``features_to_ignore``.

## Next steps

- Read the {doc}`../foundations/concepts/guarded_explanations` concept guide for
  a deeper understanding of how the in-distribution guard works.
- Explore the {doc}`../foundations/concepts/alternatives` guide for the full
  alternatives and ensured framework.
- See the {doc}`../foundations/how-to/interpret_explanations` guide to interpret
  factual, alternative, and interval outputs.
