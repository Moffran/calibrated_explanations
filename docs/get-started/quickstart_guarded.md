# Guarded explanations quickstart

Run in-distribution guarded explanations for classification.

## Prerequisites

```bash
pip install calibrated-explanations scikit-learn
```

## Classification guarded semantics note

- **Calibration prerequisites**: fit on proper split and calibrate on held-out calibration split.
- **Mode-specific guarantees**: guarded outputs are schema-compatible with CE helpers while filtering interval candidates through the shipped mixed guard rule.
- **Assumptions**: exchangeability or calibration-deployment distribution match.
- **Explicit non-guarantees**: no whole-interval guard certification, no metric identity with standard CE, no guarantee under drift, and no causal guarantee for suggested changes.
- **Explanation-envelope limits**: rule intervals remain model-response summaries.
- **Formal semantics**: {doc}`../foundations/concepts/calibrated_interval_semantics` and
  {doc}`../foundations/concepts/guarded_explanations`.

```{admonition} Fast explainers are not supported in guarded mode
:class: warning

Guarded mode requires a standard (non-fast) `CalibratedExplainer`.
Passing a fast explainer — one initialised with `CalibratedExplainer(..., fast=True)` or
retrieved via `WrapCalibratedExplainer` with fast mode enabled — raises a
`ConfigurationError` at the `explain_guarded_factual` / `explore_guarded_alternatives`
call site.  See ADR-032, decision 7 for the rationale.
```

For guarded regression workflows, start from {doc}`quickstart_regression`.
The only workflow difference is the explanation call:

- Use `explain_guarded_factual(...)` instead of `explain_factual(...)`.
- Use `explore_guarded_alternatives(...)` instead of `explore_alternatives(...)`.

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

## 2. Fit and calibrate

```python
from sklearn.ensemble import RandomForestClassifier
from calibrated_explanations import WrapCalibratedExplainer

explainer = WrapCalibratedExplainer(RandomForestClassifier(random_state=0))
explainer.fit(x_proper, y_proper)
explainer.calibrate(x_cal, y_cal, feature_names=dataset.feature_names)
```

## 3. Guarded factual and alternatives

```python
guarded_factual = explainer.explain_guarded_factual(x_test[:5], significance=0.1)
guarded_alts = explainer.explore_guarded_alternatives(x_test[:2], significance=0.1)
```

## 4. Audit guarded filtering

```python
audit = guarded_factual.get_guarded_audit()
```

Key audit fields:
- `intervals_conforming`: candidate intervals accepted by the shipped guard rule.
- `intervals_removed_guard`: candidate intervals rejected by the shipped guard rule.

Next steps:
- {doc}`../foundations/concepts/guarded_explanations`
- {doc}`../foundations/concepts/alternatives`
- {doc}`../foundations/how-to/interpret_explanations`

Entry-point tier: Tier 2.
