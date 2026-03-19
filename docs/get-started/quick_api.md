# Quick API

This page is a compact method map for `WrapCalibratedExplainer`.

## Mode-specific semantics notes

### Classification

- **Calibration prerequisites**: fit on proper split and calibrate on held-out calibration split.
- **Mode-specific guarantees**: Venn-Abers calibrated class probabilities with intervals.
- **Assumptions**: exchangeability or calibration-deployment distribution match.
- **Explicit non-guarantees**: no guarantee under drift or regime shift.
- **Explanation-envelope limits**: feature-level intervals are model-response summaries, not causal claims.
- **Formal semantics**: {doc}`../foundations/concepts/calibrated_interval_semantics`.

### Percentile or interval regression

- **Calibration prerequisites**: fit regression model and calibrate CPS on held-out calibration split.
- **Mode-specific guarantees**: CPS percentile intervals for requested `low_high_percentiles`.
- **Assumptions**: exchangeability or calibration-deployment distribution match.
- **Explicit non-guarantees**: no guarantee under drift or fixed interval width across subpopulations.
- **Explanation-envelope limits**: interval effects on explanations summarize model behavior under perturbation.
- **Formal semantics**: {doc}`../foundations/concepts/calibrated_interval_semantics`.

### Probabilistic or thresholded regression

- **Calibration prerequisites**: fit regression model and calibrate before threshold queries.
- **Mode-specific guarantees**: threshold events use CPS with Venn-Abers calibrated probabilities.
- **Assumptions**: exchangeability or calibration-deployment distribution match.
- **Explicit non-guarantees**: no guarantee under drift and no causal guarantee from threshold probabilities.
- **Explanation-envelope limits**: feature-level probability shifts are model-response summaries.
- **Formal semantics**: {doc}`../foundations/concepts/calibrated_interval_semantics`.

## Core methods

```python
pred = explainer.predict(X_query)
pred, (low, high) = explainer.predict(X_query, uq_interval=True)
probs = explainer.predict_proba(X_query)
probs, (low, high) = explainer.predict_proba(X_query, uq_interval=True)
factual = explainer.explain_factual(X_query)
alternatives = explainer.explore_alternatives(X_query)
```

## Classification

```python
explainer.fit(x_proper, y_proper)
explainer.calibrate(x_cal, y_cal, feature_names=feature_names)
probs, (low, high) = explainer.predict_proba(X_sample, uq_interval=True)
```

## Percentile or interval regression

```python
pred, (low, high) = explainer.predict(
    X_sample,
    uq_interval=True,
    low_high_percentiles=(5, 95),
)
```

## Probabilistic or thresholded regression

```python
p = explainer.predict_proba(X_sample, threshold=120.0)
p, (plo, phi) = explainer.predict_proba(X_sample, uq_interval=True, threshold=120.0)
```

Entry-point tier: Tier 2.

