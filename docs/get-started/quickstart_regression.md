# Regression quickstart

Run interval regression and probabilistic regression from one calibrated explainer.

## Prerequisites

```bash
pip install calibrated-explanations scikit-learn
```

## Interval regression semantics note

- **Calibration prerequisites**: fit on `x_proper, y_proper` and calibrate on held-out `x_cal, y_cal`.
- **Mode-specific guarantees**: percentile intervals use CPS for requested percentile bounds.
- **Assumptions**: calibration and deployment data are exchangeable or distribution-matched.
- **Explicit non-guarantees**: no guarantee under drift or fixed interval width across subpopulations.
- **Explanation-envelope limits**: feature-level intervals summarize model behavior under perturbation.
- **Formal semantics**: {doc}`../foundations/concepts/calibrated_interval_semantics`.

## Probabilistic or thresholded regression semantics note

- **Calibration prerequisites**: fit on `x_proper, y_proper` and calibrate on held-out `x_cal, y_cal`.
- **Mode-specific guarantees**: threshold queries use CPS with Venn-Abers for calibrated event probabilities.
- **Assumptions**: calibration and deployment data are exchangeable or distribution-matched.
- **Explicit non-guarantees**: no guarantee under drift, and no causal guarantee from threshold probabilities.
- **Explanation-envelope limits**: feature-level intervals summarize model behavior under perturbation.
- **Formal semantics**: {doc}`../foundations/concepts/calibrated_interval_semantics`.

## 1. Prepare the dataset

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

dataset = load_diabetes()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0
)
x_proper, x_cal, y_proper, y_cal = train_test_split(
    x_train, y_train, test_size=0.25, random_state=0
)
```

## 2. Fit and calibrate the explainer

```python
from sklearn.ensemble import RandomForestRegressor
from calibrated_explanations import WrapCalibratedExplainer

explainer = WrapCalibratedExplainer(RandomForestRegressor(random_state=0))
explainer.fit(x_proper, y_proper)
explainer.calibrate(x_cal, y_cal, feature_names=dataset.feature_names)
```

## 3. Interval and threshold predictions

```python
pred, (low, high) = explainer.predict(x_test[:3], uq_interval=True)
probabilities, probability_interval = explainer.predict_proba(
    x_test[:1], threshold=150, uq_interval=True
)
```

## 4. Explore alternatives

```python
alternatives = explainer.explore_alternatives(x_test[:2], threshold=150)
```

Next steps:
- {doc}`../foundations/concepts/probabilistic_regression`
- {doc}`../foundations/how-to/interpret_explanations`
- {doc}`../citing`

Entry-point tier: Tier 2.
