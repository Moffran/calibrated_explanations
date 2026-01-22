# Regression quickstart

Calibrate regression models with probabilistic thresholds and interval guidance
without enabling optional tooling. This quickstart mirrors the README flow and
feeds the practitioner notebooks.

## Prerequisites

```bash
pip install calibrated-explanations scikit-learn
```

```{admonition} Tested environments
:class: tip

Continuous integration runs this guide through
``pytest tests/docs/test_quickstarts.py::test_regression_quickstart`` on Linux
with CPython 3.8–3.11.
```

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
explainer.calibrate(
    x_cal,
    y_cal,
    feature_names=dataset.feature_names,
)
```

## 3. Retrieve calibrated intervals and probabilities

```python
factual = explainer.explain_factual(x_test[:3])

probabilities, probability_interval = explainer.predict_proba(
    x_test[:1], threshold=150, uq_interval=True
)
```

> 🎯 **Interval regression insight:** The prediction interval remains the
> decision boundary for interval regression scenarios; probabilistic thresholds
> complement rather than replace it.

## 4. Explore calibrated alternatives

```python
alternatives = explainer.explore_alternatives(
    x_test[:2], threshold=150
)
```

See the [Alternatives concept guide](../foundations/concepts/alternatives.md) for visual and
interpretation walkthroughs.

> 📘 **Deep dive:** The {doc}`../foundations/concepts/probabilistic_regression` guide explains
> how calibrated probabilities and intervals stay aligned across regression
> tasks.

> 📝 **Citing calibrated explanations:** When publishing regression or
> probabilistic threshold results, use {doc}`../citing` to pick the appropriate
> references.

> 🔬 **Research hub:** Cross-check {doc}`../researcher/index` for detailed
> replication notes on the Machine Learning (2025) regression study and the Fast
> Calibrated Explanations (2024) performance paper catalogued in
> {doc}`../researcher/advanced/theory_and_literature`—both reinforce the interval and
> probabilistic workflows demonstrated here.
