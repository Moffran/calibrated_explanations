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
X = dataset.data
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)
X_proper, X_cal, y_proper, y_cal = train_test_split(
    X_train, y_train, test_size=0.25, random_state=0
)
```

## 2. Fit and calibrate the explainer

```python
from sklearn.ensemble import RandomForestRegressor
from calibrated_explanations import WrapCalibratedExplainer

explainer = WrapCalibratedExplainer(RandomForestRegressor(random_state=0))
explainer.fit(X_proper, y_proper)
explainer.calibrate(
    X_cal,
    y_cal,
    feature_names=dataset.feature_names,
)
```

## 3. Retrieve calibrated intervals and probabilities

```python
factual = explainer.explain_factual(X_test[:3])
print(f"Prediction interval: {factual.prediction_interval[0]}")

probabilistic = explainer.predict(
    X_test[:1], threshold=150, uq_interval=True
)
print("Calibrated probability:", probabilistic[0])
```

> 🎯 **Interval regression insight:** The prediction interval remains the
> decision boundary for interval regression scenarios; probabilistic thresholds
> complement rather than replace it.

## 4. Explore calibrated alternatives

```python
alternatives = explainer.explore_alternatives(
    X_test[:2], threshold=150
)
```

{{ alternatives_triangular }}

{{ optional_extras_template }}
