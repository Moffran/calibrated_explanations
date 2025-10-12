# Regression quickstart

This guide walks through training a regression model, calibrating it, and
inspecting probabilistic explanations.

## Prerequisites

```bash
pip install calibrated-explanations scikit-learn
```

```{admonition} Tested environments
:class: tip

Continuous integration runs this guide through
``pytest tests/docs/test_quickstarts.py::test_regression_quickstart`` on Linux
with CPython 3.8–3.11. Align your environment with those versions to reproduce
the validated setup.
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

## 3. Produce factual and probabilistic explanations

```python
factual = explainer.explain_factual(X_test[:3])
print(f"Prediction interval: {factual.prediction_interval[0]}")

probabilistic = explainer.explore_alternatives(
    X_test[:3], threshold=2.5
)
print(probabilistic[0])
```

Both batches carry telemetry payloads that capture interval and probability
sources:

```python
telemetry = getattr(probabilistic, "telemetry", {})
print("Telemetry fields:", telemetry.keys())
print("Probability source:", telemetry.get("proba_source"))
```

## 4. Plot with PlotSpec

Install the `viz` extra to render calibrated bar charts:

```bash
pip install "calibrated_explanations[viz]"
```

```python
probabilistic.plot(interval=True, sort_by="abs")
```

PlotSpec fallbacks are tracked in telemetry so you can confirm whether the
PlotSpec adapter or legacy renderer produced the chart.

## Troubleshooting tips

- Regression quickstarts expect numeric arrays without NaNs. Use a
  `SimpleImputer` in your preprocessing pipeline if your dataset contains
  missing values.
- If plotting raises an import error, confirm the `viz` extra is installed and
  matplotlib is importable in the same environment.
- Use `explainer.runtime_telemetry` to confirm which interval calibrator (fast
  or default) executed for each batch.
