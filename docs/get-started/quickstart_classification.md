# Classification quickstart

This walkthrough demonstrates how to train a classifier, calibrate it, and
generate factual explanations with telemetry metadata.

## Prerequisites

```bash
pip install calibrated-explanations scikit-learn
```

```{admonition} Tested environments
:class: tip

The docs smoke suite executes this quickstart via
``pytest tests/docs/test_quickstarts.py::test_classification_quickstart`` on
CPython 3.8–3.11 for Linux runners. Match those versions locally to mirror the
validated environment.
```

## 1. Load data and split sets

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Binary classification dataset (malignant vs benign tumours)
dataset = load_breast_cancer()
X = dataset.data
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=0
)
X_proper, X_cal, y_proper, y_cal = train_test_split(
    X_train, y_train, test_size=0.25, stratify=y_train, random_state=0
)
```

## 2. Fit and calibrate the explainer

```python
from sklearn.ensemble import RandomForestClassifier
from calibrated_explanations import WrapCalibratedExplainer, __version__

print(f"calibrated_explanations {__version__}")

explainer = WrapCalibratedExplainer(RandomForestClassifier(random_state=0))
explainer.fit(X_proper, y_proper)
explainer.calibrate(X_cal, y_cal, feature_names=dataset.feature_names)
```

## 3. Generate factual explanations

```python
batch = explainer.explain_factual(X_test[:5])
print(batch[0])  # first explanation with rule details

telemetry = getattr(batch, "telemetry", {})
print("Telemetry keys:", sorted(telemetry))
print("Interval source:", telemetry.get("interval_source"))
```

The `telemetry` dictionary records which explanation, interval, and plot
strategies executed, together with preprocessing metadata. This payload mirrors
what `explainer.runtime_telemetry` returns.

```{admonition} Learn how to interpret the outputs
:class: seealso

Once you have the payload, read {doc}`../how-to/interpret_explanations`, paying
special attention to the sections on factual rule tables, alternative scenarios,
and PlotSpec visuals. They explain exactly how to interpret the outputs printed
above.
```

## 4. Explore alternatives or fast explanations

Switch modes by calling `explore_alternatives` or `explain_fast`:

```python
alternatives = explainer.explore_alternatives(X_test[:2])
fast = explainer.explain_fast(X_test[:2])
```

Both payloads inherit telemetry and rule metadata so downstream services can
track which plugins ran.

## Troubleshooting tips

- Ensure the dataset is numeric and finite before calibration. Apply an
  imputer if you introduce NaNs during preprocessing.
- A `NotFittedError` indicates either `fit` or `calibrate` was skipped.
- Use `explainer.runtime_telemetry` to confirm fast explanations route through
  the intended interval and probability plugins.
