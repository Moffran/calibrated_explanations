# Classification quickstart

Run calibrated explanations for binary and multiclass classification without any
optional telemetry or plugins. The flow mirrors the README snippet and powers
the practitioner notebook set.

## Prerequisites

```bash
pip install calibrated-explanations scikit-learn
```

```{admonition} Tested environments
:class: tip

The docs smoke suite executes this quickstart via
``pytest tests/docs/test_quickstarts.py::test_classification_quickstart`` on
CPython 3.8–3.11 for Linux runners.
```

## 1. Load data and split sets

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Binary classification dataset (malignant vs benign tumours)
dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, stratify=y, random_state=0
)
x_proper, x_cal, y_proper, y_cal = train_test_split(
    x_train, y_train, test_size=0.25, stratify=y_train, random_state=0
)
```

> 🧪 **Multiclass variation:** Swap `load_breast_cancer` for `load_wine` to see
> calibrated explanations across three classes. The remaining steps stay
> identical.

## 2. Fit and calibrate the explainer

```python
from sklearn.ensemble import RandomForestClassifier
from calibrated_explanations import WrapCalibratedExplainer

explainer = WrapCalibratedExplainer(RandomForestClassifier(random_state=0))
explainer.fit(x_proper, y_proper)
explainer.calibrate(x_cal, y_cal, feature_names=dataset.feature_names)
```

## 3. Generate calibrated factual explanations

```python
factual = explainer.explain_factual(x_test[:5])
```

```{admonition} Sample output (truncated)
:class: hint

```text
Prediction [ Low ,  High]
0.077 [0.000, 0.083]
Value : Feature                                  Weight [ Low  ,  High ]
0.07  : mean concave points > 0.05               -0.418 [-0.576, -0.256]
0.15  : worst concave points > 0.12              -0.308 [-0.548,  0.077]
```

The first column shows the instance value, followed by the matching feature and
its calibrated contribution with uncertainty bounds.
```

Use the [interpretation guide](../foundations/how-to/interpret_explanations.md) to understand
how calibrated predictions, intervals, and rule tables translate into actions.

## 4. Explore calibrated alternatives

```python
alternatives = explainer.explore_alternatives(x_test[:2])
```

See the [Alternatives concept guide](../foundations/concepts/alternatives.md) for visual and
interpretation walkthroughs.

> 📝 **Citing calibrated explanations:** Reference {doc}`../citing` when you
> publish results using the binary, multiclass, probabilistic, or interval
> regression workflows showcased here.

> 🔬 **Research hub:** Visit {doc}`../researcher/index` for the flagship
> {doc}`../researcher/advanced/theory_and_literature` papers—especially the Expert Systems
> with Applications (2024) evaluation and the PMLR 230 (2024) multiclass
> tutorial—that document the datasets and calibration proofs underpinning this
> quickstart.
