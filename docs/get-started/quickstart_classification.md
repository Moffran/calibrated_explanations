# Classification quickstart

Run calibrated explanations for binary or multiclass classification.

## Prerequisites

```bash
pip install calibrated-explanations scikit-learn
```

## Classification semantics note

- **Calibration prerequisites**: fit on `x_proper, y_proper` and calibrate on held-out `x_cal, y_cal`.
- **Mode-specific guarantees**: Venn-Abers provides calibrated probability intervals for class predictions.
- **Assumptions**: calibration and deployment data are exchangeable or distribution-matched.
- **Explicit non-guarantees**: no guarantee under drift or regime shift.
- **Explanation-envelope limits**: feature-level intervals are model-behavior summaries, not causal guarantees.
- **Formal semantics**: {doc}`../foundations/concepts/calibrated_interval_semantics`.

## 1. Load data and split sets

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

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

For multiclass all-class explanations:

```python
multi_factual = explainer.explain_factual(x_test[:5], multi_labels_enabled=True)
```

## 4. Explore calibrated alternatives

```python
alternatives = explainer.explore_alternatives(x_test[:2])
```

Next steps:
- {doc}`../foundations/how-to/interpret_explanations`
- {doc}`../foundations/concepts/alternatives`
- {doc}`../citing`

Entry-point tier: Tier 2.
