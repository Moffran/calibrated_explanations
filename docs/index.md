# Calibrated Explanations documentation

Welcome! Start here to run a calibrated explanation across classification and
regression, then follow the audience-specific hubs to go deeper.

## Run your first calibrated explanations

```bash
python -m pip install calibrated-explanations scikit-learn
```

```python
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from calibrated_explanations import WrapCalibratedExplainer

# Binary classification parity
breast_cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    breast_cancer.data,
    breast_cancer.target,
    test_size=0.2,
    stratify=breast_cancer.target,
    random_state=0,
)
explainer_cls = WrapCalibratedExplainer(RandomForestClassifier(random_state=0))
explainer_cls.fit(X_train, y_train)
explainer_cls.calibrate(X_train, y_train, feature_names=breast_cancer.feature_names)
factual_cls = explainer_cls.explain_factual(X_test[:1])
probabilities_cls, probability_interval_cls = explainer_cls.predict_proba(
    X_test[:1], uq_interval=True
)
low_cls, high_cls = probability_interval_cls

# Regression parity
boston = load_diabetes()
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    boston.data,
    boston.target,
    test_size=0.2,
    random_state=0,
)
explainer_reg = WrapCalibratedExplainer(RandomForestRegressor(random_state=0))
explainer_reg.fit(X_train_reg, y_train_reg)
explainer_reg.calibrate(X_train_reg, y_train_reg, feature_names=boston.feature_names)
factual_reg = explainer_reg.explain_factual(X_test_reg[:1])
prob_reg, (low_reg, high_reg) = explainer_reg.predict(X_test_reg[:1], uq_interval=True)

print(
    f"Classification probability: {probabilities_cls[0, 1]:.3f} "
    f"[{low_cls[0]:.3f}, {high_cls[0]:.3f}]"
)
print(factual_cls[0])
print(f"Regression prediction: {prob_reg[0]:.3f} [{low_reg[0]:.3f}, {high_reg[0]:.3f}]")
print(factual_reg[0])
```

Next, interpret the outputs with the
{doc}`foundations/how-to/interpret_explanations` guide and keep the
{doc}`citing` page handy when publishing results.

## Audience paths

- {doc}`get-started/index` – Quickstarts, installation, and troubleshooting.
- {doc}`practitioner/index` – Core practitioner journey plus advanced telemetry
  and performance references.
- {doc}`researcher/index` – Replication workflow and literature map.
- {doc}`contributor/index` – Plugin contract and contributor tooling.
- {doc}`foundations/index` – Shared concepts, how-to guides, references, and
  governance.
- {doc}`appendices/external_plugins` – Community plugin listings and governance
  notes.

```{toctree}
:maxdepth: 1
:caption: Get started

get-started/index
```

```{toctree}
:maxdepth: 1
:caption: Audience hubs

practitioner/index
researcher/index
contributor/index
```

```{toctree}
:maxdepth: 1
:caption: Shared foundations

foundations/index
plugins
```

```{toctree}
:maxdepth: 1
:caption: Engineering standards

standards/index
```

```{toctree}
:maxdepth: 1
:caption: Reference

api/index
```

```{toctree}
:maxdepth: 1
:caption: Upgrade guides

migration/index
```

```{toctree}
:maxdepth: 1
:caption: Appendices

appendices/external_plugins
appendices/changelog_links
maintenance/legacy-plotting-reference
citing
```

```{toctree}
:maxdepth: 1
:caption: Project Management

ROADMAP
improvement/index
```
