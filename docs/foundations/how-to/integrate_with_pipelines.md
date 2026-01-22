# Integrate with scikit-learn pipelines

WrapCalibratedExplainer can manage preprocessing pipelines so calibration and
inference use the same transformations.

## Configure a preprocessing pipeline

```python
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from calibrated_explanations.api.config import ExplainerConfig
from calibrated_explanations import WrapCalibratedExplainer

dataset = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(
    dataset.data,
    dataset.target,
    test_size=0.2,
    stratify=dataset.target,
    random_state=0,
)
x_proper, x_cal, y_proper, y_cal = train_test_split(
    x_train,
    y_train,
    test_size=0.25,
    stratify=y_train,
    random_state=0,
)

numeric = [0, 1, 2]
preprocessor = ColumnTransformer(
    [
        (
            "num",
            Pipeline(
                steps=[
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]
            ),
            numeric,
        )
    ],
    remainder="drop",
)

config = ExplainerConfig(
    model=RandomForestClassifier(random_state=0),
    preprocessor=preprocessor,
)
explainer = WrapCalibratedExplainer.from_config(config)
```

When you call `fit` and `calibrate`, the wrapper fits both the underlying model
and the preprocessing pipeline.

```python
explainer.fit(x_proper, y_proper)
explainer.calibrate(x_cal, y_cal)
factual = explainer.explain_factual(x_test)
```

## Inspect telemetry snapshots

The runtime attaches preprocessing metadata to telemetry so you can audit which
transformers executed:

```python
telemetry = explainer.runtime_telemetry
pre = telemetry.get("preprocessor", {})
print(pre.get("identifier"))
print(pre.get("pipeline"))
```

Each entry includes the pipeline identifier, fitted attributes (when safe to
expose), and whether auto-encoding is enabled.

## Tips

- Keep preprocessing deterministic so calibration and inference remain aligned.
- Prefer column selectors over positional slicing when working with pandas
  DataFrames.
- When migrating to a public configuration API, replace `_from_config` with the
  official constructor provided by the release notes.

> **Telemetry note:** Runtime telemetry remains opt-in. Enable it only when governance teams need pipeline provenance, then follow :doc:`configure_telemetry` to capture preprocessing metadata for each batch.
