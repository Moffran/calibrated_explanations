# Integrate with scikit-learn pipelines

WrapCalibratedExplainer can manage preprocessing pipelines so calibration and
inference use the same transformations.

## Configure a preprocessing pipeline

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

from calibrated_explanations.api.config import ExplainerConfig
from calibrated_explanations import WrapCalibratedExplainer

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

config = ExplainerConfig(model=RandomForestClassifier(random_state=0), preprocessor=preprocessor)
explainer = WrapCalibratedExplainer._from_config(config)  # `_from_config` is private in 0.8.0
```

When you call `fit` and `calibrate`, the wrapper fits both the underlying model
and the preprocessing pipeline.

```python
explainer.fit(X_train, y_train)
explainer.calibrate(X_cal, y_cal)
factual = explainer.explain_factual(X_test)
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
