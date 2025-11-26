# Task API Comparison

This page compares the two primary workflows for creating calibrated explanations: the **wrapper API** (`WrapCalibratedExplainer`) and the **direct API** (`CalibratedExplainer`).

## When to use each API

| Scenario | Recommended API | Reason |
|--|--|--|
| Quick start with scikit-learn models | `WrapCalibratedExplainer` | Familiar fit/calibrate workflow |
| Full control over calibration | `CalibratedExplainer` | Direct access to all parameters |
| Preprocessing integration | `WrapCalibratedExplainer` | Built-in preprocessing support |
| Advanced customization | `CalibratedExplainer` | Maximum flexibility |

## Wrapper API (WrapCalibratedExplainer)

The wrapper provides a scikit-learn-style interface:

```python
from calibrated_explanations import WrapCalibratedExplainer

# Initialize
wrapper = WrapCalibratedExplainer(model)

# Fit and calibrate
wrapper.fit(X_train, y_train)
wrapper.calibrate(X_cal, y_cal)

# Generate explanations
factual = wrapper.explain_factual(X_test)
alternatives = wrapper.explore_alternatives(X_test)
```

**Key features:**
- Automatic mode detection (classification vs regression)
- Built-in preprocessing support
- Simplified workflow for common tasks
- Compatible with scikit-learn pipelines

## Direct API (CalibratedExplainer)

The direct API offers full control:

```python
from calibrated_explanations import CalibratedExplainer

# Fit model separately
model.fit(X_train, y_train)

# Initialize explainer with explicit mode
explainer = CalibratedExplainer(
    model,
    X_cal,
    y_cal,
    mode="classification",  # or "regression"
    feature_names=feature_names,
    categorical_features=categorical_features
)

# Generate explanations
factual = explainer.explain_factual(X_test)
alternatives = explainer.explore_alternatives(X_test)
```

**Key features:**
- Explicit control over all parameters
- Fine-grained calibration customization
- Direct access to internal state
- Suitable for research and advanced use cases

## Common parameters

Both APIs support the same explanation parameters:

| Parameter | Description | Example |
|--|--|--|
| `threshold` | Threshold for probabilistic regression | `threshold=1000` |
| `low_high_percentiles` | Uncertainty interval bounds | `low_high_percentiles=(5, 95)` |
| `bins` | Custom discretization bins | `bins=custom_bins` |

## Migration between APIs

You can convert between APIs:

```python
# Wrapper wrapping a CalibratedExplainer
wrapper = WrapCalibratedExplainer(existing_explainer)

# Access underlying explainer from wrapper
explainer = wrapper.explainer
```

See the [legacy API contract](../../improvement_docs/legacy_user_api_contract.md) for the complete stable API surface.
