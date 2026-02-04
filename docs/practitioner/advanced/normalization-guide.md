# Normalization and Difficulty Estimation

For regression tasks, Calibrated Explanations can use **normalized residuals** and **difficulty estimation** to improve interval calibration, especially for heteroscedastic data (where prediction variance differs across instances).

## What is Difficulty Estimation?

Difficulty estimation adjusts prediction intervals based on how "difficult" each instance is to predict. Instances with higher variance in the underlying model get wider intervals, while easier-to-predict instances get narrower intervals.

**Without difficulty estimation**: All instances get the same interval width (homoscedastic assumption)

**With difficulty estimation**: Interval width scales with estimated prediction difficulty

## When to Use

Consider difficulty estimation when:

* **Heteroscedastic data**: Prediction errors vary systematically across the feature space
* **Mixed complexity**: Some regions of input space are inherently harder to predict
* **Instance-specific intervals**: You need intervals that reflect per-instance uncertainty
* **Residual patterns**: You observe that residuals have non-constant variance

```{admonition} Signs You Need Difficulty Estimation
:class: tip

* Residual plots show "fan" or "funnel" shapes
* Some subgroups have much larger prediction errors
* Interval coverage varies significantly across the feature space
* Simple conformal intervals are too wide for easy cases or too narrow for hard cases
```

## How It Works

The underlying `IntervalRegressor` from the `crepes` library supports several normalization strategies:

1. **Standard (no normalization)**: Residuals are used directly
2. **Sigma normalization**: Residuals are divided by an estimated standard deviation
3. **Difficulty-based**: Uses a secondary model to estimate per-instance difficulty

The difficulty estimator predicts how uncertain each instance is, and the conformal intervals are scaled accordingly.

## Configuration

Difficulty estimation is configured through the internal `IntervalRegressor` or `ConformalPredictiveSystem` when setting up the explainer.

### Basic Usage

For most users, the default CPS configuration handles difficulty automatically:

```python
from calibrated_explanations import WrapCalibratedExplainer

explainer = WrapCalibratedExplainer(model)
explainer.fit(x_proper, y_proper)
explainer.calibrate(x_cal, y_cal)

# Intervals are already calibrated with CPS
prediction, (low, high) = explainer.predict(
    x_test,
    uq_interval=True,
    low_high_percentiles=(5, 95)
)
```

### Advanced Configuration

For advanced users needing explicit difficulty estimation:

```python
from crepes import ConformalPredictiveSystem

# Create CPS with specific normalization
cps = ConformalPredictiveSystem()

# Fit with normalization model
# (Consult crepes documentation for full options)
cps.fit(
    residuals=y_cal - model.predict(x_cal),
    sigmas=difficulty_estimates  # Per-instance difficulty scores
)
```

## Trade-offs

| Aspect | Without Normalization | With Normalization |
| :--- | :--- | :--- |
| Interval width | Constant | Instance-specific |
| Complexity | Simpler | Requires difficulty model |
| Coverage | May vary by region | More uniform coverage |
| Computation | Faster | Slightly slower |

## Interpreting Normalized Intervals

When difficulty estimation is active:

* **Narrow intervals**: The model is confident about this instance
* **Wide intervals**: High estimated difficulty or sparse calibration region
* **Varying widths**: Expected behavior reflecting instance-specific uncertainty

```python
prediction, (low, high) = explainer.predict(x_test, uq_interval=True)

# Examine interval widths
widths = high - low

# Identify easy vs hard instances
easy_instances = widths < np.percentile(widths, 25)
hard_instances = widths > np.percentile(widths, 75)

print(f"Easy instances (narrow intervals): {easy_instances.sum()}")
print(f"Hard instances (wide intervals): {hard_instances.sum()}")
```

## Research Background

Difficulty estimation and normalized conformal prediction are documented in:

> Löfström, T., et al. (2025). Calibrated Explanations for Regression.
> Machine Learning 114, 100.
> [DOI: 10.1007/s10994-024-06642-8](https://link.springer.com/article/10.1007/s10994-024-06642-8)

The underlying conformal prediction methodology:

> Boström, H., et al. (2021). crepes: Conformal Regressors and Predictive Systems.
> [crepes documentation](https://crepes.readthedocs.io/)

## Cross-References

* {doc}`../../tasks/regression` - Regression task documentation
* {doc}`../../foundations/concepts/probabilistic_regression` - Probabilistic regression concepts
* {doc}`../../tasks/capabilities` - Full capability manifest
* {doc}`../../citing` - Citation information
