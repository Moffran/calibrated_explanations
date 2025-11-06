# Explanation Structures

This document provides a comprehensive overview of the internal data structures used by the `CalibratedExplanation` classes and their subclasses. Understanding these structures is crucial for developers working with or extending the calibrated explanations framework.

## Overview

Calibrated explanations build complex nested data structures to store rule information, prediction intervals, and feature weights. These structures are designed to support multiple explanation types (factual, alternative, fast) while maintaining consistency across different problem domains (classification, regression, probabilistic).

## Base Class: CalibratedExplanation

The `CalibratedExplanation` abstract base class defines the core interface and shared attributes. All explanation instances inherit these fundamental structures.

### Core Attributes

#### Instance Data
- **`calibrated_explanations`**: Reference to the parent `CalibratedExplanations` collection object
- **`index`**: Integer index of the instance being explained within the test dataset
- **`x_test`**: The original test dataset (array-like) containing all instances
- **`bin`**: Optional list containing the bin index of the instance (used for discretization)

#### Binned Data
- **`binned`**: Dictionary mapping feature indices to binned values for this instance. Each entry contains discretized feature values used in rule generation.

#### Feature-Level Predictions and Weights
- **`feature_weights`**: Dictionary containing feature weight predictions with keys:
  - `"predict"`: Point prediction of feature weight
  - `"low"`/`"high"`: Uncertainty interval bounds for feature weight
- **`feature_predict`**: Dictionary containing feature-level predictions with keys:
  - `"predict"`: Point prediction for the feature
  - `"low"`/`"high"`: Uncertainty interval bounds

#### Instance-Level Prediction
- **`prediction`**: Dictionary containing the overall model prediction for this instance with keys:
  - `"predict"`: Point prediction
  - `"low"`/`"high"`: Uncertainty interval bounds
  - `"classes"`: Class labels (for classification)
  - `"__full_probabilities__"`: Optional full probability matrix for multiclass classification

#### Threshold and Calibration
- **`y_threshold`**: Threshold value(s) for binary classification or regression explanations
- **`y_minmax`**: Min/max values from calibration data (computed from `explainer.y_cal`)

#### Rule Storage
- **`rules`**: Primary rule dictionary (populated by `_get_rules()`)
- **`conjunctive_rules`**: Dictionary for conjunctive rules (when `add_conjunctions()` is called)
- **`conditions`**: List of condition objects used in rule generation

#### State Flags
- **`_has_rules`**: Boolean indicating if rules have been computed
- **`_has_conjunctive_rules`**: Boolean indicating if conjunctive rules exist
- **`focus_columns`**: Optional list of feature indices to focus explanations on

#### Metadata
- **`explain_time`**: Optional timestamp of when explanation was generated

## FactualExplanation Structure

`FactualExplanation` extends `CalibratedExplanation` to provide factual explanations that highlight features contributing to the model's prediction.

### Rules Dictionary Structure

The `rules` dictionary (returned by `_get_rules()`) contains the following keys, each holding a list of values (one per rule):

#### Prediction Data
- **`base_predict`**: List with single element - the base prediction for the instance
- **`base_predict_low`/`base_predict_high`**: Uncertainty interval for base prediction

#### Feature Rule Data
- **`predict`**: Point predictions for each feature rule
- **`predict_low`/`predict_high`**: Uncertainty intervals for feature predictions
- **`weight`**: Feature weights (contribution to prediction)
- **`weight_low`/`weight_high`**: Uncertainty intervals for feature weights

#### Feature Information
- **`feature`**: Feature indices corresponding to each rule
- **`feature_value`**: Actual feature values from the instance
- **`sampled_values`**: Binned/discretized values used in rules
- **`value`**: String representation of feature values (rounded for display)

#### Rule Text
- **`rule`**: Human-readable rule strings (e.g., "feature_name = value")
- **`is_conjunctive`**: Boolean flags indicating conjunctive rules

#### Classification Data
- **`classes`**: Class labels for classification problems

### Usage in Factual Explanations

Factual rules explain "what is" - they describe how the current feature values contribute to the prediction. Each rule binds an observed feature value to a condition and exposes:
- The calibrated feature weight (importance)
- The uncertainty interval for that weight
- The feature's contribution to the overall prediction

## AlternativeExplanation Structure

`AlternativeExplanation` extends `CalibratedExplanation` to explore "what if" scenarios by showing how changes to feature values could alter predictions.

### Rules Dictionary Structure

Similar to `FactualExplanation`, but with key differences:

#### Prediction Data
- **`base_predict`**: Base prediction for the instance
- **`base_predict_low`/`base_predict_high`**: Base prediction uncertainty interval

#### Alternative Rule Data
- **`predict`**: Alternative predictions if feature value changes
- **`predict_low`/`predict_high`**: Uncertainty intervals for alternative predictions
- **`weight`**: Weight deltas (difference from base prediction)
- **`weight_low`/`weight_high`**: Uncertainty intervals for weight deltas

#### Feature Information
- **`feature`**: Feature indices
- **`feature_value`**: Original feature values from instance
- **`sampled_values`**: Alternative values being explored
- **`value`**: String representation of original values

#### Rule Text
- **`rule`**: Alternative conditions (e.g., "feature_name > threshold")
- **`is_conjunctive`**: Boolean flags for conjunctive rules

### Usage in Alternative Explanations

Alternative rules explore counterfactual scenarios. For each feature, they show:
- What the prediction would be if the feature had a different value
- The uncertainty interval for that alternative prediction
- How much the prediction changes (weight delta)

For categorical features, alternatives explore all other possible categories. For continuous features, alternatives examine values outside the current bin (less than lower bound or greater than upper bound).

## Rule Generation Process

### Common Steps
1. **Discretization**: Features are binned using the explainer's discretizer
2. **Condition Definition**: `_define_conditions()` creates rule text for each feature
3. **Rule Filtering**: Rules are filtered based on `features_to_ignore` and prediction differences

### Factual Rules
- Compare feature predictions vs. instance prediction
- Include rules where feature prediction differs from overall prediction
- Weight represents feature contribution

### Alternative Rules
- Generate counterfactual scenarios for each feature
- For categorical: Try all other category values
- For continuous: Try values below lower bin boundary and above upper bin boundary
- Weight represents prediction change (delta)

## Build Rules Payload

All explanation types implement `build_rules_payload()` which returns a structured dictionary separating core content from metadata:

```python
{
    "core": {
        "kind": "factual" | "alternative" | "fast",
        "prediction": {
            "value": float,
            "uncertainty_interval": [low, high]
        },
        "feature_rules": [
            {
                "weight": {"value": float, "uncertainty_interval": [low, high]},
                "condition": {...},
                "prediction": {...}  # alternative only
            }
        ]
    },
    "metadata": {
        "feature_rules": [...],  # extended metadata
        "baseline_prediction": {...},  # factual only
        ...
    }
}
```

This payload is used for JSON export, telemetry, and visualization.

## Performance Considerations

- Rules are computed lazily via `_get_rules()` and cached
- Conjunctive rules are computed separately and cached
- Large datasets may have many rules; consider `focus_columns` for targeted explanations
- Binned data is stored as `MappingProxyType` for immutability

## Extension Points

When extending explanation classes:
- Override `_get_rules()` to customize rule generation
- Implement `build_rules_payload()` for new payload structures
- Use existing helper methods for uncertainty intervals and telemetry formatting
- Maintain the core dictionary structure for compatibility
