# Classification (Binary & Multiclass)

Calibrated Explanations wraps standard scikit-learn classifiers to produce uncertainty-aware predictions and explanations.

## Classification semantics note

- **Calibration prerequisites**: fit on `x_proper, y_proper` and calibrate on held-out `x_cal, y_cal`.
- **Mode-specific guarantees**: Venn-Abers provides calibrated class probabilities with interval bounds.
- **Assumptions**: calibration and deployment data are exchangeable or distribution-matched.
- **Explicit non-guarantees**: no guarantee under drift or regime shift.
- **Explanation-envelope limits**: rule and feature-level intervals summarize model behavior, not causal effects.
- **Formal semantics**: {doc}`../foundations/concepts/calibrated_interval_semantics`.

## Supported signatures

| Method | Description |
| :--- | :--- |
| `predict(x)` | Point prediction (class label) |
| `predict_proba(x)` | Calibrated probabilities (n_samples, n_classes) |
| `predict_proba(x, uq_interval=True)` | Calibrated probabilities **plus** uncertainty intervals |
| `explain_factual(x)` | Factual explanation (rules + uncertainty) |
| `explore_alternatives(x)` | Alternative explanations (counterfactuals) |

> ℹ️ **Note:** `predict()` and `predict_proba()` match the scikit-learn API but return **calibrated** values.

## Examples

### 1. Calibrated probabilities with uncertainty

```python
# Predict probabilities for the positive class (binary) or all classes (multiclass)
probs, (low, high) = explainer.predict_proba(x_test, uq_interval=True)

print(f"Probability: {probs[0, 1]:.3f} (Interval: {low[0, 1]:.3f} – {high[0, 1]:.3f})")
```

### 2. Factual explanation

Returns the rules explaining *why* the model made this prediction, with epistemic and aleatoric explanation envelopes on the feature weights.

```python
explanation = explainer.explain_factual(x_test)
```

### 3. Explore alternatives

Finds what feature changes would be necessary to flip the prediction or increase confidence.

```python
alternatives = explainer.explore_alternatives(x_test)
```

## Multiclass Semantics: What is Explained

For multiclass classification, explanations are generated for the **predicted class** by default:

* **Factual explanations**: Show why the model predicted the specific class it chose
* **Alternative explanations**: Show what would change the prediction to a different class
* **Probability intervals**: Returned per-class via `predict_proba(x, uq_interval=True)`

```python
# Get probabilities for all classes with uncertainty
probs, (low, high) = explainer.predict_proba(x_test, uq_interval=True)
# probs.shape = (n_samples, n_classes)

# Explanation focuses on the argmax class by default
factual = explainer.explain_factual(x_test)
```

Set `multi_labels_enabled=True` to generate a **true multiclass explanation** that returns one explanation per class (all-classes view):

```python
multi_factual = explainer.explain_factual(x_test, multi_labels_enabled=True)
multi_alternative = explainer.explore_alternatives(x_test, multi_labels_enabled=True)
```

### Multi-label mode support matrix (`multi_labels_enabled=True`)

| Capability | Status | Notes |
| :--- | :--- | :--- |
| Per-class factual explanations | Supported | Returns one explanation per class for each instance |
| Per-class alternative explanations | Supported | Same container shape as factual mode |
| `predict_proba(..., uq_interval=True)` per-class intervals | Supported | Returns calibrated probability intervals per class |
| Collection operations (`add_conjunctions`, `reset`, filtering) | Supported | Applied across class-specific explanations |
| `reject_policy` orchestration | Supported | Reject handling applies in multiclass branch |
| `features_to_ignore` forwarding | Supported | Forwarded in multiclass explain paths |
| Narrative/dataframe export | Supported | Collection-level multiclass narrative/dataframe output |
| JSON export/import | Supported | `to_json` + multiclass-aware `from_json` restore grouped structure |
| Binary datasets with `multi_labels_enabled=True` | Limited | Allowed with compatibility warning; use default mode when all-classes output is unnecessary |

> ℹ️ **Note:** The rule table shows contributions toward the predicted class. Negative weights reduce confidence in the prediction; positive weights increase it.

## Key parameters

* **`bins`**: Supply Mondrian categories/bins for conditional calibration.
* **`uq_interval`**: Set to `True` to receive uncertainty intervals (tuples of lower/upper bounds).

Entry-point tier: Tier 2.

