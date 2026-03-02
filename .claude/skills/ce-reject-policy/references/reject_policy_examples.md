# Reject Policy Code Examples

## Quick start — explain with FLAG policy

```python
from calibrated_explanations import WrapCalibratedExplainer
from calibrated_explanations.core.reject.policy import RejectPolicy

explainer = WrapCalibratedExplainer(model)
explainer.fit(x_proper, y_proper)
explainer.calibrate(x_cal, y_cal)

# Trigger reject orchestration on a per-call basis
result = explainer.explain_factual(x_test, reject_policy=RejectPolicy.FLAG)

print(f"Rejected: {result.rejected.sum()} / {len(result.rejected)}")
print(f"Rate: {result.metadata.get('reject_rate', 'N/A')}")

# Iterate explanations; skip rejected instances
for i, (exp, is_rejected) in enumerate(zip(result.explanation, result.rejected)):
    if not is_rejected:
        print(f"Instance {i}: {exp.prediction}")
    else:
        print(f"Instance {i}: REJECTED — deferred")
```

---

## Per-call policy override

All entry points support `reject_policy=` as a keyword argument:

```python
result = explainer.predict(x_test, reject_policy=RejectPolicy.ONLY_ACCEPTED)
result = explainer.predict_proba(x_test, reject_policy=RejectPolicy.FLAG)
result = explainer.explain_factual(x_test, reject_policy=RejectPolicy.ONLY_REJECTED)
result = explainer.explore_alternatives(x_test, reject_policy=RejectPolicy.FLAG)
```

Per-call policies override any explainer-level default.

---

## Explainer-level default (WrapCalibratedExplainer)

Set via `calibrate(default_reject_policy=...)`. **Not** via the constructor.

```python
explainer.calibrate(
    x_cal,
    y_cal,
    default_reject_policy=RejectPolicy.FLAG,
)

# All subsequent calls inherit FLAG policy by default
result = explainer.predict(x_test)           # returns RejectResult
result2 = explainer.explain_factual(x_test)  # also returns RejectResult

# Override per call if needed
plain = explainer.predict(x_test, reject_policy=RejectPolicy.NONE)  # returns plain array
```

---

## Low-level: `predict_reject`

`predict_reject` returns the conformal prediction SET and the reject decision
directly, without generating explanations.

```python
# Returns prediction set + reject mask at confidence=0.95
rejected, prediction_set = explainer.predict_reject(x_test, confidence=0.95)

# Lower confidence → narrower prediction sets → fewer rejections
rejected, prediction_set = explainer.predict_reject(x_test, confidence=0.80)
```

---

## Regression reject

For regression, the reject learner requires a threshold to be specified:

```python
from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer

# The reject threshold is set via initialize_reject_learner or at calibrate time
explainer.explainer.initialize_reject_learner(threshold=50.0)

result = explainer.predict_reject(x_test, confidence=0.95)
```

Without a threshold, calling `predict_reject` on a regression model raises
`ValidationError`.

---

## Inspecting rejection metadata

```python
result = explainer.predict(x_test, reject_policy=RejectPolicy.FLAG)
meta = result.metadata or {}

# Which instances are ambiguous (prediction set size > 1)?
ambiguity = meta.get("ambiguity_mask")    # boolean array

# Which are novel (outside the calibration distribution)?
novelty = meta.get("novelty_mask")        # boolean array

# Prediction set sizes (larger = more uncertain)
set_sizes = meta.get("prediction_set_size")  # integer array

# Error rate and reject rate
print(f"Error rate: {meta.get('error_rate', 'N/A')}")
print(f"Reject rate: {meta.get('reject_rate', 'N/A')}")
```

---

## Initialization failures

If the reject learner fails to initialize (e.g., insufficient calibration
data), `metadata["init_error"]` will be `True`:

```python
result = explainer.explain_factual(x_test, reject_policy=RejectPolicy.FLAG)

if result.metadata and result.metadata.get("init_error"):
    print("Reject learner init failed. Check calibration set size.")
    # Fall back: treat all instances as accepted
    fallback = explainer.explain_factual(x_test)
```

---

## Consuming results safely

```python
def consume_reject_result(result):
    """Safely consume a RejectResult or a plain CalibratedExplanations."""
    from calibrated_explanations.explanations.reject import RejectResult

    if isinstance(result, RejectResult):
        explanations = result.explanation   # may be None
        rejected = result.rejected          # bool array or None
    else:
        explanations = result               # legacy CalibratedExplanations
        rejected = None

    return explanations, rejected
```
