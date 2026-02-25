---
name: ce-reject-policy
description: >
  Configure reject and defer decision policies and interpret RejectResult behavior
  in prediction and explanation flows.
---

# CE Reject Policy

You are configuring reject / defer logic for production decision pipelines.
Reject policies let the framework flag, skip, or selectively process
instances where the model's prediction is ambiguous, novel, or uncertain —
all without modifying the underlying model.

**ADR reference**: ADR-029 (Reject Integration Strategy).

---

## The four policies

```python
from calibrated_explanations.core.reject.policy import RejectPolicy
```

| Policy | Behaviour |
|---|---|
| `RejectPolicy.NONE` (default) | No reject logic. Returns the same type as without any policy (plain array or `CalibratedExplanations`). |
| `RejectPolicy.FLAG` | Process ALL instances. Annotate which ones were rejected in `RejectResult.rejected`. |
| `RejectPolicy.ONLY_REJECTED` | Process and return **only** rejected instances. Accepted instances are omitted. |
| `RejectPolicy.ONLY_ACCEPTED` | Process and return **only** accepted (non-rejected) instances. Rejected ones are omitted. |

**Legacy policy strings** (deprecated, emit `DeprecationWarning`):

| Old name | Maps to |
|---|---|
| `"predict_and_flag"` / `"explain_all"` | `FLAG` |
| `"explain_rejects"` | `ONLY_REJECTED` |
| `"explain_non_rejects"` / `"skip_on_reject"` | `ONLY_ACCEPTED` |

Use the enum members directly, not string values, to avoid deprecation warnings.

---

## RejectResult envelope

When any non-`NONE` policy is active, the return type changes from a plain
array / `CalibratedExplanations` to a `RejectResult`:

```python
from calibrated_explanations.explanations.reject import RejectResult

# result is a RejectResult
result.prediction   # calibrated predictions (array or None if policy skips them)
result.explanation  # CalibratedExplanations or None
result.rejected     # boolean mask: True = rejected, False = accepted
result.policy       # RejectPolicy member that generated this result
result.metadata     # dict with telemetry: error_rate, reject_rate, etc.
```

Consume the envelope explicitly — do not assume plain-array return type when
any reject policy could be active.

---

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

Use this when you only need the reject decision, not explanations.

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

## Policy selection guide

| When to use | Policy |
|---|---|
| Audit mode: flag all uncertain, explain everything | `FLAG` |
| Focus investigation on uncertain instances | `ONLY_REJECTED` |
| Production only-confident mode (skip uncertain) | `ONLY_ACCEPTED` |
| Legacy behavior / benchmarking | `NONE` |

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

---

## Out of Scope

- Mondrian group calibration (see `ce-mondrian-conditional`).
- Calibrated predictions without reject logic (see `ce-calibrated-predict`).
- Guarded factual explanations (`explain_guarded_factual` — legacy interface,
  use `explain_factual` + reject policy instead).

---

## Evaluation Checklist

- [ ] `RejectPolicy` enum member used (not deprecated string value).
- [ ] Return type checked: `RejectResult` when policy != `NONE`, plain type otherwise.
- [ ] `result.rejected` mask inspected for actual rejection counts.
- [ ] Initialization failure path tested (`metadata["init_error"]`).
- [ ] Per-call override tested alongside explainer-level default.
- [ ] Regression: `initialize_reject_learner(threshold=t)` called before `predict_reject`.
- [ ] Tests use `pytest.warns(UserWarning)` when fallback warning is expected.
