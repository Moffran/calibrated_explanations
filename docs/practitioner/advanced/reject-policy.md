# Reject Policy Guide

Reject policies control how the calibrated explanations runtime handles rejection
decisions when confidence or uncertainty thresholds no longer support the
requested output. The policy-driven API introduced around ADR-029 keeps the
legacy `reject=False` behaviour while optionally enabling reject orchestration
and returning a structured envelope that annotates the policy, rejection mask,
and metadata.

## RejectPolicy overview

The `RejectPolicy` enum in `calibrated_explanations.core.reject.policy` defines the
available strategies:

- `NONE`: Preserve legacy behaviour (no reject orchestration; the call returns the
  original prediction or explanation).
- `PREDICT_AND_FLAG`: Always compute predictions but attach a rejection flag to
  the `RejectResult`.
- `EXPLAIN_ALL`: Explain every instance while tagging its rejection status.
- `EXPLAIN_REJECTS`: Only explain the rejected instances and skip explanations for
  the rest.
- `EXPLAIN_NON_REJECTS`: Explain only the non-rejected instances.
- `SKIP_ON_REJECT`: Short-circuit prediction/explanation when any rejects occur.

Selecting any policy other than `NONE` implicitly enables reject orchestration; it
is equivalent to `reject=True` for that call or explainer, so you no longer need to
set the legacy `reject` flag explicitly.

## CalibratedExplainer configuration

Pass `default_reject_policy` to the explainer constructor to set a reusable default,
but you can still override the behaviour per-call with the `reject_policy` argument
on `predict_*` and `explain_*` entry points.

```python
from calibrated_explanations import CalibratedExplainer
from calibrated_explanations.core.reject.policy import RejectPolicy

explainer = CalibratedExplainer(
    model,
    X_cal,
    y_cal,
    default_reject_policy=RejectPolicy.EXPLAIN_ALL,
)

envelope = explainer.explain_factual(
    X_test,
    reject_policy=RejectPolicy.PREDICT_AND_FLAG,
)

assert envelope.policy == RejectPolicy.PREDICT_AND_FLAG
if envelope.rejected:
    # The runtime evaluated a reject decision even though the legacy
    # `reject` parameter remained False.
    print("Some instances triggered the reject policy.")
```

When `reject_policy` is left at its default (`RejectPolicy.NONE`) the call returns
the original prediction/explanation as before; no reject orchestration is performed.

## RejectResult envelope

When a reject policy is active (per-call or via `default_reject_policy`) the runtime
returns a `RejectResult` from `calibrated_explanations.explanations.reject`. The
envelope includes:

- `prediction`: optional prediction payload (present unless the policy skips
  predictions).
- `explanation`: optional explanation collection (omitted when the policy skips them).
- `rejected`: optional boolean mask or flag showing which inputs were rejected.
- `policy`: the `RejectPolicy` that generated this result.
- `metadata`: dictionary of supplementary telemetry (rate limits, thresholds, etc.).

Treat the envelope as the canonical return value when `reject_policy != RejectPolicy.NONE`;
legacy expectations remain unchanged when the policy is `NONE`.

## WrapCalibratedExplainer example

The `WrapCalibratedExplainer` exposes the same two knobs (default + per-call). Pass
`default_reject_policy` to `calibrate`, and specify `reject_policy` on `predict`
or `explain`. The result is again returned as a `RejectResult`.

```python
from calibrated_explanations.core.wrap_explainer import WrapCalibratedExplainer
from calibrated_explanations.core.reject.policy import RejectPolicy

wrapper = WrapCalibratedExplainer(model)
wrapper.calibrate(
    X_cal,
    y_cal,
    default_reject_policy=RejectPolicy.EXPLAIN_NON_REJECTS,
)

reject_result = wrapper.predict(
    X_new,
    reject_policy=RejectPolicy.SKIP_ON_REJECT,
)

assert reject_result.policy == RejectPolicy.SKIP_ON_REJECT
if reject_result.rejected:
    print("The policy skipped prediction/explanation on rejects.")
```

## Policy selection advice

- Use `RejectPolicy.PREDICT_AND_FLAG` when you want to keep the original output but
  annotate which instances were rejected.
- Use the `EXPLAIN_*` variants when you need to limit explanation cost to a subset.
- Use `SKIP_ON_REJECT` for quick-fail semantics when rejects indicate unacceptable risk.
- Keep `RejectPolicy.NONE` for fully backward compatible behaviour.

Always inspect `RejectResult.policy` when consuming reject-aware outputs so the
calling application can differentiate fallback and short-circuit cases.

## ABI/API Guarantees for RejectResult

The `RejectResult` dataclass provides a stable contract for reject-aware consumers.
These guarantees help you write robust production code that handles all scenarios.

### Field Presence Guarantees by Policy

| Policy | `prediction` | `explanation` | `rejected` | `metadata` |
|--------|-------------|--------------|-----------|-----------|
| `NONE` | `None` | `None` | `None` | `None` |
| `PREDICT_AND_FLAG` | Present | `None` | Present | Present |
| `EXPLAIN_ALL` | Present | Present | Present | Present |
| `EXPLAIN_REJECTS` | Present | Present or `None`* | Present | Present |
| `EXPLAIN_NON_REJECTS` | Present | Present or `None`* | Present | Present |
| `SKIP_ON_REJECT` | Present or `None`** | Present or `None`** | Present | Present |

\* `None` when the relevant subset (rejected or non-rejected) is empty.

\** `None` when all instances are rejected.

### Metadata Dictionary Contract

When `metadata` is not `None`, it contains the following keys:

| Key | Type | Description |
|-----|------|-------------|
| `error_rate` | `float` | Estimated error rate on accepted samples |
| `reject_rate` | `float` | Proportion of instances rejected |
| `init_error` | `bool` (optional) | Present and `True` only when reject learner initialization failed |

### Type Specifications

- `rejected`: `numpy.ndarray[bool]` or `None` - Boolean array where `True` indicates rejection
- `policy`: `RejectPolicy` - Always present, never `None`
- `metadata`: `dict[str, Any]` or `None`

### Backwards Compatibility

When `policy` is `NONE`, all other fields are `None`, preserving legacy behavior. Consumers
can check `if result.policy is RejectPolicy.NONE` to determine whether reject orchestration
was active.

## Policy Decision Matrix

Use this matrix to select the appropriate policy for your use case:

| Use Case | Recommended Policy | Rationale |
|----------|-------------------|-----------|
| Audit logging | `PREDICT_AND_FLAG` | Predict everything, log rejection status |
| Full transparency | `EXPLAIN_ALL` | Complete explanations with rejection annotations |
| Anomaly investigation | `EXPLAIN_REJECTS` | Focus resources on uncertain predictions |
| Conservative deployment | `EXPLAIN_NON_REJECTS` | Only explain confident predictions |
| Strict safety gating | `SKIP_ON_REJECT` | Fail fast when uncertainty is too high |
| Legacy compatibility | `NONE` | No reject orchestration |

## Reject Hardening in Practice

### Example 1: Production Deployment with Audit Logging

Use `PREDICT_AND_FLAG` to always generate predictions while tracking rejection events
for compliance and monitoring.

```python
from calibrated_explanations import WrapCalibratedExplainer
from calibrated_explanations.core.reject.policy import RejectPolicy
import logging

# Setup
wrapper = WrapCalibratedExplainer(model)
wrapper.fit(X_train, y_train)
wrapper.calibrate(X_cal, y_cal, default_reject_policy=RejectPolicy.PREDICT_AND_FLAG)

# Production inference
result = wrapper.predict(X_new)

# Log rejection events
if result.rejected is not None and result.rejected.any():
    rejected_indices = [i for i, r in enumerate(result.rejected) if r]
    logging.warning(
        f"Rejected {len(rejected_indices)} predictions: indices {rejected_indices}"
    )
    logging.info(f"Error rate: {result.metadata['error_rate']:.4f}")

# Use predictions regardless of rejection status
predictions = result.prediction
```

### Example 2: Conservative Mode with EXPLAIN_NON_REJECTS

Use `EXPLAIN_NON_REJECTS` when you only want to explain predictions the model is
confident about. Rejected instances get predictions but no explanations.

```python
from calibrated_explanations import WrapCalibratedExplainer
from calibrated_explanations.core.reject.policy import RejectPolicy

wrapper = WrapCalibratedExplainer(model)
wrapper.fit(X_train, y_train)
wrapper.calibrate(X_cal, y_cal)

# Only explain confident predictions
result = wrapper.explain_factual(X_new, reject_policy=RejectPolicy.EXPLAIN_NON_REJECTS)

if result.explanation is not None:
    # Process explanations for confident predictions
    for i, expl in enumerate(result.explanation):
        if not result.rejected[i]:
            print(f"Instance {i}: {expl}")
else:
    print("All instances were rejected - no explanations generated")
```

### Example 3: Human-in-the-Loop with EXPLAIN_REJECTS

Use `EXPLAIN_REJECTS` to create a review queue of uncertain predictions that need
human oversight.

```python
from calibrated_explanations import WrapCalibratedExplainer
from calibrated_explanations.core.reject.policy import RejectPolicy

wrapper = WrapCalibratedExplainer(model)
wrapper.fit(X_train, y_train)
wrapper.calibrate(X_cal, y_cal)

# Generate explanations only for rejected (uncertain) instances
result = wrapper.explain_factual(X_new, reject_policy=RejectPolicy.EXPLAIN_REJECTS)

# Build review queue
review_queue = []
if result.rejected is not None:
    for i, is_rejected in enumerate(result.rejected):
        if is_rejected:
            review_queue.append({
                "index": i,
                "prediction": result.prediction[i] if result.prediction is not None else None,
                "needs_review": True,
            })

print(f"Review queue: {len(review_queue)} items need human review")
print(f"Reject rate: {result.metadata['reject_rate']:.2%}")
```

## Error Handling

### Detecting Initialization Failures

When the reject learner fails to initialize (e.g., missing calibration data), the
`metadata` dictionary contains `init_error: True`.

```python
result = wrapper.explain_factual(X_new, reject_policy=RejectPolicy.EXPLAIN_ALL)

if result.metadata and result.metadata.get("init_error"):
    logging.error("Reject learner initialization failed")
    # Fall back to non-reject behavior or raise an error
    raise RuntimeError("Cannot proceed without reject learner")
```

### Handling Empty Subsets

When using `EXPLAIN_REJECTS` or `EXPLAIN_NON_REJECTS`, the explanation may be `None`
if the relevant subset is empty:

```python
result = wrapper.explain_factual(X_new, reject_policy=RejectPolicy.EXPLAIN_REJECTS)

if result.explanation is None:
    # No rejected instances to explain
    print("All predictions are confident - nothing to review")
else:
    # Process rejected instance explanations
    pass
```

### Confidence Level Selection

The reject rate depends on the confidence level used during calibration. Higher
confidence levels result in more rejections:

| Confidence | Typical Reject Rate | Use When |
|------------|---------------------|----------|
| 0.90 | Lower | Acceptable to have some errors |
| 0.95 | Medium | Balanced tradeoff (default) |
| 0.99 | Higher | Strict accuracy requirements |

See `evaluation/reject_policy_ablation.py` for empirical comparisons of different
confidence levels on standard datasets.
