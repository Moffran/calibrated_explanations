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
- `FLAG`: Process all instances while tagging their rejection status.
- `ONLY_REJECTED`: Only process the rejected instances and skip processing for
  the rest.
- `ONLY_ACCEPTED`: Process only the non-rejected (accepted) instances.

Selecting any policy other than `NONE` implicitly enables reject orchestration; it
is equivalent to `reject=True` for that call or explainer, so you no longer need to
set the legacy `reject` flag explicitly.

### Deprecated Policies

The following policy names are deprecated and will be removed in v1.0.0:

| Deprecated | New Name | Notes |
|------------|----------|-------|
| `PREDICT_AND_FLAG` | `FLAG` | Use `FLAG` instead |
| `EXPLAIN_ALL` | `FLAG` | Use `FLAG` instead |
| `EXPLAIN_REJECTS` | `ONLY_REJECTED` | Use `ONLY_REJECTED` instead |
| `EXPLAIN_NON_REJECTS` | `ONLY_ACCEPTED` | Use `ONLY_ACCEPTED` instead |
| `SKIP_ON_REJECT` | `ONLY_ACCEPTED` | Use `ONLY_ACCEPTED` instead |

Using deprecated names will emit a `DeprecationWarning`.

## CalibratedExplainer configuration

Pass `default_reject_policy` to the explainer constructor to set a reusable default,
but you can still override the behaviour per-call with the `reject_policy` argument
on `predict_*` and `explain_*` entry points.

```python
from calibrated_explanations import CalibratedExplainer
from calibrated_explanations.core.reject.policy import RejectPolicy

explainer = CalibratedExplainer(
    model,
    x_cal,
    y_cal,
    default_reject_policy=RejectPolicy.FLAG,
)

envelope = explainer.explain_factual(
    x_test,
    reject_policy=RejectPolicy.FLAG,
)

assert envelope.policy == RejectPolicy.FLAG
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
    x_cal,
    y_cal,
    default_reject_policy=RejectPolicy.ONLY_ACCEPTED,
)

reject_result = wrapper.predict(
    X_new,
    reject_policy=RejectPolicy.ONLY_ACCEPTED,
)

assert reject_result.policy == RejectPolicy.ONLY_ACCEPTED
if reject_result.rejected:
    print("The policy skipped processing on rejects.")
```

## NCF auto-selection

The reject learner uses a *non-conformity function* (NCF) to score how unusual each
instance is compared to the calibration set. You can specify the NCF explicitly via
`RejectPolicySpec` or `initialize_reject_learner(ncf=...)`, or let the framework
choose automatically.

**Auto-selection rule:**

- `margin` is selected for **multiclass** classification (more than two classes).
- `hinge` is selected for **binary** classification and regression.

When the NCF is auto-selected, `explainer.reject_ncf_auto_selected` is set to `True`
and `explainer.reject_ncf` records which NCF was chosen. You can read these attributes
to understand which NCF was used:

```python
wrapper.initialize_reject_learner()          # auto-selects based on task type
print(wrapper.explainer.reject_ncf)          # e.g. "hinge" or "margin"
print(wrapper.explainer.reject_ncf_auto_selected)  # True
```

To override the auto-selection, pass `ncf` explicitly:

```python
from calibrated_explanations import RejectPolicySpec

spec = RejectPolicySpec.flag(ncf="entropy", w=0.5)
result = wrapper.predict(X_new, reject_policy=spec, confidence=0.95)
print(wrapper.explainer.reject_ncf)           # "entropy"
print(wrapper.explainer.reject_ncf_auto_selected)  # False
```

**Available NCFs and the `w` blending parameter:**

The `w` parameter blends the selected NCF with `hinge`. `w=1.0` uses pure hinge;
lower values increase the weight of the chosen NCF.

| NCF | Binary | Multiclass | Recommended `w` | Notes |
| --- | ------ | ---------- | --------------- | ----- |
| `hinge` | Yes | Yes (binarized) | 1.0 | Default for binary; `w` is ignored |
| `margin` | Yes | Yes (binarized) | 0.5 | Default for multiclass |
| `entropy` | Yes | Yes (binarized) | 0.3–0.7 | Sensitive to probability spread |
| `ensured` | Yes | Yes (binarized) | 0.3–0.7 | Requires `w > 0.0`; use `w ≥ 0.1` |

> **Multiclass note:** In multiclass mode, `entropy` and `margin` operate on a
> *binarized* `(n, 2)` probability matrix `[1 - p_argmax, p_argmax]`, not the full
> K-class distribution. Scores will differ from the standard full-class entropy or
> margin definition. A `UserWarning` is emitted when these NCFs are used with a
> multiclass explainer.
>
> **w=0.0 guard:** Passing `w=0.0` with any non-hinge NCF raises a `ValidationError`
> because it produces class-independent scores that reject every instance. Values
> `w < 0.1` emit a `UserWarning`.

## Policy selection advice

- Use `RejectPolicy.FLAG` when you want to process all instances and annotate which
  ones were rejected.
- Use `RejectPolicy.ONLY_REJECTED` when you need to focus resources on uncertain
  predictions.
- Use `RejectPolicy.ONLY_ACCEPTED` when you only want to process confident predictions.
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
| `FLAG` | Present | Present | Present | Present |
| `ONLY_REJECTED` | Present | Present or `None`* | Present | Present |
| `ONLY_ACCEPTED` | Present | Present or `None`* | Present | Present |

\* `None` when the relevant subset (rejected or accepted) is empty.

### Metadata Dictionary Contract

When `metadata` is not `None`, it contains the following keys:

| Key | Type | Description |
| --- | ---- | ----------- |
| `error_rate` | `float` | Estimated error rate on accepted samples (≥ 0.0; see `error_rate_defined`) |
| `error_rate_defined` | `bool` | `False` when no singleton prediction sets exist (error_rate is 0.0 sentinel, not a real estimate) |
| `reject_rate` | `float` | Proportion of instances rejected |
| `ambiguity_rate` | `float` | Proportion of instances with ambiguous (multi-label) prediction sets |
| `novelty_rate` | `float` | Proportion of instances with empty prediction sets |
| `reject_ncf` | `str` | NCF used for this result (e.g. `"hinge"`, `"entropy"`) |
| `reject_ncf_w` | `float` | Blend weight `w` for the NCF |
| `reject_ncf_auto_selected` | `bool` | `True` when the NCF was auto-selected (not specified by the caller) |
| `matched_count` | `int` | Number of instances matched by `ONLY_REJECTED` or `ONLY_ACCEPTED` (0 when subset is empty) |
| `init_error` | `bool` (optional) | Present and `True` only when reject learner initialization failed |

Additionally, when a per-call reject policy is active the `metadata` dictionary
contains per-instance breakdowns that let you inspect ambiguity and
uncertainty without calling the orchestrator directly:

| Key | Type | Description |
|-----|------|-------------|
| `ambiguity_mask` | `numpy.ndarray[bool]` | `True` for instances with ambiguous (multi-label) prediction sets |
| `novelty_mask` | `numpy.ndarray[bool]` | `True` for instances with empty prediction sets (novelty) |
| `prediction_set_size` | `numpy.ndarray[int]` | Size of the prediction set for each instance |
| `epsilon` | `numpy.ndarray[float]` | Per-instance epsilon threshold used when constructing the prediction set |

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
| Audit logging | `FLAG` | Process everything, log rejection status |
| Full transparency | `FLAG` | Complete explanations with rejection annotations |
| Anomaly investigation | `ONLY_REJECTED` | Focus resources on uncertain predictions |
| Conservative deployment | `ONLY_ACCEPTED` | Only process confident predictions |
| Legacy compatibility | `NONE` | No reject orchestration |

## Reject Hardening in Practice

### Example 1: Production Deployment with Audit Logging

Use `FLAG` to always generate predictions while tracking rejection events
for compliance and monitoring.

```python
from calibrated_explanations import WrapCalibratedExplainer
from calibrated_explanations.core.reject.policy import RejectPolicy
import logging

# Setup
wrapper = WrapCalibratedExplainer(model)
wrapper.fit(x_train, y_train)
wrapper.calibrate(x_cal, y_cal, default_reject_policy=RejectPolicy.FLAG)

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

### Example 2: Conservative Mode with ONLY_ACCEPTED

Use `ONLY_ACCEPTED` when you only want to explain predictions the model is
confident about. Rejected instances get predictions but no explanations.

```python
from calibrated_explanations import WrapCalibratedExplainer
from calibrated_explanations.core.reject.policy import RejectPolicy

wrapper = WrapCalibratedExplainer(model)
wrapper.fit(x_train, y_train)
wrapper.calibrate(x_cal, y_cal)

# Only explain confident predictions
result = wrapper.explain_factual(X_new, reject_policy=RejectPolicy.ONLY_ACCEPTED)

if result.explanation is not None:
    # Process explanations for confident predictions
    for i, expl in enumerate(result.explanation):
        if not result.rejected[i]:
            print(f"Instance {i}: {expl}")
else:
    print("All instances were rejected - no explanations generated")
```

### Example 3: Human-in-the-Loop with ONLY_REJECTED

Use `ONLY_REJECTED` to create a review queue of uncertain predictions that need
human oversight.

```python
from calibrated_explanations import WrapCalibratedExplainer
from calibrated_explanations.core.reject.policy import RejectPolicy

wrapper = WrapCalibratedExplainer(model)
wrapper.fit(x_train, y_train)
wrapper.calibrate(x_cal, y_cal)

# Generate explanations only for rejected (uncertain) instances
result = wrapper.explain_factual(X_new, reject_policy=RejectPolicy.ONLY_REJECTED)

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
result = wrapper.explain_factual(X_new, reject_policy=RejectPolicy.FLAG)

if result.metadata and result.metadata.get("init_error"):
    logging.error("Reject learner initialization failed")
    # Fall back to non-reject behavior or raise an error
    raise RuntimeError("Cannot proceed without reject learner")
```

### Reading per-instance breakdowns

When a reject policy is active you can inspect the masks and sizes directly:

```python
res = wrapper.predict(X_new, reject_policy=RejectPolicy.FLAG)
meta = res.metadata or {}
ambiguity = meta.get("ambiguity_mask")  # boolean array
novelty = meta.get("novelty_mask")  # boolean array
set_sizes = meta.get("prediction_set_size")  # integer array
eps = meta.get("epsilon")  # float array

# Example: indices that are ambiguous but not uncertain
ambiguous_only = np.where(ambiguity & ~uncertainty)[0]
print("Ambiguous-only indices:", ambiguous_only)
```

### Handling Empty Subsets

When using `ONLY_REJECTED` or `ONLY_ACCEPTED`, the explanation may be `None`
if the relevant subset is empty:

```python
result = wrapper.explain_factual(X_new, reject_policy=RejectPolicy.ONLY_REJECTED)

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
