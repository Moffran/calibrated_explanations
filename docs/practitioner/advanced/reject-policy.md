# Reject Policy Guide

Reject policies control how the calibrated explanations runtime handles rejection
decisions when confidence or uncertainty thresholds no longer support the
requested output. The policy-driven API introduced around ADR-029 keeps the
legacy `reject=False` behaviour while optionally enabling reject orchestration.
For prediction entrypoints, reject-enabled calls return a structured
`RejectResult` envelope. For explanation entrypoints, reject-enabled calls
return reject-aware explanation collections carrying the same reject metadata.

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
if envelope.rejected is not None and envelope.rejected.any():
    # The runtime evaluated a reject decision even though the legacy
    # `reject` parameter remained False.
    print("Some instances triggered the reject policy.")
```

When `reject_policy` is left at its default (`RejectPolicy.NONE`) the call returns
the original prediction/explanation as before; no reject orchestration is performed.

## Reject-aware return types

When a reject policy is active (per-call or via `default_reject_policy`), return
shape depends on entrypoint:

- `predict` / `predict_proba`: returns `RejectResult`
- `explain_factual` / `explore_alternatives` / guarded explain variants: returns
  a reject-aware explanation collection (for example
  `RejectCalibratedExplanations` or `RejectAlternativeExplanations`)

Prediction envelopes include:

- `prediction`: optional prediction payload (present unless policy or fallback omits it)
- `explanation`: optional explanation payload (used in orchestration paths that request it)
- `rejected`: full-batch boolean reject mask
- `policy`: the `RejectPolicy` that generated this result
- `metadata`: supplementary telemetry, including contract keys listed below

Reject-aware explanation collections expose:

- `.explanations`: filtered explanation payload (policy-dependent)
- `.rejected`: policy-aligned reject mask for collection indexing safety
- `.metadata`: contract metadata including `source_indices` and `original_count`
- `.policy`: effective reject policy

Use `metadata["source_indices"]` to map explanation rows back to original input rows.

### Schema versioning (advanced)

The runtime now exposes strict v2 reject artifacts internally:

- `RejectDecisionArtifact`: decision diagnostics (mask/rates/epsilon/confidence)
- `RejectPayloadArtifact`: policy-filtered payload mapping (`source_indices`)
- `RejectResultV2`: versioned envelope (`schema_version="2.0"`)

Compatibility adapters keep existing callers working:

- `RejectResultV2.to_legacy()` converts v2 to legacy `RejectResult`
- `RejectResultV2.from_legacy(...)` (or `upgrade_reject_result(...)`) upgrades when
  required metadata is present

## WrapCalibratedExplainer example

The `WrapCalibratedExplainer` exposes the same two knobs (default + per-call). Pass
`default_reject_policy` to `calibrate`, and specify `reject_policy` on `predict`
or `explain`. Prediction calls return `RejectResult`; explanation calls return
reject-aware explanation collections.

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
if reject_result.rejected is not None and reject_result.rejected.any():
    print("The policy skipped processing on rejects.")
```

## NCF auto-selection

The reject learner uses a *non-conformity function* (NCF) to score how unusual each
instance is compared to the calibration set. You can specify the NCF explicitly via
`RejectPolicySpec` or `initialize_reject_learner(ncf=...)`, or let the framework
choose automatically.

Public choices are `default` and `ensured`:
- `default` is task-dependent internal scoring (`margin` for multiclass, `hinge` otherwise).
- `ensured` uses `score = (1 - w) * interval_width + w * default_score`.

When the NCF is auto-selected, `explainer.reject_ncf_auto_selected` is set to `True`
and `explainer.reject_ncf` records which NCF was chosen. You can read these attributes
to understand which NCF was used:

```python
wrapper.initialize_reject_learner()          # auto-selects based on task type
print(wrapper.explainer.reject_ncf)          # "default"
print(wrapper.explainer.reject_ncf_auto_selected)  # True
```

To override the auto-selection, pass `ncf` explicitly:

```python
from calibrated_explanations import RejectPolicySpec

spec = RejectPolicySpec.flag(ncf="default", w=0.5)
result = wrapper.predict(X_new, reject_policy=spec, confidence=0.95)
print(wrapper.explainer.reject_ncf)           # "default"
print(wrapper.explainer.reject_ncf_auto_selected)  # False
```

**Available NCFs and the `w` parameter:**

The `w` parameter is operational only for `ensured`:
`score = (1 - w) * interval_width + w * default_score`.
For `default`, `w` is accepted for API compatibility but ignored.

| NCF | Binary | Multiclass | Recommended `w` | Notes |
| --- | ------ | ---------- | --------------- | ----- |
| `default` | Yes | Yes | — | Internal hinge/margin by task; `w` ignored |
| `ensured` | Yes | Yes | 0.3–0.7 | Requires `w > 0.0`; use `w ≥ 0.1` |
>
> **w=0.0 guard:** Passing `w=0.0` with `ncf='ensured'` raises a `ValidationError`.
> Values `w < 0.1` with `ensured` emit a `UserWarning`.

## Regression and the reject framework

> **Important:** The reject framework supports regression **only when a decision threshold
> is provided**. Conformal prediction intervals for regression (lower/upper bounds on the
> target value) are a separate CE feature and are **not** available through the reject
> framework.

### Why a threshold is required

For classification, the reject learner works directly with calibrated class probabilities
(`predict_proba`). For regression there are no inherent class probabilities, so the
framework converts the problem into a binary event: *"will the target be below the
threshold?"* It then applies conformal prediction to that binary event.

Concretely, `initialize_reject_learner(threshold=t)` calls
`predict_probability(x, y_threshold=t)` to obtain calibrated probabilities
`P(y ≤ t)`, converts them to a binary matrix `[[1-p, p], ...]`, and fits a conformal
classifier on those scores. The NCF and rejection logic proceed exactly as for binary
classification.

If `threshold` is not provided for a regression explainer, a `ValidationError` is raised
immediately.

### Threshold tie behavior

Regression threshold binarization uses strict `< threshold` semantics on calibration
targets (`y_cal < threshold`). Values equal to `threshold` are treated as the
non-event class. This tie policy is deterministic and should be reflected in
downstream analysis.

### Regression usage example

```python
import numpy as np
from calibrated_explanations.core.wrap_explainer import WrapCalibratedExplainer
from calibrated_explanations.core.reject.policy import RejectPolicy

wrapper = WrapCalibratedExplainer(reg_model)
wrapper.fit(x_train, y_train)
wrapper.calibrate(x_cal, y_cal)

# Threshold is REQUIRED — choose a meaningful decision boundary
threshold = float(np.median(y_cal))
wrapper.initialize_reject_learner(threshold=threshold, ncf="default")

result = wrapper.predict(x_test, reject_policy=RejectPolicy.FLAG)
print(f"Reject rate: {result.metadata['reject_rate']:.2%}")
```

To use `ensured` NCF with regression:

```python
wrapper.initialize_reject_learner(threshold=threshold, ncf="ensured", w=0.5)
```

### What the threshold means

The threshold defines the binary question the conformal classifier answers. Choose it to
reflect the decision your application cares about — for example:

- Risk scoring: "Will the predicted cost exceed budget X?"
- Quality control: "Will the output metric fall below acceptable level Y?"
- Medical triage: "Will the predicted value be in the high-risk range (> Z)?"

Instances where the model is uncertain about the threshold crossing are rejected.
Instances where the model is confident (singleton prediction set for the binary event)
are accepted.

> **NCF auto-selection for regression:** When `ncf` is omitted, `default` is selected
> (internal hinge scoring on the binarized `[1-p, p]` representation).

## Policy selection advice

- Use `RejectPolicy.FLAG` when you want to process all instances and annotate which
  ones were rejected.
- Use `RejectPolicy.ONLY_REJECTED` when you need to focus resources on uncertain
  predictions.
- Use `RejectPolicy.ONLY_ACCEPTED` when you only want to process confident predictions.
- Keep `RejectPolicy.NONE` for fully backward compatible behaviour.

Always inspect `.policy` when consuming reject-aware outputs so the
calling application can differentiate fallback and short-circuit cases.

## ABI/API Guarantees for RejectResult (prediction entrypoints)

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

For all non-`NONE` policies, `metadata` is always present and contains at least
the required contract keys below.

| Key | Type | Description |
| --- | ---- | ----------- |
| `policy` | `str` | Effective reject policy name (`"flag"`, `"only_rejected"`, `"only_accepted"`) |
| `error_rate` | `float` | Estimated error rate on accepted samples (≥ 0.0; see `error_rate_defined`) |
| `error_rate_defined` | `bool` | `False` when no singleton prediction sets exist (error_rate is 0.0 sentinel, not a real estimate) |
| `reject_rate` | `float` | **Original-batch** proportion of rejected instances (`rejected_count / original_count`) |
| `accepted_count` | `int` | **Original-batch** accepted count |
| `rejected_count` | `int` | **Original-batch** rejected count |
| `ambiguity_rate` | `float` | Proportion of instances with ambiguous (multi-label) prediction sets |
| `novelty_rate` | `float` | Proportion of instances with empty prediction sets |
| `reject_ncf` | `str` | NCF used for this result (`"default"` or `"ensured"`) |
| `reject_ncf_w` | `float` | Effective/canonical NCF weight (operational for `ensured`) |
| `reject_ncf_auto_selected` | `bool` | `True` when the NCF was auto-selected (not specified by the caller) |
| `matched_count` | `int \| None` | Number of payload rows matched by `ONLY_REJECTED`/`ONLY_ACCEPTED` (`None` for `FLAG`) |
| `effective_confidence` | `float \| None` | Runtime confidence used for reject decisions |
| `effective_threshold` | `Any \| None` | Runtime threshold used for regression reject decisions |
| `source_indices` | `list[int]` | Source-row mapping from returned payload rows to original input rows |
| `original_count` | `int` | Number of rows in original input batch for this call |
| `init_ok` | `bool` | `True` when reject initialization completed for this call |
| `init_error` | `bool` | `True` when reject initialization failed |
| `fallback_used` | `bool` | `True` when any degraded/fallback path was used |
| `degraded_mode` | `tuple[str, ...]` | Deterministic list of degradation markers for this call |

Additionally, when a per-call reject policy is active the `metadata` dictionary
contains per-instance breakdowns that let you inspect ambiguity and
uncertainty without calling the orchestrator directly:

| Key | Type | Description |
|-----|------|-------------|
| `ambiguity_mask` | `numpy.ndarray[bool]` | `True` for instances with ambiguous (multi-label) prediction sets |
| `novelty_mask` | `numpy.ndarray[bool]` | `True` for instances with empty prediction sets (novelty) |
| `prediction_set_size` | `numpy.ndarray[int]` | Size of the prediction set for each instance |
| `epsilon` | `float` | Scalar epsilon threshold (`1 - confidence`) used for prediction-set construction |

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

Use `ONLY_ACCEPTED` when you only want explanations for confident predictions.
For explanation APIs the return object is a reject-aware explanation collection.

```python
from calibrated_explanations import WrapCalibratedExplainer
from calibrated_explanations.core.reject.policy import RejectPolicy

wrapper = WrapCalibratedExplainer(model)
wrapper.fit(x_train, y_train)
wrapper.calibrate(x_cal, y_cal)

# Only explain confident predictions
result = wrapper.explain_factual(X_new, reject_policy=RejectPolicy.ONLY_ACCEPTED)

if len(result.explanations) == 0:
    print("All instances were rejected - no explanations generated")
else:
    # Map explanation-local rows to original batch rows.
    for local_idx, expl in enumerate(result.explanations):
        global_idx = result.metadata["source_indices"][local_idx]
        print(f"Original instance {global_idx}: {expl}")
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
for local_idx, _expl in enumerate(result.explanations):
    global_idx = result.metadata["source_indices"][local_idx]
    review_queue.append({
        "index": int(global_idx),
        "needs_review": True,
    })

print(f"Review queue: {len(review_queue)} items need human review")
print(f"Reject rate: {result.metadata['reject_rate']:.2%}")
```

## Error Handling

### Detecting Initialization Failures

Use `init_ok`, `init_error`, and `fallback_used` together to distinguish hard
failure from successful-but-degraded execution.

```python
result = wrapper.explain_factual(X_new, reject_policy=RejectPolicy.FLAG)
meta = result.metadata or {}

if meta.get("init_error"):
    logging.error("Reject learner initialization failed")
    # Fall back to non-reject behavior or raise an error
    raise RuntimeError("Cannot proceed without reject learner")

if meta.get("fallback_used"):
    logging.warning("Reject fallback path used: %s", meta.get("degraded_mode", ()))
```

Contract-level fallback/coercion paths emit `RejectContractWarning`
(a `UserWarning` subclass), so existing `pytest.warns(UserWarning, ...)`
assertions remain valid.

### Reading per-instance breakdowns

When a reject policy is active you can inspect the masks and sizes directly:

```python
res = wrapper.predict(X_new, reject_policy=RejectPolicy.FLAG)
meta = res.metadata or {}
ambiguity = meta.get("ambiguity_mask")  # boolean array
novelty = meta.get("novelty_mask")  # boolean array
set_sizes = meta.get("prediction_set_size")  # integer array
eps = meta.get("epsilon")  # scalar float

# Example: indices that are ambiguous but not uncertain
ambiguous_only = np.where(ambiguity & ~novelty)[0]
print("Ambiguous-only indices:", ambiguous_only)
```

### Handling Empty Subsets

When using `ONLY_REJECTED` or `ONLY_ACCEPTED`, the explanation collection may
be empty if the relevant subset is empty:

```python
result = wrapper.explain_factual(X_new, reject_policy=RejectPolicy.ONLY_REJECTED)

if len(result.explanations) == 0:
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
