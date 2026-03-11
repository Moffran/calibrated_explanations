---
title: Reject Policy Usage
description: How to drive per-call and explainer-level reject policies.
---

# Reject Policy Usage

This document captures the v0.10.3 guidance for opting into ADR-029’s reject integration via the new `RejectPolicy` enum. The goal is to preserve the legacy `reject=False` behavior (policy `NONE`) while allowing either per-call overrides or explainer-level defaults that implicitly enable reject orchestration when a non-`NONE` policy is selected.

## Overview

- **Policy enum:** All call-sites that want reject-aware behavior now accept a `reject_policy: RejectPolicy = RejectPolicy.NONE`. The enum includes `NONE`, `PREDICT_AND_FLAG`, `EXPLAIN_ALL`, `EXPLAIN_REJECTS`, `EXPLAIN_NON_REJECTS`, and `SKIP_ON_REJECT`. See `src/calibrated_explanations/core/reject/policy.py` for the list and docstrings.
- **Implicit orchestration:** Selecting any policy other than `NONE` automatically initializes and invokes `RejectOrchestrator`, even if the legacy `reject` flag was `False`. The return value becomes a `RejectResult` envelope (`src/calibrated_explanations/explanations/reject.py`) carrying predictions, explanations, reject status, policy, and metadata.

## Per-call policy overrides

Each explanation or prediction entry point supports the `reject_policy` keyword argument to vary integration behavior on a per-call basis:

```python
from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer
from calibrated_explanations.core.reject.policy import RejectPolicy

explainer = CalibratedExplainer(learner, x_cal, y_cal)

# Legacy behavior (returns CalibratedExplanations / prediction tuple)
result = explainer.explain_factual(x_test)

# Non-NONE policy returns a RejectResult envelope
envelope = explainer.explain_factual(x_test, reject_policy=RejectPolicy.EXPLAIN_NON_REJECTS)
assert envelope.policy == RejectPolicy.EXPLAIN_NON_REJECTS
```

- Per-call policies override any explainer default.
- Passing `RejectPolicy.NONE` (the default) keeps the original return type and skips reject orchestration entirely.

## Explainer-level defaults

To avoid specifying the same policy on every call, you can configure a default policy on the explainer itself:

```python
from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer
from calibrated_explanations.core.reject.policy import RejectPolicy

explainer = CalibratedExplainer(
    learner,
    x_cal,
    y_cal,
    default_reject_policy=RejectPolicy.PREDICT_AND_FLAG,
)

# Subsequent calls inherit the policy
envelope = explainer.predict(x_test)
assert envelope.policy == RejectPolicy.PREDICT_AND_FLAG
```

For the `WrapCalibratedExplainer`, pass `default_reject_policy` at calibration time only (the wrapper’s constructor intentionally omits this argument to keep defaults centralized):

```python
from calibrated_explanations.core.wrap_explainer import WrapCalibratedExplainer

wrapper = WrapCalibratedExplainer(learner)
wrapper.fit(X_fit, y_fit)
wrapper.calibrate(x_cal, y_cal, default_reject_policy=RejectPolicy.EXPLAIN_ALL)

envelope = wrapper.explain_factual(x_test)
assert envelope.policy == RejectPolicy.EXPLAIN_ALL
```

## Release note summary

- Added the `RejectPolicy` enum and `RejectResult` envelope so callers can opt into reject-aware outputs without changing existing defaults.
- Explanation and prediction entry points now accept `reject_policy`, with non-`NONE` selections implicitly enabling reject orchestration and returning a structured envelope.
- Explainer defaults (`default_reject_policy`) and wrapper calibration now offer reusable policy configuration, while per-call overrides continue to take precedence.

## NCF selection and the `w` parameter

The non-conformity function (NCF) controls how the reject learner scores each instance.
Pass `ncf` and `w` via `RejectPolicySpec` or `initialize_reject_learner`. When omitted,
the framework uses `ncf="default"` (task-dependent internal score).

Public NCF values are:
- `default`: internal score is `hinge` for binary + thresholded regression, `margin` for multiclass.
- `ensured`: blended score
  `score = (1 - w) * interval_width + w * default_score`.

Legacy `ncf="entropy"` is silently mapped to `ncf="default"` for compatibility.
Explicit `ncf="hinge"` and `ncf="margin"` are no longer supported.

The `w` parameter is operational only for `ensured`; for `default` it is accepted but ignored.

**Recommended `w` ranges per NCF:**

| NCF | Task | Safe `w` range | Starting point | Notes |
| --- | ---- | -------------- | -------------- | ----- |
| `default` | Binary / Multiclass / Regression¹ | — | — | Task-dependent internal score; `w` ignored |
| `ensured` | Binary / Multiclass / Regression¹ | 0.1–0.9 | 0.5 | Uses blending; requires `w > 0.0`; `w < 0.1` warns |

**Guard rails:**

- `w=0.0` with `ncf='ensured'` raises `ValidationError`.
- `w < 0.1` with `ncf='ensured'` emits a `UserWarning`.
- In multiclass mode, `default` uses internal margin scoring.

¹ **Regression requires a threshold.** The reject framework supports regression only when
a decision threshold is supplied to `initialize_reject_learner(threshold=t)`. The framework
converts regression into threshold-binarized conformal classification — it models
`P(y ≤ threshold)` and runs conformal prediction on that binary event. This is **not**
conformal prediction intervals. Omitting `threshold` for a regression explainer raises
`ValidationError`.

```python
from calibrated_explanations import RejectPolicySpec

# Safe starting configuration for multiclass:
spec = RejectPolicySpec.flag(ncf="default", w=0.5)

# Check which NCF was selected:
wrapper.initialize_reject_learner(ncf="default", w=0.4)
print(wrapper.explainer.reject_ncf)             # "default"
print(wrapper.explainer.reject_ncf_auto_selected)  # False
```

## Per-instance breakdowns

When a non-`NONE` policy is active the `RejectResult.metadata` dictionary contains per-instance keys that let you inspect the rejection breakdown without invoking the orchestrator directly. These keys are:

- `ambiguity_mask`: `numpy.ndarray[bool]` — True for instances whose prediction set contains more than one label (ambiguous).
- `novelty_mask`: `numpy.ndarray[bool]` — True for instances whose prediction set is empty (novelty).
- `prediction_set_size`: `numpy.ndarray[int]` — Integer size of the prediction set per instance.
- `epsilon`: `numpy.ndarray[float]` — Per-instance epsilon used when constructing the prediction set.

### Metadata audit semantics (hardened)

Reject-aware wrapped collections expose two denominator scopes:

- `raw_total_examples`: original collection size used by the unsliced reject computation (audit baseline).
- `sliced_total_examples`: current view length after slicing/indexing.

`raw_reject_counts` always stores canonical sums for the active view:

- `rejected`, `ambiguity_mask`, `novelty_mask` → sum of `True` entries.
- `prediction_set_size` → numeric sum of per-instance set sizes.

`metadata()` returns a lightweight aggregate view, `metadata_summary()` is an alias to that lightweight view, and `metadata_full()` includes JSON-safe per-instance arrays for the current view.

`RejectPolicySpec` supports canonical user-facing NCF values (`default`, `ensured`)
and is fully round-trippable via `to_dict()` / `from_dict()`. Legacy `entropy`
payloads are normalized to `default` on read.
For custom runtime callables, initialize the reject learner directly rather than
encoding callables in policy specs.

`resolve_policy_spec(...)` accepts multiple interoperable input forms:

- `RejectPolicySpec` objects,
- policy dict payloads from `to_dict()`,
- plain policy values (for legacy compatibility).

Short example:

```python
res = explainer.predict(x_test, reject_policy=RejectPolicy.PREDICT_AND_FLAG)
meta = res.metadata or {}
ambig = meta.get("ambiguity_mask")
nov = meta.get("novelty_mask")
sizes = meta.get("prediction_set_size")

print("Ambiguous count:", int(np.sum(ambig)) if ambig is not None else 0)
print("Novelty count:", int(np.sum(nov)) if nov is not None else 0)
print("Prediction set sizes sample:", sizes[:10] if sizes is not None else None)
```
