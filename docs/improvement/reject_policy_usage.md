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

## NCF selection and the `w` blending parameter

The non-conformity function (NCF) controls how the reject learner scores each instance.
Pass `ncf` and `w` via `RejectPolicySpec` or `initialize_reject_learner`. When omitted,
the framework auto-selects `hinge` (binary/regression) or `margin` (multiclass).

The `w` parameter blends the chosen NCF with `hinge`: `score = w * hinge + (1-w) * ncf`.
`w=1.0` gives pure hinge; `w=0.0` is forbidden for non-hinge NCFs.

**Recommended `w` ranges per NCF:**

| NCF | Task | Safe `w` range | Starting point | Notes |
| --- | ---- | -------------- | -------------- | ----- |
| `hinge` | Binary / Regression | — | 1.0 | `w` is ignored; always pure hinge |
| `margin` | Binary / Multiclass | 0.3–0.9 | 0.5 | Safe default for multiclass |
| `entropy` | Binary / Multiclass | 0.3–0.7 | 0.5 | Sensitive to probability spread; avoid extremes |
| `ensured` | Binary / Multiclass | 0.1–0.9 | 0.5 | Requires `w > 0.0`; `w < 0.1` warns |

**Guard rails:**

- `w=0.0` with a non-hinge NCF raises `ValidationError` (class-independent scores, all instances rejected).
- `w < 0.1` with a non-hinge NCF emits a `UserWarning`.
- In multiclass mode, `entropy` and `margin` operate on a binarized `[1-p_argmax, p_argmax]`
  representation, not the full K-class distribution. A `UserWarning` is emitted automatically.

```python
from calibrated_explanations import RejectPolicySpec

# Safe starting configuration for multiclass:
spec = RejectPolicySpec.flag(ncf="margin", w=0.5)

# Specify entropy with a conservative w:
spec = RejectPolicySpec.flag(ncf="entropy", w=0.4)

# Check which NCF was selected:
wrapper.initialize_reject_learner(ncf="entropy", w=0.4)
print(wrapper.explainer.reject_ncf)             # "entropy"
print(wrapper.explainer.reject_ncf_auto_selected)  # False
```

## Per-instance breakdowns

When a non-`NONE` policy is active the `RejectResult.metadata` dictionary contains per-instance keys that let you inspect the rejection breakdown without invoking the orchestrator directly. These keys are:

- `ambiguity_mask`: `numpy.ndarray[bool]` — True for instances whose prediction set contains more than one label (ambiguous).
- `novelty_mask`: `numpy.ndarray[bool]` — True for instances whose prediction set is empty (novelty).
- `prediction_set_size`: `numpy.ndarray[int]` — Integer size of the prediction set per instance.
- `epsilon`: `numpy.ndarray[float]` — Per-instance epsilon used when constructing the prediction set.

Short example:

```python
res = explainer.predict(x_test, reject_policy=RejectPolicy.PREDICT_AND_FLAG)
meta = res.metadata or {}
ambig = meta.get("ambiguity_mask")
nov = meta.get("novelty_mask")
sizes = meta.get("prediction_set_size")

print("Ambiguous count:", int(np.sum(ambig)) if ambig is not None else 0)
print("Uncertain count:", int(np.sum(unc)) if unc is not None else 0)
print("Prediction set sizes sample:", sizes[:10] if sizes is not None else None)
```
