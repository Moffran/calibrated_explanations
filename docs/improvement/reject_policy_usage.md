---
title: Reject Policy Usage
description: How to drive per-call and explainer-level reject policies.
---

# Reject Policy Usage

This document captures the v0.10.2 guidance for opting into ADR-029’s reject integration via the new `RejectPolicy` enum. The goal is to preserve the legacy `reject=False` behavior (policy `NONE`) while allowing either per-call overrides or explainer-level defaults that implicitly enable reject orchestration when a non-`NONE` policy is selected.

## Overview

- **Policy enum:** All call-sites that want reject-aware behavior now accept a `reject_policy: RejectPolicy = RejectPolicy.NONE`. The enum includes `NONE`, `PREDICT_AND_FLAG`, `EXPLAIN_ALL`, `EXPLAIN_REJECTS`, `EXPLAIN_NON_REJECTS`, and `SKIP_ON_REJECT`. See `src/calibrated_explanations/core/reject/policy.py` for the list and docstrings.
- **Implicit orchestration:** Selecting any policy other than `NONE` automatically initializes and invokes `RejectOrchestrator`, even if the legacy `reject` flag was `False`. The return value becomes a `RejectResult` envelope (`src/calibrated_explanations/explanations/reject.py`) carrying predictions, explanations, reject status, policy, and metadata.

## Per-call policy overrides

Each explanation or prediction entry point supports the `reject_policy` keyword argument to vary integration behavior on a per-call basis:

```python
from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer
from calibrated_explanations.core.reject.policy import RejectPolicy

explainer = CalibratedExplainer(learner, X_cal, y_cal)

# Legacy behavior (returns CalibratedExplanations / prediction tuple)
result = explainer.explain_factual(X_test)

# Non-NONE policy returns a RejectResult envelope
envelope = explainer.explain_factual(X_test, reject_policy=RejectPolicy.EXPLAIN_NON_REJECTS)
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
    X_cal,
    y_cal,
    default_reject_policy=RejectPolicy.PREDICT_AND_FLAG,
)

# Subsequent calls inherit the policy
envelope = explainer.predict(X_test)
assert envelope.policy == RejectPolicy.PREDICT_AND_FLAG
```

For the `WrapCalibratedExplainer`, pass `default_reject_policy` at calibration time only (the wrapper’s constructor intentionally omits this argument to keep defaults centralized):

```python
from calibrated_explanations.core.wrap_explainer import WrapCalibratedExplainer

wrapper = WrapCalibratedExplainer(learner)
wrapper.fit(X_fit, y_fit)
wrapper.calibrate(X_cal, y_cal, default_reject_policy=RejectPolicy.EXPLAIN_ALL)

envelope = wrapper.explain_factual(X_test)
assert envelope.policy == RejectPolicy.EXPLAIN_ALL
```

## Release note summary

- Added the `RejectPolicy` enum and `RejectResult` envelope so callers can opt into reject-aware outputs without changing existing defaults.
- Explanation and prediction entry points now accept `reject_policy`, with non-`NONE` selections implicitly enabling reject orchestration and returning a structured envelope.
- Explainer defaults (`default_reject_policy`) and wrapper calibration now offer reusable policy configuration, while per-call overrides continue to take precedence.
