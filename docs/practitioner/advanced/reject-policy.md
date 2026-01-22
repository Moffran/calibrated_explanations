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
