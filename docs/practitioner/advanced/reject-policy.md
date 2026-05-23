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

### Removed legacy policy aliases

Older releases exposed alias names for the four policy modes. Those aliases have
now been removed; use the canonical enum members directly:

| Removed alias | Use instead | Notes |
|------------|----------|-------|
| `PREDICT_AND_FLAG` | `FLAG` | Use `FLAG` instead |
| `EXPLAIN_ALL` | `FLAG` | Use `FLAG` instead |
| `EXPLAIN_REJECTS` | `ONLY_REJECTED` | Use `ONLY_REJECTED` instead |
| `EXPLAIN_NON_REJECTS` | `ONLY_ACCEPTED` | Use `ONLY_ACCEPTED` instead |
| `SKIP_ON_REJECT` | `ONLY_ACCEPTED` | Use `ONLY_ACCEPTED` instead |

Passing old string values to `RejectPolicy(...)` raises `ValueError`; module-level
attribute access raises `AttributeError`.

## WrapCalibratedExplainer configuration

For CE-first application code, prefer `WrapCalibratedExplainer`. Pass
`default_reject_policy` to `calibrate(...)` to set a reusable default, and
override the behaviour per-call with the `reject_policy` argument on prediction
and explanation entry points.

```python
from calibrated_explanations import WrapCalibratedExplainer
from calibrated_explanations.core.reject.policy import RejectPolicy

wrapper = WrapCalibratedExplainer(model)
wrapper.fit(x_train, y_train)
wrapper.calibrate(x_cal, y_cal, default_reject_policy=RejectPolicy.FLAG)

envelope = wrapper.predict(
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
wrapper.fit(x_train, y_train)
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
wrapper.explainer.reject_orchestrator.initialize_reject_learner()
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

## Reject-option conformal classification semantics

Reject-option conformal classification in CE builds a conformal prediction set for each
instance and uses the set geometry to route outcomes:

- Singleton set (`|S(x)| = 1`): accepted (confident classification).
- Empty set (`|S(x)| = 0`): novelty reject (out-of-distribution / non-covered behavior).
- Multi-label set (`|S(x)| >= 2`): ambiguity reject (too many plausible classes).

At runtime this appears in metadata as:

- `reject_rate = ambiguity_rate + novelty_rate`
- `singleton_rate` (accepted)
- `prediction_set_size`, `ambiguity_mask`, `novelty_mask`

These semantics are the same whether you use `default`, `ensured`, or an experimental
reject scoring strategy.

## Difficulty in CE reject flows

CE currently has two distinct difficulty paths for classification reject workflows:

1. Existing behavior (indirect):
    `difficulty_estimator -> VennAbers probability scaling -> reject NCF`
2. Experimental behavior (direct reject normalization):
    `difficulty_estimator -> reject nonconformity normalization -> conformal p-values`

The existing built-in path already allows `difficulty_estimator` to influence calibrated
probabilities through Venn-Abers. The experimental path additionally normalizes reject
scores directly before conformal p-values and prediction sets are computed.

### Why normalize before p-values and sets

Difficulty normalization must happen at the nonconformity-score stage.

- Correct place: transform calibration/test nonconformity scores, then calibrate p-values.
- Incorrect place: keep scores unchanged and only move a final reject threshold post-hoc.

Post-hoc thresholding changes only the decision boundary and does not redefine the
conformal score distribution. In contrast, pre-p-value normalization changes the score
space used by conformal calibration itself.

## Difficulty Estimator Provenance (Experimental Strategy)

The experimental reject strategy `experimental.difficulty_normalized` modifies
the non-conformity scores directly by dividing by per-instance difficulty before
conformal p-values and prediction sets are computed.

This is different from post-hoc thresholding. Post-hoc thresholding changes only
the final decision boundary and does not change the conformal score distribution.
Difficulty-aware conformal reject scoring must act on the scores themselves.

## When to use which reject mode

Use this decision guide for classification tasks:

- `ncf="default"`:
    Best baseline. Use when you want conservative changes and current public behavior.
- `ncf="ensured"`:
    Use when you need interval-width-aware scoring and can tune `w`.
- `strategy="experimental.difficulty_normalized"`:
    Use for research/ablation runs when difficulty should directly shape reject scoring.
    Keep this opt-in and experimental.
- `strategy="experimental.ambiguity_normalized_novelty_penalized"`:
    Use only for research diagnostics when exploring ambiguity-vs-novelty routing tradeoffs.
    Treat as evaluation-only until promoted.

## Validity and methodology caveats

- Keep the difficulty estimator fixed before reject calibration alphas are estimated.
- Avoid fitting the estimator on calibration labels/residuals unless you apply
    explicit cross-fitting.
- Separate empirical utility from formal finite-sample coverage guarantees.
- Treat difficulty-normalized and novelty-penalized strategies as experimental evidence,
    not yet public contract expansion.

## Scenario 8-11 evaluation summary

The reject evaluation suite under `evaluation/reject/` reports:

- Scenario 8 (baseline indirect difficulty effect):
    enabling VA difficulty made reject stricter (`accept_rate` down; error capture up),
    with strong accepted-accuracy cost in this setup.
- Scenario 9 (direct difficulty-normalized scores):
    primary contrast A vs C showed that arm C selects harder instances for rejection
    (`difficulty_reject_auc` delta +0.1651 full-grid, driven by high-confidence rows).
    At matched reject-rate operating points (10–40% targets), accepted-accuracy delta
    was marginally negative (−0.0089 at target 0.40). The AUC advantage seen in the
    full confidence grid is a selection effect at high rejection rates (>40%), not a
    benefit in the deployment-relevant 10–40% range. Arm C remains the current
    experimental baseline but is not yet promoted to the public API.
- Scenario 10 (ambiguity-normalized novelty-penalized variant):
    novelty increase was small and did not clearly outperform arm C; C remains the
    simpler recommended experimental path.
- Scenario 11 (matched operating-point selection):
    matched target reject-rate evidence was mixed. C had accepted-accuracy deltas
    of +0.0012, -0.0029, -0.0070, and -0.0089 at targets 0.10, 0.20, 0.30, and
    0.40, so public API promotion is not justified yet. The novelty-aware variant
    should remain internal/experimental.

See scenario artifacts for exact metrics:

- `evaluation/reject/artifacts/scenario_8_difficulty_reject_ablation.md`
- `evaluation/reject/artifacts/scenario_9_difficulty_normalized_ncf.md`
- `evaluation/reject/artifacts/scenario_10_ambiguity_novelty_reject.md`
- `evaluation/reject/artifacts/scenario_11_operating_point_selection.md`

## Minimal CE-first examples

### Public baseline reject (`default` / `ensured`)

```python
from calibrated_explanations import RejectPolicySpec, WrapCalibratedExplainer

wrapper = WrapCalibratedExplainer(model)
wrapper.fit(x_train, y_train)
wrapper.calibrate(x_cal, y_cal, feature_names=feature_names)

policy = RejectPolicySpec.flag(ncf="default")
result = wrapper.predict(x_test, reject_policy=policy, confidence=0.95)
print(result.metadata["reject_rate"])
```

```python
policy = RejectPolicySpec.flag(ncf="ensured", w=0.5)
result = wrapper.predict(x_test, reject_policy=policy, confidence=0.95)
print(result.metadata["reject_ncf"], result.metadata["reject_ncf_w"])
```

### Experimental difficulty-normalized reject scoring

```python
from calibrated_explanations import RejectPolicySpec, WrapCalibratedExplainer

wrapper = WrapCalibratedExplainer(model)
wrapper.fit(x_train, y_train)
wrapper.calibrate(
    x_cal,
    y_cal,
    feature_names=feature_names,
    difficulty_estimator=difficulty_estimator,
)

policy = RejectPolicySpec.flag(ncf="default")
result = wrapper.predict(
    x_test,
    reject_policy=policy,
    confidence=0.95,
    strategy="experimental.difficulty_normalized",
)
print(result.metadata["reject_rate"])
print(result.metadata["difficulty_normalized"])
```

Keep this path explicitly experimental. It preserves the public `RejectPolicy`
contract, but the strategy identifier is not a promoted public NCF mode.
The experimental difficulty strategies require `difficulty_estimator` to be set
on the calibrated explainer; omitting it raises `ConfigurationError` instead of
silently falling back to the built-in reject score.

### Why provenance matters

Conformal validity depends on calibration/test exchangeability with respect to the
scoring pipeline. If a difficulty estimator is fitted using calibration labels or
calibration residuals without cross-fitting, score calibration can be biased.

### Safe and unsafe provenance patterns

- Safe: estimator fitted on proper-training data only.
- Safe: unsupervised feature-only use of calibration features when explicitly
    marked by estimator metadata.
- Risky: estimator fitted on calibration labels/residuals without cross-fitting.

### Runtime behavior and compatibility

By default, provenance checks are permissive and backward compatible:

- Estimators with only `fitted` + `apply(x)` continue to work.
- If optional provenance metadata is absent, CE does not fail.
- If metadata indicates calibration-label/residual leakage without cross-fitting,
    CE emits a `UserWarning` and INFO log in permissive mode.

You can request strict enforcement per explainer:

```python
wrapper.explainer.reject_difficulty_provenance_policy = "strict"
```

In strict mode, the same leakage pattern raises `ValidationError`.

When `experimental.difficulty_normalized` is used, result metadata includes:

- `difficulty_estimator_provenance_available`
- `difficulty_estimator_provenance_warning_emitted`
- `difficulty_estimator_provenance_validation_mode`

plus optional fields such as fit source and calibration-label/residual markers.

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
wrapper.explainer.reject_orchestrator.initialize_reject_learner(
    threshold=threshold,
    ncf="default",
)

result = wrapper.predict(x_test, reject_policy=RejectPolicy.FLAG)
print(f"Reject rate: {result.metadata['reject_rate']:.2%}")
```

To use `ensured` NCF with regression:

```python
wrapper.explainer.reject_orchestrator.initialize_reject_learner(
    threshold=threshold,
    ncf="ensured",
    w=0.5,
)
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

Entry-point tier: Tier 2
