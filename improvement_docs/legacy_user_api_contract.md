> **Status note (2025-10-24):** Last edited 2025-10-24 · Archive after: Re-evaluate post-v1.0.0 maintenance review · Implementation window: v0.9.0–v1.0.0.

# Legacy User API Contract Analysis

This note enumerates the public-facing methods and parameters that library users
rely on today, based on the executable examples in the README and maintained
notebooks. The goal is to make the existing surface area explicit so we can add
hard guardrails without accidentally breaking published workflows.

## Primary entry point: `WrapCalibratedExplainer`

### Construction and lifecycle
- `WrapCalibratedExplainer(learner)` instantiates the wrapper around a
  user-supplied estimator for both classification and regression quick-starts.
  The object is expected to expose `.learner`, `.fitted`, and `.calibrated`
  state after `fit` and `calibrate` are called.【F:docs/getting_started.md†L5-L47】
- `.fit(X_proper_training, y_proper_training)` mirrors the scikit-learn fit
  contract before calibration.【F:docs/getting_started.md†L14-L31】
- `.calibrate(X_calibration, y_calibration, ...)` wires calibration metadata,
  and is invoked with feature names, categorical labels, and optional difficulty
  estimators across the tutorials.【F:docs/getting_started.md†L18-L31】【F:docs/getting_started.md†L294-L311】【F:notebooks/quickstart_wrap.ipynb†L685-L717】
- Users routinely re-wrap existing learners or calibrated explainers to resume
  work in a new session, so the constructor must continue to accept either kind
  of object.【F:docs/getting_started.md†L400-L413】

### Prediction helpers
- `.predict(x, uq_interval=False, calibrated=True, **kwargs)` must continue to
  support:
  - raw predictions before calibration; users call it immediately after `fit`
    and again before invoking `.calibrate(...)`.【F:docs/getting_started.md†L287-L299】
  - calibrated predictions with optional uncertainty intervals via
    `uq_interval=True` (returning `(prediction, (low, high))`).【F:docs/getting_started.md†L316-L331】
  - one-sided or asymmetric intervals by forwarding
    `low_high_percentiles=(low, high)` through `**kwargs`. Tutorials demonstrate
    symmetric, one-sided, and asymmetric settings.【F:docs/getting_started.md†L352-L359】
  - thresholded regression predictions for probabilistic labels via
    `threshold=...`, including downstream plotting of those explanations.【F:docs/getting_started.md†L325-L394】
  - uncalibrated access via `calibrated=False` for both classification and
    regression when users want to inspect the base learner.【F:README.md†L176-L198】
- `.predict_proba(x, uq_interval=False, calibrated=True, threshold=None, **kwargs)`
  must keep support for calibrated probability estimates with optional
  uncertainty tuples and regression threshold usage.【F:docs/getting_started.md†L316-L330】【F:README.md†L190-L200】

### Explanation factories
- `.explain_factual(x, low_high_percentiles=..., threshold=..., bins=...)`
  underpins all factual example workflows, including the probabilistic, one-
  sided, and asymmetric interval variants.【F:docs/getting_started.md†L333-L359】【F:docs/getting_started.md†L379-L388】
- `.explore_alternatives(x, low_high_percentiles=..., threshold=..., bins=...)`
  powers alternative explanations for both standard and probabilistic cases with
  identical parameter expectations.【F:docs/getting_started.md†L361-L377】【F:docs/getting_started.md†L389-L394】
- The wrapper’s explanation factories rely on parameter aliases such as
  `low_high_percentiles`, `threshold`, and optional `bins`; these keys must stay
  recognized by the normalization layer so notebooks keep working.【F:docs/getting_started.md†L352-L359】【F:notebooks/demo_conditional.ipynb†L549-L552】

### Explanation collection ergonomics
- Returned explanation collections must keep `.plot()` with the current keyword
  set (`index`, `filter_top`, `uncertainty`, `filename`, ranking knobs) and
  default behaviors for whole-collection rendering.【F:docs/getting_started.md†L206-L243】
- Users rely on standard indexing semantics (`collection[0]`, `collection[:1]`)
  to focus on individual instances.【F:notebooks/quickstart_wrap.ipynb†L434-L435】
- `.add_conjunctions(...)` and `.remove_conjunctions()` must preserve their
  signatures:
  - The zero-argument form is used throughout the quick-start and regression
    guides.【F:docs/getting_started.md†L218-L223】【F:docs/getting_started.md†L374-L377】
  - `max_rule_size` and `n_top_features` arguments appear in the regression and
    probabilistic notebooks and therefore need to remain supported.【F:notebooks/demo_regression.ipynb†L614-L614】【F:notebooks/demo_probabilistic_regression.ipynb†L1213-L1213】
- Downstream consumers call `.get_explanation(i)` to pull individual
  explanations, and immediately call `.plot(...)` (often with filenames) on the
  returned objects; both operations and their keyword arguments are
  contractually required.【F:notebooks/demo_multiclass_glass.ipynb†L1477-L2487】

## Direct use of `CalibratedExplainer`

While the wrapper is the default entry point, the notebooks also instantiate
`CalibratedExplainer` directly, so its constructor and helper methods are part
of the legacy contract.

- Constructor parameters `mode`, `feature_names`, `categorical_features`,
  `categorical_labels`, `class_labels`, `bins`, and `difficulty_estimator` are
  all used in published notebooks and must remain stable.【F:notebooks/quickstart.ipynb†L1337-L1338】【F:notebooks/demo_conditional.ipynb†L502-L529】
- `.set_difficulty_estimator(...)` is applied after construction to swap
  strategies, and must continue to accept fitted `DifficultyEstimator`
  instances.【F:notebooks/quickstart.ipynb†L1338-L1338】
- `.explain_factual(...)` and `.explore_alternatives(...)` share the same
  parameter expectations as the wrapper versions and are invoked with custom
  `low_high_percentiles`, `threshold`, and `bins` arguments across regression and
  conditional notebooks.【F:notebooks/demo_conditional.ipynb†L544-L610】【F:notebooks/demo_regression.ipynb†L513-L1394】
- Alternative initialization paths (e.g., Mondrian bins for conditional fairness
  experiments) assume the constructor accepts precomputed `bins` along with the
  usual metadata.【F:notebooks/demo_conditional.ipynb†L508-L529】

## Explanation object behaviors

Single-instance explanation objects (`FactualExplanation`,
`AlternativeExplanation`, `FastExplanation`) inherit the collection behaviors and
expose `.plot(filter_top=..., uncertainty=..., filename=...)` used when looping
through explanations to generate artifacts for papers and tutorials.【F:notebooks/demo_multiclass_glass.ipynb†L1477-L2487】

## Summary of required guardrails

- Preserve the methods and keyword parameters outlined above for
  `WrapCalibratedExplainer`, `CalibratedExplainer`, and explanation collections.
- Maintain indexing, slicing, and `.get_explanation(...)` semantics on
  explanation collections.
- Keep support for probabilistic regression flags (`threshold`), interval knobs
  (`low_high_percentiles`), and conjunction controls (`max_rule_size`,
  `n_top_features`).

These findings feed into ADR-020, which proposes concrete guardrails to freeze
this API surface going forward.
