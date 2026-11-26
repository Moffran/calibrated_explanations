# ADR-032 Guarded Semantics Reference

Use guarded explanations only for ADR-032 interval-plausibility filtering of
candidate rules and perturbations.

- Canonical API: `guarded_options=GuardedOptions()`
- Allowed calls:
  - `explainer.explain_factual(X, guarded_options=GuardedOptions())`
  - `explainer.explore_alternatives(X, guarded_options=GuardedOptions())`
- Removed surfaces:
  - `guarded=True`
  - `explain_guarded_factual(...)`
  - `explore_guarded_alternatives(...)`

Guarded explanation is not an instance-level out-of-distribution detector.
When the workflow needs OOD detection, use dedicated OOD tooling rather than
relying on guarded explanation semantics.
