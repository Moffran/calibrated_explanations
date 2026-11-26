# CE-First Policy Reference

> **Source:** `CONTRIBUTOR_INSTRUCTIONS.md §1` — canonical, do not edit here.
> This file is a loadable reference copy for skills that need the policy inline.

---

## Mandatory CE-First Checklist

All agent workflows **must** follow this checklist before producing any
prediction or explanation output:

1. **Library presence** — if `calibrated_explanations` is not importable, fail fast:
   ```bash
   pip install calibrated-explanations
   ```
2. **Wrapper** — use `WrapCalibratedExplainer` directly. A verified subclass may
   preserve the contract for an intentional extension; never invent an ad-hoc
   wrapper class.
3. **Fit** — `explainer.fit(x_proper, y_proper)` → assert `explainer.fitted is True`.
4. **Calibrate** — `explainer.calibrate(x_cal, y_cal)` → assert `explainer.calibrated is True`.
5. **Explain (standard)** — `explainer.explain_factual(X)` or
   `explainer.explore_alternatives(X)`.
6. **Explain (guarded)** — for production / unknown distributions, use
   `guarded_options=GuardedOptions()` (import from
   `calibrated_explanations.explanations.guarded_options`):
   `explainer.explain_factual(X, guarded_options=GuardedOptions())` or
   `explainer.explore_alternatives(X, guarded_options=GuardedOptions())`.
   `explain_guarded_factual` / `explore_guarded_alternatives` were REMOVED in v0.11.3.
7. **Calibrated by default** — never return uncalibrated outputs unless
   explicitly requested.
8. **Conjunctions** — `explanations.add_conjunctions(...)` or
   `explanations[idx].add_conjunctions(...)`.
9. **Narratives & plots** — `.to_narrative(output_format=...)` and `.plot(...)`.
10. **Probabilistic regression** — `threshold=` for probabilistic intervals;
    `low_high_percentiles=` for conformal.

## Direct public API

Use `WrapCalibratedExplainer` and the documented public CE API directly.

## Architecture Constraint

`WrapCalibratedExplainer` lives in `core/`. Plugins live in `plugins/`.
Core never imports plugins (ADR-001). Every new feature goes into `plugins/`.

## Failure Messages

| Condition | Message |
|---|---|
| Library missing | calibrated_explanations is required. Install with: pip install calibrated-explanations |
| Invalid wrapper | Wrapper must be WrapCalibratedExplainer or a verified subclass from calibrated_explanations. |
| Not fitted | Wrapper must be fitted before calibration or explanation. |
| Not calibrated | Wrapper must be calibrated before prediction or explanation. |
