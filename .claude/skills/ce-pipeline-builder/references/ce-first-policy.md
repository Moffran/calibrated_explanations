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
2. **Wrapper** — use `WrapCalibratedExplainer` (or a verified subclass). Never
   invent a new wrapper class.
3. **Fit** — `explainer.fit(x_proper, y_proper)` → assert `explainer.fitted is True`.
4. **Calibrate** — `explainer.calibrate(x_cal, y_cal)` → assert `explainer.calibrated is True`.
5. **Explain (standard)** — `explainer.explain_factual(X)` or
   `explainer.explore_alternatives(X)`.
6. **Explain (guarded)** — for production / unknown distributions, use
   `explainer.explain_guarded_factual(X)` or
   `explainer.explore_guarded_alternatives(X)` instead.
7. **Calibrated by default** — never return uncalibrated outputs unless
   explicitly requested.
8. **Conjunctions** — `explanations.add_conjunctions(...)` or
   `explanations[idx].add_conjunctions(...)`.
9. **Narratives & plots** — `.to_narrative(output_format=...)` and `.plot(...)`.
10. **Probabilistic regression** — `threshold=` for probabilistic intervals;
    `low_high_percentiles=` for conformal.

## Helper Utilities

`src/calibrated_explanations/ce_agent_utils.py` exposes validated helpers:

| Helper | Purpose |
|---|---|
| `ensure_ce_first_wrapper(model)` | Wrap and validate |
| `fit_and_calibrate(explainer, x_proper, y_proper, x_cal, y_cal)` | Fit + calibrate |
| `explain_and_narrate(explainer, X, mode)` | Explain + return narrative |
| `wrap_and_explain(model, x_proper, y_proper, x_cal, y_cal, X_query)` | Full pipeline |
| `probe_optional_features(explainer)` | Check available optional features |

## Architecture Constraint

`WrapCalibratedExplainer` lives in `core/`. Plugins live in `plugins/`.
Core never imports plugins (ADR-001). Every new feature goes into `plugins/`.

## Failure Messages

```python
CE_FIRST_POLICY["failure_messages"] = {
    "missing_library": "calibrated_explanations is required. Install with: pip install calibrated-explanations",
    "invalid_wrapper": "Wrapper must be a WrapCalibratedExplainer (or subclass) from calibrated_explanations.",
    "not_fitted": "Wrapper must be fitted before calibration or explanation.",
    "not_calibrated": "Wrapper must be calibrated before prediction or explanation.",
}
```
