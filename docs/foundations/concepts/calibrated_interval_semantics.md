# Calibrated interval semantics

This page is the semantics source for user-facing documentation.

## Scope

Calibrated Explanations has three distinct semantics modes:

1. Classification with Venn-Abers probability intervals.
2. Percentile/interval regression without thresholds using CPS percentile intervals.
3. Probabilistic or thresholded regression using CPS with Venn-Abers event probabilities.

Do not merge these into one guarantee statement.

## Mode 1: Classification (Venn-Abers)

### Calibration prerequisites
- Fit on a proper training split.
- Calibrate on a held-out calibration split.

### Mode-specific guarantees
- Outputs are calibrated class probabilities with interval bounds from Venn-Abers.

### Assumptions
- Calibration and deployment samples are exchangeable or distribution-matched.

### Explicit non-guarantees
- No guarantee under distribution drift or regime shift.
- No guarantee that class probability intervals transfer unchanged across domains.

### Explanation-envelope and feature-level limits
- Rule-level and feature-level intervals are explanation artifacts tied to calibrated
  perturbation behavior.
- They are not causal guarantees.

## Mode 2: Percentile or interval regression (CPS)

### Calibration prerequisites
- Fit on a proper training split.
- Calibrate with CPS on a held-out calibration split.

### Mode-specific guarantees
- Percentile intervals are CPS-based predictive intervals for requested percentiles.

### Assumptions
- Exchangeability or calibration-deployment distribution match.

### Explicit non-guarantees
- No guarantee that requested percentiles remain calibrated after drift.
- No guarantee of fixed interval width across subpopulations.

### Explanation-envelope and feature-level limits
- Feature-level interval effects describe model behavior under perturbation.
- They do not guarantee intervention outcomes in the real world.

## Mode 3: Probabilistic or thresholded regression (CPS + Venn-Abers)

### Calibration prerequisites
- Fit regression model on a proper training split.
- Build threshold event probabilities through CPS outputs calibrated with Venn-Abers.

### Mode-specific guarantees
- Returns calibrated event probabilities for threshold queries such as `P(y <= t)`
  or interval events.

### Assumptions
- Exchangeability or deployment match to calibration distribution.

### Explicit non-guarantees
- No guarantee for threshold probability calibration under drift.
- No guarantee that threshold semantics imply causal actionability.

### Explanation-envelope and feature-level limits
- Feature-level probability shifts and envelopes describe model response patterns.
- They are not guarantees of controlled intervention effects.

## Cross-mode non-guarantees

- Calibration guarantees are conditional on calibration assumptions.
- No unconditional guarantee under dataset shift, temporal drift, or adversarial change.
- Explanation-level intervals should not be promoted as formal per-feature coverage
  guarantees unless explicitly proven for that claim.

## Related standards and ADRs

- {doc}`../../standards/STD-004-documentation-audience-standard`
- {doc}`../../standards/STD-002-code-documentation-standard`
- `docs/improvement/adrs/ADR-021-calibrated-interval-semantics.md` (maintainer-only docs set)
- `docs/improvement/adrs/ADR-012-documentation-and-gallery-build-policy.md` (maintainer-only docs set)
- `docs/improvement/adrs/ADR-026-explanation-plugin-semantics.md` (maintainer-only docs set)

Entry-point tier: Tier 3.

