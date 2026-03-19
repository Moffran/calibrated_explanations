# Capabilities manifest

This page maps capabilities to their primary entry points.

Semantics are mode-specific. Use
{doc}`../foundations/concepts/calibrated_interval_semantics` for calibration
prerequisites, guarantees, assumptions, and non-guarantees.

## Core capability map

| Capability | Mode | Primary entry point |
| --- | --- | --- |
| Calibrated classification probabilities and intervals | Classification | {doc}`../get-started/quickstart_classification` |
| CPS percentile intervals for regression | Percentile/interval regression | {doc}`../get-started/quickstart_regression` |
| Threshold probabilities for regression targets | Probabilistic/thresholded regression | {doc}`probabilistic_regression` |
| Factual explanations | All modes | {doc}`../foundations/how-to/interpret_explanations` |
| Alternative explanations and ensured filters | All modes | {doc}`../foundations/concepts/alternatives` |
| Guarded in-distribution explanations | All modes | {doc}`../get-started/quickstart_guarded` |
| Plugin-based acceleration or custom rendering | Optional extension | {doc}`../plugins` |

## Audience routing

- Practitioners: {doc}`../practitioner/index`
- Researchers: {doc}`../researcher/index`
- Contributors: {doc}`../contributor/index`

Entry-point tier: Tier 2.

