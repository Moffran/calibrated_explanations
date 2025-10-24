Last updated: 2025-10-24

# Code Documentation Review

## Summary of Findings
- **Audience experience drift:** Entry points (README, Overview, quickstarts) still pull readers into telemetry payloads and plugin inspection before they master calibrated explanations. Practitioners are not led through a simple, calibration-first workflow, and probabilistic regression is easy to miss.
- **Differentiator underexposed:** Probabilistic regression lacks a dedicated narrative, leaving the project's unique capability buried in subsections and notebooks.
- **Research story hidden:** Research citations exist but are not surfaced in high-traffic landing pages, weakening the "research-based" positioning.
- **Plugin message muddled:** Extensibility docs emphasise registry/telemetry mechanics without stating that plugins must uphold calibrated explanations, leading to the perception that instrumentation is the goal.
- **Optional extras overloaded:** Telemetry, performance scaffolding, and PlotSpec routing appear as mandatory steps across the doc set instead of labelled add-ons.
- **Docstring baseline maintained:** Automated inspection (2025-10-10) still reports overall docstring coverage of **94.18%** across `src/calibrated_explanations` (modules 47/47, classes 67/68, functions 166/175, methods 319/346). Remaining gaps cluster in legacy utilities slated for follow-up in Phase 3 of the documentation standardization plan.

## Severity Assessment
**Overall severity: High for user-facing docs; moderate for code docstrings.**

The current doc navigation undermines the core value proposition (calibrated explanations with probabilistic regression) and risks confusing practitioners and researchers about priorities. Coverage metrics remain solid, but the narrative/structure drift requires coordinated remediation before v0.9.0.

## Supporting Data
- Coverage snapshot produced by `scripts/check_docstring_coverage.py` (2025-10-10): overall 94.18% (modules 47/47, classes 67/68, functions 166/175, methods 319/346).
- All modules sampled in ADR-018 batches C (`explanations/`, `perf/`) and D (`plugins/`) now report full compliance with numpydoc conventions.
- Four independent gap analyses (2025-01-14 through 2025-01-21) converge on the same themes: telemetry overshadowing calibrated explanations, poor visibility for probabilistic regression, lack of role-based navigation, limited research pointers, and optional extras presented as core steps.

## Priority Actions (aligned to owner tracks)

1. **Reframe landing pages (Practitioner owner)**
   - Update README quickstart and docs quickstarts to keep telemetry in an "Optional extras" section and to add probabilistic regression side-by-side with classification.
   - Add interpretation guides that mirror notebook examples and emphasise calibrated intervals and uncertainty breakdowns before any operational guidance.
2. **Stand up researcher hub (Research liaison)**
   - Create a theory overview page summarising calibration proofs, probabilistic regression rationale, and links to published papers and benchmarks.
   - Cross-link research citations from Overview, quickstarts, and probabilistic regression guides.
3. **Clarify plugin story (Contributor experience lead)**
   - Introduce a "hello, calibrated plugin" walkthrough that demonstrates extending the framework while preserving calibrated predictions.
   - Move telemetry/CLI deep dives to optional appendices; label telemetry and performance scaffolding as opt-in aids.
4. **Sustain docstring quality (Runtime tech lead)**
   - Complete Phase 3 clean-up for legacy utilities to reach ≥95% coverage.
   - Keep ADR-018 enforcement active; update dashboards to highlight calibration-focused examples in docstrings.
