# Practitioner playbooks

These playbooks assemble the shared foundations that practitioners reach for
when promoting calibrated explanations into production pipelines.

## Core Capability Playbooks

* {doc}`mondrian-calibration` – Conditional calibration for fairness-aware
  deployments, heterogeneous data, and domain-specific groupings.
* {doc}`ensured-explanations` – Reduce epistemic uncertainty in alternative
  explanations for high-stakes decisions.
* {doc}`decision-policies` – Convert uncertainty outputs into systematic
  accept/reject/defer decision policies.

## Integrate calibrated explanations

* {doc}`../../foundations/how-to/integrate_with_pipelines` – Batch and realtime
  integration recipes, including configuration notes for the
  ``WrapCalibratedExplainer``.
* {doc}`../../foundations/how-to/export_explanations` – Persistence patterns for
  dashboards, audits, and downstream analytics.

## Regulatory Compliance

* {doc}`eu-ai-act-compliance` – Article-by-article mapping of EU AI Act (Regulation
  (EU) 2024/1689) obligations to `calibrated_explanations` capabilities, including a
  compliance checklist, limitation analysis, and a quick-start integration recipe.

* {doc}`eu-data-liability-compliance` -- Companion compliance reference covering GDPR, the Product Liability Directive (2024/2853), the proposed AI Liability Directive, the Digital Services Act (2022/2065), and Equal Treatment Directives -- with code patterns, a cross-regulation checklist, and a gaps analysis.

## Ship with governance guardrails

* {doc}`../../foundations/governance/release_checklist` – Calibrated
  explanations release gates and documentation updates.
* {doc}`../../foundations/governance/section_owners` – Ownership map for runtime
  and documentation components.

```{toctree}
:maxdepth: 1
:hidden:

mondrian-calibration
ensured-explanations
decision-policies
eu-ai-act-compliance
eu-data-liability-compliance
../../foundations/how-to/integrate_with_pipelines
../../foundations/how-to/export_explanations
../../foundations/governance/release_checklist
../../foundations/governance/section_owners
```
