# ADR-027: Documentation Audience Standard

Status: Accepted
Date: 2025-10-28
Deciders: Documentation Working Group
Reviewers: Core maintainers
Supersedes: ADR-022
Superseded-by: None

## Context

The documentation set drifted into a mix of telemetry walkthroughs, plugin mechanics, and governance policies that obscured the calibrated explanations contract. Practitioners struggle to find runnable factual and alternative examples, probabilistic and interval regression guidance is hidden, and contributors wade through optional tooling before learning the plugin guardrails. Researchers must hop between duplicated hubs to reconstruct methodology. A new, audience-first standard is required to enforce clarity, parity between classification and regression, and an explanations-first narrative.

## Decision

Adopt an audience-led documentation structure anchored on three journeys—practitioners, researchers, and contributors/maintainers—with the following rules:

1. **Getting started first:** The docs landing page and navigation begin with a getting started section containing installation, classification quickstart with links to task api comparison and troubleshooting.
2. **Explanations first:** Every audience hub opens with factual and alternative workflows, highlighting calibrated probabilities, uncertainty intervals, and interpretation checkpoints before any optional tooling.
3. **Classification and regression parity:** Binary and multiclass classification content appears alongside interval and probabilistic regression guidance wherever workflows are described.
4. **Interpretation emphasis:** Quickstarts and practitioner content link immediately to interpretation guides that translate calibrated outputs into decisions.
5. **Audience hubs with advanced tracks:**
   - **Practitioners:** Core page focuses on deployment-ready explanations; an advanced page collects telemetry, performance, PlotSpec, and other optional operational aids.
   - **Researchers:** Hub consolidates theory summaries, replication steps, benchmark manifests, and a future work ledger tied to published literature.
   - **Contributors/Maintainers:** Landing content foregrounds the plugin contract, registry guardrails, and extension checklist; an advanced page houses telemetry hooks, performance tuning, and PlotSpec instrumentation guidance.
6. **Shared foundations:** Concepts, reference material, schemas, governance, and citing instructions live under a shared "Foundations" grouping linked from every hub.
7. **External plugin boundaries:** Core docs reference fast explanations and other optional bundles only as external plugins, pointing to the external plugin index and aggregated installation extras.

## Consequences

### Positive
- Streamlined navigation anchored on audience intent, reducing time to the first calibrated explanation.
- Equal visibility for classification and regression ensures practitioners adopt the full calibrated toolkit.
- Contributors encounter the plugin contract and guardrails before optional telemetry/performance tooling, protecting the calibrated explanations contract.
- Research collateral lives in a single hub, simplifying replication and future work tracking.

### Risks / Mitigations
- **Migration churn:** Moving files into the new structure may break links. Mitigation: run `make -C docs html` with linkcheck and update redirects where necessary.
- **Content ownership:** Audience hubs require dedicated maintainers. Mitigation: assign owners per hub in the governance index and review quarterly.

## Adoption Plan
1. Archive superseded improvement docs (`component_diagram.md`, `documentation_information_architecture.md`, `documentation_review.md`) and replace them with a unified blueprint.
2. Rebuild the Sphinx navigation so Getting Started leads, followed by the practitioner, researcher, and contributor hubs, then shared foundations.
3. Refresh audience landing pages and create advanced-path documents per this ADR.
4. Consolidate research content under `docs/researcher/` and update cross-links across the repository.
5. Maintain parity in future changes by requiring PR reviewers to verify classification/regression coverage and interpretation-first framing.

## References
- Documentation Overhaul Blueprint (docs/improvement/documentation_overhaul.md)
- ADR-012: Documentation & Gallery Build Policy
- ADR-018: Code Documentation Standard
- ADR-026: Explanation Plugin Semantics
