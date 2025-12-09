> **Status note (2025-10-28):** Last edited 2025-10-28 · Archive after: Re-evaluate post-v1.0.0 maintenance review · Implementation window: v0.9.0–v1.0.0.

Last updated: 2025-10-28

# Documentation Overhaul Blueprint

> Terminology: This blueprint uses the shared definitions in hard guardrails without accidentally breaking published workflows. Terminology follows [terminology](RELEASE_PLAN_v1.md#terminology-for-improvement-plans): release milestones mark the gates in the enforcement roadmap, and phases referenced in dependent plans keep those definitions.

## Status
- **State:** Draft (audience-first restructure in progress)
- **Owner:** Documentation working group
- **Related ADRs:** ADR-027 (Documentation Audience Standard)

### ADR-027 status snapshot

| Item | Status | Notes |
| --- | --- | --- |
| Audience-first navigation and hubs | ⚙️ In progress | Navigation reordered; audience hubs under active rewrite. |
| PR template parity review gate | ✅ Complete | Checklist item added so new docs keep practitioner/researcher parity. |
| ADR alignment | ⚙️ In progress | ADR-022 marked superseded; remaining work tracked through this blueprint. |

## Purpose
Establish a single, audience-led plan for rebuilding the documentation set so practitioners, researchers, and contributors can each reach calibrated explanations quickly without wading through optional telemetry or plugin details. This blueprint replaces the superseded information-architecture, review, and standardization plans that previously overlapped.

## Audience pillars
1. **Practitioners** – Production teams who need trustworthy, calibrated explanations (factual and alternative) plus probabilistic and interval regression guidance. Keep their path concise, runnable, and interpretation-first.
2. **Researchers** – Scientists validating methodology against the literature and exploring future work. Provide reproducible experiments, theory summaries, and research roadmaps without burying them under operational tooling.
3. **Contributors & maintainers** – Engineers extending the framework through plugins while preserving the calibrated contract. Highlight the plugin guardrails first, then advanced observability/performance hooks as opt-in extras.

## Structural mandates
- The docs root opens with the getting started guide before any other navigation entry.
- Each audience hub owns both a core path and an advanced track. Core paths begin with factual and alternative explanations for both classification (binary and multiclass) and regression (probabilistic and interval). Advanced tracks gather telemetry, performance, PlotSpec, and other optional tooling.
- Shared foundations (concepts, reference, governance) live under a single "Foundations" grouping referenced by all hubs.
- Plugin contract material is elevated in the contributor hub and links directly to the external plugin index and ADR-026.
- Research content collapses into the researcher hub: literature summaries, replication workflow, benchmark manifests, and future work proposals.

## Content actions
- Rewrite landing pages (`index.md`, `get-started/index.md`, audience hubs) so explanations lead and classification/regression parity is explicit.
- Pair every quickstart with an interpretation call-to-action and cross-links to probabilistic and interval regression guidance.
- Introduce advanced practitioner and contributor pages that consolidate telemetry, performance, and PlotSpec topics instead of scattering them across the default flow.
- Migrate research collateral into the researcher hub, eliminating duplicate navigation under `docs/research/`.
- Update cross-links so external plugin bundles, governance policies, and research proofs reinforce the calibrated explanations contract instead of pulling readers into optional tooling prematurely.

## Implementation checklist (mapped to release milestones)
1. Update ADR-022 to mark it superseded by ADR-027 and adopt the new documentation standard. (done; completed in v0.8.0)
2. Reorganize the Sphinx toctree so Getting Started appears first and the three audience hubs headline the navigation. (v0.8.0 gate)
3. Create advanced audience pages and refresh practitioner/researcher/contributor landing copy per the mandates above. (v0.9.1 sustainment)
4. Consolidate research content under `docs/researcher/` and clean up orphaned references. (v0.9.1)
5. Run `make -C docs html` with `-W` and linkcheck enabled at every milestone branch cut (v0.8.0, v0.9.1, v0.10.x) to ensure navigation and cross-references remain valid and block release if warnings surface.
6. Attach rollback notes to each milestone: if Sphinx/linkcheck fails at release branch cut, revert navigation commits and fall back to the last green toctree; waivers require a dated follow-up issue and expire after one iteration.

## Success metrics
- Practitioner quickstart + interpretation sequence executes without modification in CI smoke tests and is referenced in all practitioner-facing pages.
- Research hub lists parity coverage for classification (binary & multiclass) and regression (interval & probabilistic) benchmarks with direct citations.
- Contributor hub surfaces the plugin contract within the first screenful and clearly delineates optional telemetry/performance extras.
- Docs navigation reduces to Getting Started, three audience hubs, and shared foundations with no duplicate research or contributor sections.
