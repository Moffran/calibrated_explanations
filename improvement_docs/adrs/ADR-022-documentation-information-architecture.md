> **Status note (2025-10-24):** Last edited 2025-10-24 · Archive after: Retain indefinitely as architectural record · Implementation window: Per ADR status (see Decision).

# ADR-022: Documentation Information Architecture

Status: Accepted
Date: 2025-10-12
Deciders: Documentation Working Group
Reviewers: Core maintainers
Supersedes: None
Superseded-by: None

## Context

The documentation set has grown organically, mixing audience needs (users, contributors, researchers) in a single navigation tree. Quickstart content is brittle, the plugin guide is stale, and contributor policies crowd the main user journey. Previous efforts (ADR-012 build policy, ADR-018 docstring standard) focus on quality gates but not on information architecture. We need a structured approach before shipping v0.8.0, which introduces telemetry documentation and plugin defaults, to ensure readers can discover relevant material.

## Decision

Adopt the role-based documentation information architecture defined in `improvement_docs/documentation_information_architecture.md` with the following commitments:

1. Restructure the Sphinx toctree into seven top-level sections: Overview, Get Started, How-to Guides, Concepts & Architecture, Reference, Extending the Library, Governance & Support.
2. Split quickstart material into runnable classification/regression guides with validated code snippets and environment troubleshooting.
3. Move maintainer and contributor workflows (PR guide, coding standards, release policy) under Extending the Library and Governance & Support.
4. Create a telemetry concept page and align plugin documentation with the registry/CLI workflows introduced in v0.8.0.
5. Establish content ownership and pre-release review checkpoints so every minor release confirms the navigation remains accurate and updated.

## Consequences

### Positive

- Readers land on audience-specific portals, reducing time to first success.
- Maintainers can assign ownership per section, keeping updates scoped and accountable.
- Aligns with ADR-012 by ensuring CI gate results map to clearly owned content.

### Negative / Risks

- Initial restructuring requires coordinated effort across docs maintainers and feature owners.
- Potential for broken links during migration; mitigated via linkcheck and nav tests in CI.

## Options Considered

1. **Status quo with minor edits** – Rejected; fails to address audience separation and quickstart issues.
2. **Move to a new docs platform immediately** – Deferred; platform migration (e.g., MkDocs) would slow near-term releases without first establishing the desired architecture.

## Adoption Plan

- Implement toctree changes, quickstart refactor, telemetry page, and plugin guide updates in the v0.8.0 release cycle.
- Update contributor templates to reference section ownership and review cadence.
- Track progress in the release plan and mark this ADR as Accepted once the new navigation ships and ownership is assigned.

## References

- `improvement_docs/documentation_information_architecture.md`
- ADR-012: Documentation & Gallery Build Policy
- ADR-018: Code Documentation Standard
