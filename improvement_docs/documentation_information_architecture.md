# Documentation Information Architecture Proposal

## Purpose

Define a documentation structure that serves three distinct audiences:

1. **Practitioners** embedding calibrated explanations into ML systems.
2. **Researchers** exploring the theory, evaluation, and extensibility of the library.
3. **Contributors and maintainers** evolving the plugin stack, telemetry, and tooling.

The structure clarifies ownership, navigational hierarchy, and update workflows so documentation can evolve in lock-step with releases.

## Guiding principles

- **Role-based entry points:** Each audience has an obvious landing page with curated tasks and cross-links.
- **Task-first navigation:** "How-to" and tutorials remain runnable and scoped to a single workflow; advanced concepts move to dedicated concept guides.
- **Single source of truth:** API details come from autodoc. Policy/ADR decisions live under `improvement_docs/` with summaries surfaced in docs.
- **Release alignment:** Each minor release identifies documentation deltas and assigns owners.

## Proposed top-level navigation

1. **Overview**
   - Introduction, positioning, high-level capabilities.
   - Release highlights synced with `CHANGELOG.md` and the current version banner.
   - Quick links to install instructions and the latest release notes.
2. **Get Started**
   - Installation matrix (pip, conda, optional extras).
   - Quickstart walkthroughs for classification and regression (two discrete runnable code blocks).
   - Troubleshooting for environment setup (link to FAQ).
3. **How-to Guides**
   - Task-oriented guides: integrating with scikit-learn pipelines, configuring telemetry, exporting explanations.
   - Separate pages for factual, alternative, fast modes with consistent structure (inputs, outputs, examples).
   - Copy-friendly CLI usage scenarios.
4. **Concepts & Architecture**
   - Explanation lifecycle, calibration strategies, plugin registry, telemetry semantics.
   - Versioning and stability guarantees (summaries of ADR-005/006/017/021 with links).
5. **Reference**
   - Auto-generated API (modules grouped by domains: `core`, `plugins`, `viz`).
   - Schema reference (current schema version, change log).
   - Configuration reference (pyproject settings, env vars, CLI flags).
6. **Extending the Library**
   - Plugin authoring guides (calibrator, plot, interval examples).
   - Contribution workflow: local setup, coding standards (ADR-017/018), test expectations.
   - ADR summaries relevant to extension points.
7. **Governance & Support**
   - Release policy, support SLA, security reporting, roadmap snapshots.
   - Link to release plan milestones and community forums (if any).

## Page inventory & migration notes

| Existing page | Target location | Action |
| --- | --- | --- |
| `docs/index.rst` overview paragraphs | Overview | Update copy to reflect v0.8.0 feature set and link to release highlights. |
| `docs/getting_started.rst` | Get Started | Split into install + two quickstart guides; ensure runnable code per audience. |
| `docs/calibrated_explanations.md` | Concepts & Architecture | Reframe as lifecycle overview, move step-by-step guidance into how-to guides. |
| `docs/plugins.md` | Extending the Library | Expand with registry workflow, align with ADR-015 plugin policy. |
| `docs/telemetry.md` (to be added) | Concepts & Architecture | Author new page for schema/usage aligned with v0.8 telemetry tasks. |
| `docs/api_reference/` auto docs | Reference | Keep structure but ensure toctree matches module domains. |
| `docs/contributing.md` | Extending the Library | Pair with contributor checklist and link to governance section. |
| `improvement_docs/RELEASE_PLAN_v1.md` doc tasks | Governance & Support | Surface milestone summaries in release notes page. |

## Update workflow

- **Content owners:**
  - Overview & Get Started – release manager for each minor release.
  - How-to Guides – feature owners (e.g., plugin maintainers).
  - Concepts & Reference – tech lead for runtime core.
  - Extending & Governance – contributor experience lead.
- **Review cadence:**
  - Pre-release checklist includes verifying version banner, quickstart tests, and ADR linkage.
  - Quarterly doc audit compares sitemap against repo to catch orphaned content.
- **Tooling adjustments:**
  - Adopt `myst_parser` for Markdown parity (already enabled) and enforce `sphinx-build -W` per ADR-012.
  - Introduce nav tests (e.g., failing build if orphaned toctree entries) alongside linkcheck.

## Milestones

- **Short-term (v0.8.0):** Establish new toctree, refactor quickstart, create telemetry concept page, migrate plugin guide.
- **Medium-term (v0.9.0):** Integrate example gallery, finalize contributor workflow docs, add governance landing page.
- **Long-term (v1.0.0):** Implement versioned docs, freeze schema reference, embed health dashboards (coverage, doc lint).

## Success metrics

- Reduced onboarding questions (tracked via issue labels) within two releases.
- Quickstart notebooks execute without modification in CI smoke tests.
- Docs build passes without warnings; linkcheck and nav tests green.
- Positive feedback in release surveys (≥80% satisfaction from internal stakeholders).
