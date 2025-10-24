> **Status note (2025-10-24):** Last edited 2025-10-24 · Archive after: Re-evaluate post-v1.0.0 maintenance review · Implementation window: v0.9.0–v1.0.0.

Last updated: 2025-10-24

# Documentation Information Architecture Proposal

## Purpose

Define a documentation structure that serves three distinct audiences while keeping the value proposition crystal clear:

1. **Practitioners** who need trustworthy, calibrated explanations (and probabilistic regression) in production systems.
2. **Researchers** validating the methodology against the published literature and exploring future work.
3. **Contributors and maintainers** extending the framework through plugins without diluting the calibrated-explanations contract.

The structure clarifies ownership, navigational hierarchy, and update workflows so documentation can evolve in lock-step with releases while consistently reinforcing that calibrated explanations—not telemetry or auxiliary tooling—are the product.

## Guiding principles

1. **Deliver calibrated explanations first.** The first message in every hub and
   task flow must be how to obtain, interpret, and trust calibrated explanations
   before introducing telemetry or other extras.
2. **Spotlight probabilistic regression.** Probabilistic regression guidance sits
   alongside classification in navigation, quickstarts, and release messaging.
3. **Keep examples simple by default.** Tutorials mirror README/notebook flows
   with minimal prerequisites; complexity is introduced only when explaining the
   plugin architecture or optional extras.
4. **Structure content for the three core audiences.** Practitioner, researcher,
   and contributor journeys each get curated landing pages with clear cross-links
   and calls to action.
5. **Ground every narrative in published research.** High-traffic pages link to
   peer-reviewed papers, benchmarks, and citing guidance.
6. **Champion plugin extensibility with guardrails.** Extensibility stories start
   by reinforcing calibration semantics, showing how plugins extend the core
   contract while labelling telemetry/governance as optional.
7. **Treat telemetry and similar extras as optional.** Extras are clearly marked,
   documented as opt-in, and never distract from the calibrated explanation
   journey.

## Proposed top-level navigation

1. **Overview** (shared landing page)
   - Crisp statement of calibrated explanations and probabilistic regression as the core differentiators.
   - Release highlights ordered by impact on calibrated explanations; telemetry or operational extras appear in an "Optional tooling" callout.
   - "Backed by research" banner linking to citing.md and key papers.
   - Quick links for each audience (Practitioner, Researcher, Contributor) to dive deeper.
2. **Practitioner track**
   - **Install & Quickstart** page featuring two runnable snippets: calibrated classification and probabilistic regression. Telemetry/PlotSpec steps are collapsed under "Optional: operational add-ons".
   - **Interpretation guides** for factual, alternative, and fast explanations that reuse notebook examples and highlight calibrated intervals, uncertainty breakdowns, and regression thresholds.
   - **Integration how-tos** (pipelines, deployment) that reference telemetry only as an optional compliance aid.
3. **Researcher track**
   - Theory overview summarising calibration guarantees, probabilistic regression math, and links to published papers/preprints.
   - Evaluation playbook with benchmark references, dataset notes, and reproducibility checklists.
   - Roadmap for contributing new research-backed modes (with criteria for alignment to calibrated explanations).
4. **Contributor track**
   - Plugin system overview starting with a "hello, calibrated plugin" example before registry/CLI details; reiterates that plugins must preserve calibration semantics.
   - Development workflow, coding standards, and governance documents.
   - Telemetry and performance scaffolding pages flagged as optional extras for observability.
5. **Concepts & Architecture** (shared)
   - Deep dives on explanation lifecycle, calibration strategies, and uncertainty decomposition.
   - Probabilistic regression concept article with diagrams, comparisons, and cross-links to research.
   - Plugin registry, telemetry schema, and governance concepts, each clearly labelled as optional or advanced when applicable.
6. **Reference**
   - Auto-generated API grouped by domains: calibrated explanations core, probabilistic regression helpers, plugins, viz.
   - Schema reference (current schema version, change log).
   - Configuration reference (pyproject settings, env vars, CLI flags) with optional extras tagged accordingly.
7. **Governance & Support**
   - Release policy, support SLA, security reporting, roadmap snapshots.
   - Link to release plan milestones and community forums (if any).

## Page inventory & migration notes

| Existing page | Target location | Action |
| --- | --- | --- |
| `docs/index.rst` overview paragraphs | Overview | Rewrite hero copy to foreground calibrated explanations and probabilistic regression; move telemetry notes into "Optional tooling". |
| `docs/getting_started.rst` | Practitioner track – Install & Quickstart | Split into install + two quickstart guides; keep telemetry in optional collapsible section. |
| `docs/quickstart_classification.md` (new) | Practitioner track – Install & Quickstart | Base on README example without telemetry; add pointer to interpretation guide and research citations. |
| `docs/quickstart_probabilistic_regression.md` (new) | Practitioner track – Install & Quickstart | Provide runnable probabilistic regression workflow; link to theory page. |
| `docs/interpret_factual.md` / `docs/interpret_alternative.md` (new) | Practitioner track – Interpretation guides | Showcase simple examples mirroring notebooks; include uncertainty breakdowns and probability intervals. |
| `docs/calibrated_explanations.md` | Concepts & Architecture | Reframe as lifecycle overview, add probabilistic regression deep dive, and embed research citations throughout. |
| `docs/research_hub.md` (new) | Researcher track – Theory overview | Curate papers, benchmarks, and ongoing studies. |
| `docs/plugins.md` | Contributor track – Plugin overview | Start with minimal calibrated plugin example, then cover registry workflow and compliance guardrails. |
| `docs/telemetry.md` | Contributor track / Optional extras | Position telemetry as optional; cross-link from governance and plugin compliance notes only. |
| `docs/api_reference/` auto docs | Reference | Group modules into calibrated core, probabilistic regression, plugin interfaces, viz extras; ensure probabilistic regression helpers are first. |
| `docs/contributing.md` | Contributor track – Development workflow | Pair with contributor checklist and governance links; emphasise preserving calibration semantics. |
| `improvement_docs/RELEASE_PLAN_v1.md` doc tasks | Governance & Support | Sync release milestones with audience-specific doc checkpoints and optional extras labelling. |

## Update workflow

- **Content owners:**
  - Overview & Practitioner track – release manager for each minor release.
  - Researcher track – research liaison (ensures new papers and benchmarks are linked).
  - Contributor track – contributor experience lead.
  - Concepts & Reference – runtime tech lead (coordinates probabilistic regression coverage).
- **Review cadence:**
  - Pre-release checklist includes verifying practitioner quickstarts remain telemetry-optional, probabilistic regression links are intact, and research citations appear on all landing pages.
  - Quarterly doc audit compares sitemap against repo to catch orphaned content and ensures optional extras remain labelled as such.
- **Tooling adjustments:**
  - Adopt `myst_parser` for Markdown parity (already enabled) and enforce `sphinx-build -W` per ADR-012.
  - Introduce nav tests (e.g., failing build if orphaned toctree entries) alongside linkcheck.

## Milestones

- **Short-term (v0.8.0):** Establish new toctree, refactor quickstart, create telemetry concept page, migrate plugin guide.
- **Medium-term (v0.9.0):** Launch practitioner/researcher/contributor landing pages, ship probabilistic regression quickstart + concept guide, and ensure telemetry/other extras are labelled optional throughout.
- **Long-term (v1.0.0):** Implement versioned docs, freeze schema reference, embed health dashboards (coverage, doc lint).

## Success metrics

- Reduced onboarding questions (tracked via issue labels) within two releases.
- Quickstart notebooks execute without modification in CI smoke tests.
- Docs build passes without warnings; linkcheck and nav tests green.
- Positive feedback in release surveys (≥80% satisfaction from internal stakeholders).
