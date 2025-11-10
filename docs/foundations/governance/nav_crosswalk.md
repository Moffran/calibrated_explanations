# Navigation crosswalk

Track where legacy documentation pages now live in the **ADR-027** audience-first
information architecture. Update this checklist whenever a page moves or is retired.

> **Status note:** Last updated 2025-11-10 · Per ADR-027 (Documentation Audience Standard) · Supersedes ADR-022.

## Legacy → New location mapping

| Legacy page | New location | Notes |
| ----------- | ------------ | ----- |
| `calibrated_explanations.rst` (legacy landing page) | :doc:`../../index` | Landing narrative split into role-based overview and section toctrees with audience-first callouts. |
| `getting_started.md` | :doc:`../../get-started/index` with :doc:`../../get-started/installation`, :doc:`../../get-started/quickstart_classification`, :doc:`../../get-started/quickstart_regression`, :doc:`../../get-started/troubleshooting` | Content decomposed into dedicated installation, runnable quickstarts, and troubleshooting checklists. |
| `practitioner_guide.md` (legacy) | :doc:`../../practitioner/index` with :doc:`../../practitioner/playbooks/index` and :doc:`../../practitioner/advanced/index` | Practitioner core and advanced workflows split: core focuses on factual/alternative explanations; advanced collects optional telemetry and performance tooling. |
| `researcher_guide.md` (legacy) | :doc:`../../researcher/index` with :doc:`../../researcher/replication/index` and :doc:`../../researcher/advanced/index` | Research content consolidated: replication workflows, benchmark manifests, and literature map in one hub with advanced observability extras. |
| `contributor_guide.md` (legacy) | :doc:`../../contributor/index` with :doc:`../../contributor/plugin-contract` and :doc:`../../contributor/extending/guides/index` | Contributor paths split: plugin contract first (core requirement), then extending guides for CLI and configuration surfaces. |
| `viz_plotspec.md` | :doc:`../how-to/plot_with_plotspec` | PlotSpec guidance converted to a how-to focused on configuration plus telemetry links. Practitioner playbooks reference this for advanced visualization. |
| `architecture.md` | :doc:`../concepts/index`, :doc:`../concepts/alternatives`, :doc:`../concepts/probabilistic_regression` | Architecture material split into focused concept pages; optional telemetry moved under :doc:`../governance/optional_telemetry` so provenance scaffolding is clearly opt-in. |
| *(new)* interpretation guide | :doc:`../how-to/interpret_explanations` | Primary resource for reading factual and alternative explanations, PlotSpec visuals, and telemetry provenance. Linked from all quickstarts. |
| `plugin_guide.md` (legacy) | :doc:`../../contributor/plugin-contract` with :doc:`../../contributor/extending/guides/index` and :doc:`../../appendices/external_plugins` | Plugin contract elevated in contributor hub; extending guides for wiring CLI and config; external plugin index (community plugins) moved to appendices. |
| `pr_guide.md` | :doc:`../governance/pr_guide` and :doc:`../governance/section_owners` | Governance actions captured under maintainership docs with owners and release narrative under shared foundations. |
| *(new)* migration guides | :doc:`../../migration/index` | Terminology changes, breaking changes, and upgrade guidance captured in a dedicated section linked from all audience hubs. |

## Audience-first structure verification

Per ADR-027, the documentation now follows an audience-led structure:

- [x] **Getting Started** appears first in the main toctree (`:doc:`../../get-started/index``)
- [x] **Practitioner hub** includes core workflow (quickstarts → interpretation) and advanced extras (telemetry, performance, PlotSpec)
- [x] **Researcher hub** consolidates theory, replication, and benchmarks with equal coverage for classification and regression
- [x] **Contributor hub** foregrounds plugin contract and extending guides; defers optional telemetry/performance to advanced section
- [x] **Shared Foundations** (concepts, how-to, reference, governance) linked from all audience hubs
- [x] **Appendices & Migration** segregated for upgrade guidance and external plugin references
- [ ] Re-run the gating commands in :doc:`release_checklist` after updating cross-references

## Verification checklist

- [ ] Confirm all pages in the mapping table are added to their respective section toctrees
- [ ] Ensure deprecated filenames or redirects point readers to the new audience-first paths
- [ ] Validate that classification and regression parity is maintained across quickstarts and advanced guides
- [ ] Confirm interpretation guides are linked from every quickstart (classification and regression)

## Navigation patterns to maintain

When editing audience landing hubs, playbooks, or quickstarts, follow these patterns:

1. **Interpretation-first callouts:** All quickstarts (classification and regression) link to :doc:`../how-to/interpret_explanations` immediately after the runnable code block.
2. **Parity enforcement:** Practitioner and researcher pages must provide equivalent coverage for binary/multiclass classification and probabilistic/interval regression.
3. **Research hub references:** Audience hubs can point readers to :doc:`../../researcher/index` in a Resources or Further Reading section to avoid navigation clutter.
4. **Plugin boundaries:** Contributor and practitioner docs reference external plugins (fast explanations, shap-lime, etc.) via :doc:`../../appendices/external_plugins` instead of embedding them in the core workflow.
5. **Cross-link validation:** After structural changes, run `sphinx-build -b html -W` and check for broken cross-references in the build log.
