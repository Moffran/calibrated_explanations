# Navigation crosswalk

Track where legacy documentation pages now live in the ADR-022 information
architecture. Update this checklist whenever a page moves or is retired.

| Legacy page | New location | Notes |
| ----------- | ------------ | ----- |
| `calibrated_explanations.rst` (legacy landing page) | :doc:`../overview/index` | Landing narrative split into role-based overview and section toctrees. |
| `getting_started.md` | :doc:`../get-started/index` with :doc:`../get-started/installation`, :doc:`../get-started/quickstart_classification`, :doc:`../get-started/quickstart_regression`, :doc:`../get-started/troubleshooting` | Content decomposed into dedicated installation, runnable quickstarts, and troubleshooting checklists. |
| `viz_plotspec.md` | :doc:`../how-to/plot_with_plotspec` | PlotSpec guidance converted to a how-to focused on configuration plus telemetry links. |
| `architecture.md` | :doc:`../concepts/index` and :doc:`../concepts/telemetry` | Architecture material merged into the concepts index with telemetry instrumentation guidance. |
| *(new)* interpretation guide | :doc:`../how-to/interpret_explanations` | Primary resource for reading factual and alternative explanations, PlotSpec visuals, and telemetry provenance. |
| `pr_guide.md` | :doc:`release_notes` and :doc:`section_owners` | Governance actions captured under maintainership docs with owners and release narrative. |

## Verification checklist

- [ ] Confirm new or renamed pages are added to the appropriate section toctree.
- [ ] Ensure deprecated filenames include a short pointer to their replacement.
- [ ] Re-run the gating commands in :doc:`release_checklist` after updating this table.

## Shared fragments rollout

Use the `_shared` partials when editing audience landing hubs, README mirrors, or quickstarts:

- Include the canonical hero copy with `{{ hero_calibrated_explanations }}` at the top of README, Overview, practitioner, researcher, and contributor landing pages.
- End each of those pages—and any quickstart or interpretation guide that references telemetry—with the opt-in callouts from `{{ optional_extras_template }}`. Optional extras must remain the final section in the page outline.
- When creating new audience or quickstart pages, add them to the verification checklist above and confirm the shared fragments render correctly by running `sphinx-build -b html -W`.
