# Release checklist

Before publishing a minor release, complete the following documentation and
governance tasks in line with ADR-012, ADR-017, ADR-018, ADR-019, and the
v0.9.0 release plan:

1. **Navigation audit** – verify each top-level section matches the information
   architecture and that new pages are linked from the appropriate toctree.
   Confirm the :doc:`nav_crosswalk` is updated for every legacy page that moved.
2. **Hero & research hub verification** – confirm README plus the practitioner,
   researcher, and contributor landing hubs render the shared hero copy,
   dual quickstart table, interval regression signpost, and the existing
   research hub link in its usual spot (hero body or Resources callout). This
   step is a release gate for the calibrated-explanations-first narrative
   (FR-001 – FR-005).
3. **Optional extras** – Optional tooling (telemetry, PlotSpec,
   CLI, external plugins) must not appear early in the flow.
4. **Alternatives storytelling** – check every `explore_alternatives` snippet in
   docs and notebooks uses the triangular plot to complement the standard plot.
5. **Quickstart validation** – execute the classification and regression
   quickstarts (or CI equivalent) to ensure code snippets run without edits and
   update {doc}`../../get-started/troubleshooting` with any new callouts. Confirm
   interval regression guidance sits adjacent to probabilistic regression steps.
   Cross-check {doc}`test_policy` so every updated snippet has a matching
   `tests/docs/` module.
6. **Interpretation guide audit** – confirm {doc}`../how-to/interpret_explanations`
   reflects the latest telemetry schema, alternative rule structure, triangular
   plot copy, and quickstart outputs. Ensure README, quickstarts, and landing
   hubs still link to it.
7. **Telemetry & plugin governance review** – verify the telemetry schema page
   exposes current keys, plugin docs cite ADR-024/025/026, `CE_DENY_PLUGIN`
   messaging remains opt-in, and the external plugins catalogue lists the
   aggregated install extra.
8. **Runtime governance** – coordinate with the runtime tech lead to audit the
   coverage waiver log at code freeze. Record the sign-off (or waivers retired)
   in the release issue and update :doc:`waivers_inventory` if entries changed.
9. **Reference refresh** – regenerate autosummary output, review schema version
   notes, and update API alias guidance to maintain ADR-017/ADR-018 compliance.
10. **Gating checks** – block the release unless the following pass locally or in
    CI (no warnings allowed):

      ```bash
      python -m sphinx -b html -W docs docs/_build/html
      python -m sphinx -b linkcheck docs docs/_build/linkcheck
      pytest tests/docs
      pytest --cov --cov-fail-under=90
      ```

    Capture the `pytest --cov` summary for the release notes and update the
    public documentation coverage badge if the percentage changed.
11. **Ownership sign-off** – collect sign-off from the section owners listed in
    :doc:`../OWNERS`, including the runtime tech lead acknowledgement from step 8.
12. **Release notes** – summarise highlights, note telemetry/plugin optionality,
    cite the cache & parallel toggles as opt-in, and link to the release plan
    milestone (`docs/improvement/RELEASE_PLAN_v1.md`).

Document completion in the release issue template so regressions surface quickly
in future audits.
