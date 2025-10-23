# Release checklist

Before publishing a minor release, complete the following documentation tasks in
line with ADR-022:

1. **Navigation audit** – verify each top-level section matches the information
   architecture and that new pages are linked from the appropriate toctree.
   Confirm the :doc:`nav_crosswalk` is updated for every legacy page that moved.
2. **Quickstart validation** – execute the classification and regression
   quickstarts (or CI equivalent) to ensure code snippets run without edits and
   update {doc}`../get-started/troubleshooting` with any new callouts.
3. **Interpretation guide audit** – confirm {doc}`../how-to/interpret_explanations`
   reflects the latest telemetry schema, alternative rule structure, and
   quickstart outputs. Ensure README and upgrade notes still link to it.
4. **Telemetry review** – confirm the telemetry schema page reflects the latest
   runtime keys and that plugin docs reference new identifiers.
5. **Reference refresh** – regenerate autosummary output, review schema version
   notes, and update API alias guidance.
6. **Gating checks** – block release unless the following pass locally or in CI:

   ```bash
   python -m sphinx -b html -W docs docs/_build/html
   python -m sphinx -b linkcheck docs docs/_build/linkcheck
   pytest tests/docs
   ```

7. **Ownership sign-off** – collect sign-off from the section owners listed in
   :doc:`../OWNERS`.
8. **Release notes** – summarise highlights and link to the release plan milestone
   (`improvement_docs/RELEASE_PLAN_v1.md`).

Document completion in the release issue template so regressions surface quickly
in future audits.
