# Release notes

Highlights for recent releases are summarised here. For the complete list of
changes see the project changelog on GitHub.

## v0.9.0

- Documentation CI now enforces notebook linting, docstring coverage (≥94%), and
  optional-extras placement checks so telemetry, PlotSpec, and plugin tooling
  stay clearly labelled as opt-in guardrails.
- Release automation exercises the aggregated ``external-plugins`` extra to
  confirm the curated fast-explanation bundle remains an optional install with
  the expected dependencies.
- Runtime performance controls—the calibrator cache, multiprocessing toggle, and
  FAST vectorised perturbations—stay opt-in in v0.9.0. Follow
  :doc:`../how-to/tune_runtime_performance` to enable them deliberately and to
  record rollback steps in your change log.
- For the full milestone scope—including calibrated-explanations-first messaging
  and governance updates—see the `v0.9.0 release plan <../../improvement_docs/RELEASE_PLAN_v1.md#v090-documentation-realignment-targeted-runtime-polish>`_.

## v0.8.0

- Adopted the role-based documentation information architecture defined in
  ADR-022.
- Introduced PlotSpec as the default renderer with telemetry coverage for
  fallback chains.
- Expanded telemetry payloads to include preprocessing snapshots and plugin
  provenance.
- Published a practitioner-focused interpretation guide that explains calibrated
  predictions, interval semantics, alternative rules, and telemetry provenance;
  README quick-starts and notebooks now link to it as the primary learning path.

Refer to the `CHANGELOG.md <https://github.com/Moffran/calibrated_explanations/blob/main/CHANGELOG.md>`_
for historical entries and patch release details.
