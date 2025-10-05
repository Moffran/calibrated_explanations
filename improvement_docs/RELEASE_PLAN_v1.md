# Release Plan to v1.0.0

Last updated: 2025-10-07
Maintainers: Core team
Scope: Concrete steps from v0.6.0 to a stable v1.0.0 with plugin-first execution.

## Current baseline (v0.6.0)

- Plugin orchestration is live for factual/alternative/fast modes with trusted
  in-tree adapters and runtime validation while legacy flows remain available via
  `_use_plugin=False`.【F:src/calibrated_explanations/core/calibrated_explainer.py†L388-L420】【F:src/calibrated_explanations/core/calibrated_explainer.py†L520-L606】
- WrapCalibratedExplainer preserves the public batch-first predict/predict_proba
  and explain helpers with keyword-compatible behaviour, ensuring enterprise
  wrappers keep working.【F:src/calibrated_explanations/core/wrap_explainer.py†L260-L471】
- Exception taxonomy and schema commitments are stable (`core.exceptions`,
  schema v1 docs/tests).【F:src/calibrated_explanations/core/exceptions.py†L1-L63】【F:docs/schema_v1.md†L1-L120】
- Interval plugins are defined but not yet resolved through the runtime; plot
  routing still defaults to the legacy adapter path.【F:src/calibrated_explanations/plugins/intervals.py†L1-L80】【F:src/calibrated_explanations/core/calibration_helpers.py†L1-L78】【F:src/calibrated_explanations/core/calibrated_explainer.py†L680-L720】

## Guiding principles

- Maintain compatibility for the v0.6.x OSS series (no breaking contract changes
  before v0.7). Honour ADR-005 schema, WrapCalibratedExplainer surface, and core
  exceptions.
- Keep plugin trust and telemetry hooks intact: mode/task metadata, interval and
  plot hints, `PredictBridge` monitoring.
- Avoid scope creep (no new ML strategies) so we can reach v1.0.0 with a polished
  plugin stack, documentation, and support tooling.

## Release milestones

### v0.6.x (stabilisation patches)

- Hardening: add regression tests for plugin parity, schema validation, and
  WrapCalibratedExplainer keyword defaults.
- Documentation polish: refresh plugin guide with registry/CLI examples and note
  compatibility guardrails.
- No behavioural changes beyond docs/tests.

### v0.7.0 (interval & configuration integration)

1. Implement interval plugin resolution and fast-mode reuse per
   `PLUGIN_GAP_CLOSURE_PLAN` step 1, ensuring calibrators resolve via registry and
   trusted fallbacks.【F:improvement_docs/PLUGIN_GAP_CLOSURE_PLAN.md†L24-L43】
2. Surface interval/plot configuration knobs (keywords, env vars, pyproject) and
   propagate telemetry metadata for `interval_source`/`proba_source`.【F:improvement_docs/PLUGIN_GAP_CLOSURE_PLAN.md†L45-L61】
3. Wire CLI console entry point and smoke tests; document usage in README and
   contributing guides.【F:improvement_docs/PLUGIN_GAP_CLOSURE_PLAN.md†L63-L70】
4. Update ADR-013/ADR-015 statuses to Accepted with implementation notes.

Release gate: parity tests green for factual/alternative/fast, interval override
coverage exercised, CLI packaging verified.

### v0.8.0 (plot routing & telemetry completeness)

1. Promote PlotSpec builders to default for at least factual/alternative plots
   while keeping legacy style available as fallback.【F:src/calibrated_explanations/core/calibrated_explainer.py†L680-L720】【F:src/calibrated_explanations/viz/builders.py†L150-L208】
2. Ensure explain* APIs emit CE-formatted intervals when percentile arguments are
   provided; extend tests covering dict payloads and telemetry fields.【F:src/calibrated_explanations/core/calibrated_explainer.py†L760-L820】
3. Document telemetry schema (interval_source/proba_source/plot_source) for
   enterprise integrations and provide examples in docs/plugins.md.
4. Review preprocessing persistence contract (ADR-009) to confirm saved
   preprocessor metadata matches expectations.【F:improvement_docs/adrs/ADR-009-input-preprocessing-and-mapping-policy.md†L1-L80】

Release gate: PlotSpec default route parity, telemetry docs/tests in place.

### v0.9.0 (docs, packaging, performance polish)

1. Finalise documentation workflow per ADR-012 (CI build, gallery/linkcheck) and
   ensure plugin/telemetry pages are cross-linked.【F:improvement_docs/adrs/ADR-012-documentation-and-gallery-build-policy.md†L1-L80】
2. Publish plugin authoring guide + cookiecutter or scaffolding tasks (stretch
   from plugin gap plan).【F:improvement_docs/PLUGIN_GAP_CLOSURE_PLAN.md†L72-L78】
3. Reassess optional perf features (caching/parallel) and either mark ADR-003/004
   as deferred beyond v1 or land minimal opt-in implementations guarded by docs.【F:improvement_docs/adrs/ADR-003-caching-key-and-eviction.md†L1-L64】【F:improvement_docs/adrs/ADR-004-parallel-backend-abstraction.md†L1-L64】
4. Publish migration notes summarising plugin configuration defaults and
   remaining legacy escapes.

Release gate: Docs CI green, packaging metadata includes CLI, migration guide
available.

### v1.0.0 (stability declaration)

1. Announce stable plugin/telemetry contracts and freeze Explanation Schema v1
   (patch-only updates afterwards).【F:docs/schema_v1.md†L1-L120】
2. Verify wrap interfaces and exception taxonomy unchanged, update README &
   CHANGELOG with compatibility statement.【F:src/calibrated_explanations/core/wrap_explainer.py†L260-L471】【F:src/calibrated_explanations/core/exceptions.py†L1-L63】
3. Provide upgrade checklist covering environment variables, pyproject settings,
   and CLI usage.
4. Tag release and backport documentation to enterprise extension repositories.

Release gate: No pending high-priority bugs, docs/tests/telemetry stable, plugin
registry feature-complete for explanation/interval/plot categories.

## Post-1.0 considerations

- Consider enabling caching/parallel features by default once performance data
  supports it (may require updated ADRs).【F:improvement_docs/adrs/ADR-003-caching-key-and-eviction.md†L1-L64】【F:improvement_docs/adrs/ADR-004-parallel-backend-abstraction.md†L1-L64】
- Evaluate additional renderer plugins (plotly) after verifying PlotSpec default
  adoption.
- Plan schema v2 requirements with enterprise consumers before making breaking
  changes.
