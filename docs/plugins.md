# Plugin registry, trust model, and ADR protocols

Calibrated explanations ship with an in-process plugin registry that now covers
explanation strategies (ADR-015), interval calibrators (ADR-013), and plotting
adapters (ADR-014). The registry keeps execution explicit: users opt-in by
registering plugins and, when appropriate, marking them as trusted. Metadata is
validated against the active schema version (`schema_version=1` today) so that
future or incompatible payloads fail fast with actionable messages.

## Registry overview

Plugins are keyed by identifiers (`core.explanation.factual`, `core.interval.fast`,
`legacy`, …) and expose structured metadata. The registry normalises mode names,
tracks declared tasks (`classification`, `regression`, or `both`), and records
specialised dependency hints such as `interval_dependency`, `plot_dependency`,
and per-mode fallback chains. These hints are used when constructing
`ExplanationContext` objects so explanation plugins can request the interval
calibrator or plot builder they expect without hard-coding imports.【F:src/calibrated_explanations/plugins/registry.py†L24-L171】【F:src/calibrated_explanations/core/calibrated_explainer.py†L482-L559】

The trusted state is stored alongside each descriptor. Trusted-only listings or
resolution paths can therefore exclude unreviewed plugins, and the CLI helpers
expose round-trip commands (`trust` / `untrust`) to toggle that bit without
restarting the process.【F:src/calibrated_explanations/plugins/registry.py†L243-L333】【F:src/calibrated_explanations/plugins/cli.py†L75-L145】

## Explanation plugin workflow (ADR-015)

Explanation plugins receive two frozen dataclasses from the core explainer:
`ExplanationContext` (static model metadata and dependency hints) and
`ExplanationRequest` (per-batch parameters). Plugins must call the provided
predict bridge to obtain calibrated predictions/intervals and return an
`ExplanationBatch` describing the explanation collection they produced.【F:src/calibrated_explanations/plugins/explanations.py†L23-L74】【F:src/calibrated_explanations/core/calibrated_explainer.py†L525-L608】

At runtime the explainer enforces ADR constraints:

* Plugin metadata is checked for schema compatibility, declared modes, tasks,
  and capability flags before the plugin instance is initialised. A plugin that
  omits `explanation:{mode}` or lacks `task:{classification,regression}` is
  rejected with a clear `ConfigurationError` before any predictions run.【F:src/calibrated_explanations/core/calibrated_explainer.py†L469-L559】
* A monitor wraps the predict bridge to ensure plugins actually route inference
  through the calibrated path. Batches produced without invoking
  `PredictBridge.predict`, `predict_interval`, or `predict_proba` trigger an
  error explaining the missing bridge call.【F:src/calibrated_explanations/core/calibrated_explainer.py†L104-L142】【F:src/calibrated_explanations/core/calibrated_explainer.py†L582-L606】
* Returned batches are validated (`validate_explanation_batch`) to confirm the
  container/explanation classes derive from the expected base types, metadata
  modes/tasks align with the active request, and any embedded container instance
  matches the declared class.【F:src/calibrated_explanations/plugins/explanations.py†L79-L132】【F:src/calibrated_explanations/core/calibrated_explainer.py†L582-L602】

These checks turn schema mismatches into precise diagnostics rather than silent
corruptions.

## Configuring plugin selection

Plugin selection is composed from multiple sources in priority order:

1. Keyword overrides on `CalibratedExplainer` (`factual_plugin=…`) which may be
   instances or identifiers.
2. Environment variables (`CE_EXPLANATION_PLUGIN_FACTUAL`, plus
   `…_FALLBACKS` for comma-separated chains).
3. `pyproject.toml` entries under `[tool.calibrated_explanations.explanations]`,
   where each mode can declare a primary identifier and fallback list.
4. Metadata-provided fallbacks declared by plugins themselves (via the
   `fallbacks` field).【F:src/calibrated_explanations/core/calibrated_explainer.py†L324-L418】

Dependency hints are propagated to interval and plot registries so the explainer
can align calibrators (`interval_dependency`) and preferred renderers
(`plot_dependency`) automatically.【F:src/calibrated_explanations/core/calibrated_explainer.py†L500-L541】

### CLI helpers

The registry ships with a small CLI that surfaces this metadata:

```bash
python -m calibrated_explanations.plugins.cli list            # list explanation/interval/plot plugins
python -m calibrated_explanations.plugins.cli show <id>       # inspect metadata for a specific plugin
python -m calibrated_explanations.plugins.cli trust <id>      # mark an explanation plugin as trusted
python -m calibrated_explanations.plugins.cli untrust <id>    # revoke trust for an explanation plugin
```

`list` accepts an optional category (`explanations`, `intervals`, `plots`, or
`all`) and a `--trusted-only` flag to focus on pre-authorised plugins. Output
includes dependency fields so operators can spot missing interval or plot
adapters before running large jobs.【F:src/calibrated_explanations/plugins/cli.py†L15-L145】

## Runtime validation & compatibility

Runtime guards surface actionable errors when plugins drift from ADR contracts:

* Unsupported `schema_version` or mismatched capabilities raise immediately,
  pointing to the offending identifier.
* Batches that report the wrong mode/task or embed containers of unexpected
  types are rejected before any consumer code touches them.
* Plugins that bypass the calibrated prediction bridge are blocked so outputs
  cannot silently diverge from calibrated expectations.【F:src/calibrated_explanations/core/calibrated_explainer.py†L469-L606】【F:src/calibrated_explanations/plugins/explanations.py†L79-L132】

These safeguards complement static registry validation and give CLI consumers a
way to audit the active configuration before running explanations.

## Legacy compatibility and migration

Legacy metadata aliases such as `"explanation:factual"` are still accepted but
now emit deprecation warnings and are normalised to canonical mode names. During
migration, explanation and plotting APIs retain a `use_legacy=True` escape hatch
that forces the original renderers; plugin metadata can also specify fallbacks to
`legacy` builders to replicate prior behaviour until custom adapters are
available.【F:src/calibrated_explanations/plugins/registry.py†L41-L170】【F:src/calibrated_explanations/_plots.py†L278-L559】

The default in-tree plugins (`core.explanation.factual`, `core.explanation.fast`,
`core.interval.fast`, etc.) remain trusted and provide parity with historical
flows. Third-party plugins can gradually opt in while benefitting from the
runtime validation described above.【F:src/calibrated_explanations/plugins/builtins.py†L120-L318】
