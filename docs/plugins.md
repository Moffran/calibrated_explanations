# Plugin registry, trust model, and ADR protocols

Calibrated explanations ship with an in-process plugin registry that now covers
explanation strategies (ADR-015), interval calibrators (ADR-013), and plotting
adapters (ADR-014). The registry keeps execution explicit: users opt-in by
registering plugins and, when appropriate, marking them as trusted. Metadata is
validated against the active schema version (`schema_version=1` today) so that
future or incompatible payloads fail fast with actionable messages.

Plot plugins now default to the PlotSpec pipeline: adapters receive a backend-agnostic
specification and render it without importing the core package. This separation keeps
plot builders focused on producing PlotSpec documents while adapters provide the
runtime glue, making the plugin surface explicit for downstream maintainers.

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

## Explanation plugin workflow (ADR-015 & ADR-026)

Explanation plugins receive two frozen dataclasses from the core explainer:
`ExplanationContext` (static model metadata and dependency hints) and
`ExplanationRequest` (per-batch parameters). ADR-026 documents the runtime
semantics expected from explanation plugins—when metadata is injected, how the
bridge monitor is enforced, and which collection hooks must be preserved—so new
implementations can reason about behaviour without cloning the legacy code. In
practice, plugins must call the provided predict bridge to obtain calibrated
predictions/intervals and return an `ExplanationBatch` describing the
explanation collection they produced.【F:src/calibrated_explanations/plugins/explanations.py†L23-L74】【F:src/calibrated_explanations/core/calibrated_explainer.py†L525-L608】【F:improvement_docs/adrs/ADR-026-explanation-plugin-semantics.md†L1-L153】

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

1. Keyword overrides on `CalibratedExplainer` which may be instances or
   identifiers: `factual_plugin`, `alternative_plugin`, `fast_plugin`, plus the
   interval/plot hooks `interval_plugin`, `fast_interval_plugin`, and
   `plot_style`.
2. Environment variables. Each explanation mode understands
   `CE_EXPLANATION_PLUGIN_<MODE>` and `CE_EXPLANATION_PLUGIN_<MODE>_FALLBACKS`;
   interval plugins honour `CE_INTERVAL_PLUGIN`, `CE_INTERVAL_PLUGIN_FALLBACKS`,
   `CE_INTERVAL_PLUGIN_FAST`, `CE_INTERVAL_PLUGIN_FAST_FALLBACKS`; plots use
   `CE_PLOT_STYLE` and `CE_PLOT_STYLE_FALLBACKS`.
3. `pyproject.toml` entries under `[tool.calibrated_explanations.explanations]`,
   `[tool.calibrated_explanations.intervals]`, and
   `[tool.calibrated_explanations.plots]` to declare primary identifiers and
   fallback lists.
4. Metadata-provided fallbacks declared by plugins themselves (via the
   `fallbacks`, `interval_dependency`, or `plot_dependency` fields).【F:src/calibrated_explanations/core/calibrated_explainer.py†L324-L418】【F:src/calibrated_explanations/core/calibrated_explainer.py†L652-L722】

Dependency hints are propagated to interval and plot registries so the explainer
can align calibrators (`interval_dependency`) and preferred renderers
(`plot_dependency`) automatically. The runtime also records the selected
`interval_source`, `proba_source`, and plot fallbacks on
`CalibratedExplainer.runtime_telemetry` and attaches the same payload to
returned `CalibratedExplanations` collections for downstream telemetry.【F:src/calibrated_explanations/core/calibrated_explainer.py†L1006-L1099】

### Telemetry payloads

Every call to `explain_*`, `predict`, or `predict_proba` refreshes the telemetry
dictionary returned by `CalibratedExplainer.runtime_telemetry` and attached to
the explanation collection. The payload mirrors the v0.8.0 data model:

| Key                | Description                                                                                 |
| ------------------ | ------------------------------------------------------------------------------------------- |
| `mode` / `task`    | Explanation mode (`factual`, `alternative`, `fast`) and learner task (`classification`/`regression`). |
| `interval_source`  | The identifier of the interval calibrator that produced the uncertainty bounds.            |
| `proba_source`     | Source identifier for calibrated probabilities (often the same as `interval_source`).      |
| `plot_source`      | The plot style that rendered the figure (defaults to `plot_spec.default.*`).                |
| `plot_fallbacks`   | Ordered tuple of plot style fallbacks (e.g. `("plot_spec.default.factual", "legacy")`).     |
| `interval_dependencies` | Tuple of interval plugin hints propagated from explanation metadata.                  |
| `uncertainty`      | Structured CE interval object containing calibrated value, bounds, percentiles, threshold metadata, and the backward-compatible `legacy_interval`. |
| `rules`            | Per-feature rule telemetry (factual and alternative) including feature-level uncertainty.  |
| `preprocessor`     | ADR-009 snapshot describing preprocessing (auto-encode policy, transformer id, optional mapping snapshot). |

Downstream services can therefore audit which plugin branches executed, confirm
PlotSpec routing versus legacy fallbacks, and capture preprocessing provenance
without probing runtime internals.【F:docs/concepts/telemetry.md†L1-L25】【F:src/calibrated_explanations/core/calibrated_explainer.py†L1098-L1138】
See also {doc}`concepts/telemetry` for schema details, and point practitioners to
{doc}`how-to/interpret_explanations` so they understand how the telemetry fields
translate into actionable insights.

### CLI helpers

The registry ships with a small CLI that surfaces this metadata:

```bash
ce.plugins list all            # list explanation/interval/plot plugins
ce.plugins show <id> --kind intervals
ce.plugins trust <id>          # mark an explanation plugin as trusted
ce.plugins untrust <id>        # revoke trust for an explanation plugin
```

`list` accepts an optional category (`explanations`, `intervals`, `plots`, or
`all`) and a `--trusted-only` flag to focus on pre-authorised plugins. Output
includes dependency fields so operators can spot missing interval or plot
adapters before running large jobs. Plot listings now surface
`is_default`, `legacy_compatible`, and `default_for` metadata so the active
PlotSpec default is discoverable at the CLI (JSON output includes the same
fields for automation-friendly consumption).【F:src/calibrated_explanations/plugins/cli.py†L15-L145】

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

## v0.6.x hardening checklist

Patch release 0.6.1 focuses on regression coverage and operational guidance to
keep plugin-first execution aligned with the legacy flows.

### Regression tests

- Ensure runtime parity between the plugin orchestrator and the legacy escape
  hatch by comparing `CalibratedExplainer` results with and without
  `_use_plugin` enabled:

  ```bash
  pytest tests/integration/core/test_explanation_parity.py::test_plugin_runtime_matches_legacy_factual \
         tests/integration/core/test_explanation_parity.py::test_plugin_runtime_matches_legacy_alternative \
         tests/integration/core/test_explanation_parity.py::test_plugin_runtime_matches_legacy_fast
  ```

- Keep schema validation sharp by exercising the v1 JSON Schema guardrails:

  ```bash
  pytest tests/unit/core/test_serialization_and_quick.py::test_validate_payload_rejects_missing_required_fields
  ```

- Lock in wrapper keyword defaults and alias handling when configs are used to
  spin up explainers:

  ```bash
  pytest tests/unit/core/test_wrap_keyword_defaults.py
  ```

### Operational notes

- The `_use_plugin=False` parameter remains supported as a safety valve for
  enterprise deployments. The parity tests above ensure that disabling the
  orchestrator still matches plugin-backed outputs.
- Schema validation remains optional at runtime, but installing
  `jsonschema>=4` enables the stricter guardrails covered by the regression
  suite.

## Legacy compatibility and migration

Legacy metadata aliases such as `"explanation:factual"` are still accepted but
now emit deprecation warnings and are normalised to canonical mode names. During
migration, explanation and plotting APIs retain a `use_legacy=True` escape hatch
that forces the original renderers; plugin metadata can also specify fallbacks to
`legacy` builders to replicate prior behaviour until custom adapters are
available.【F:src/calibrated_explanations/plugins/registry.py†L41-L170】【F:src/calibrated_explanations/plotting.py†L278-L559】

The default in-tree plugins (`core.explanation.factual`, `core.explanation.fast`,
`core.interval.fast`, etc.) remain trusted and provide parity with historical
flows. Third-party plugins can gradually opt in while benefitting from the
runtime validation described above.【F:src/calibrated_explanations/plugins/builtins.py†L120-L318】
