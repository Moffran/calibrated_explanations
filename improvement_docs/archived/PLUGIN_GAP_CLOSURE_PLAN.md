> **Status note (2025-10-24):** Last edited 2025-10-24 · Archive after: Re-evaluate post-v1.0.0 maintenance review · Implementation window: v0.9.0–v1.0.0.

# Plugin Solution Gap Closure Plan

Last updated: 2025-10-05
Status: Draft roadmap owned by core maintainers

## Context snapshot

- The explainer now routes explain* flows through explanation plugins with
  immutable request/context objects and runtime validation, keeping legacy
  behaviour accessible via `_use_plugin=False` escapes.【F:src/calibrated_explanations/core/calibrated_explainer.py†L388-L420】【F:src/calibrated_explanations/core/calibrated_explainer.py†L520-L606】
- Built-in adapters register factual/alternative/fast explanation plugins plus
  interval and legacy plot plugins so parity paths keep working out of the box.【F:src/calibrated_explanations/plugins/builtins.py†L120-L183】【F:src/calibrated_explanations/plugins/builtins.py†L360-L400】
- Interval plugins exist at the registry/protocol level, but the core calibration
  helpers still instantiate VennAbers/IntervalRegressor directly without running
  through the registry selection logic.【F:src/calibrated_explanations/plugins/intervals.py†L1-L80】【F:src/calibrated_explanations/core/calibration_helpers.py†L1-L78】
- Plot resolution defers to metadata fallbacks (`legacy`), yet no pyproject/env
  knobs exist for interval/plot selection and the CLI is only reachable via
  `python -m` because no console entry point is exposed.【F:src/calibrated_explanations/core/calibrated_explainer.py†L680-L720】【F:pyproject.toml†L1-L74】【F:src/calibrated_explanations/plugins/cli.py†L75-L145】

## Goals

1. Align runtime interval selection with ADR-013 so calibrators resolve through
   the same registry/trust machinery used by explanation plugins.
2. Finish configuration/CLI ergonomics so operators can set interval/plot
   plugins via keywords, environment variables, or project config instead of
   editing code.
3. Extend validation/telemetry hooks to cover interval/plot execution and update
   docs/ADR statuses accordingly.
4. Prepare the external plugin ecosystem with a discoverable folder, aggregated
   installation extras, and documentation placeholders that reinforce
   calibrated-explanations guardrails.

## Step-by-step plan

### 1. Wire interval plugin resolution into calibration

1. Introduce `_build_interval_chain` / `_resolve_interval_plugin` helpers on
   `CalibratedExplainer`, mirroring the explanation chain builder.
2. Update `calibration_helpers.initialize_interval_learner` and
   `initialize_interval_learner_for_fast_explainer` to request calibrators from
   the resolved plugin instead of constructing VennAbers/IntervalRegressor
   inline. Keep behaviour identical by delegating to the legacy plugins when no
   override is provided.【F:src/calibrated_explanations/core/calibration_helpers.py†L1-L78】
3. Capture returned calibrators inside `IntervalCalibratorContext` metadata so
   FAST mode still reuses per-feature calibrators without mutating shared
   objects.
4. Add regression/unit tests that simulate interval override selection,
   ensuring trusted/untrusted filters and capability gates behave as in ADR-013.

### 2. Expose interval/plot configuration surfaces

1. Extend `CalibratedExplainer.__init__` to accept `interval_plugin`,
   `fast_interval_plugin`, and `plot_style` keyword overrides and normalise them
   alongside the existing explanation overrides.【F:src/calibrated_explanations/core/calibrated_explainer.py†L388-L418】
2. Read `[tool.calibrated_explanations.intervals]` / `.plots` tables from
   `pyproject.toml` and new environment variables (`CE_INTERVAL_PLUGIN`,
   `CE_PLOT_STYLE`, etc.) to populate fallback chains for interval and plot
   resolution.
3. Expand `_build_explanation_context` to record the final interval/plot
   identifiers actually used so telemetry can tag `interval_source` and
   `proba_source` consistently.
4. Update docs (`docs/plugins.md`) with configuration examples and migrate ADRs
   to Accepted status once the runtime surfaces match the specification.

### 3. Package and tooling polish

1. Publish the plugin CLI as a console script (e.g., `ce.plugins`) to avoid the
   `python -m` dance and document usage in README/Contributing.【F:pyproject.toml†L1-L74】【F:src/calibrated_explanations/plugins/cli.py†L75-L145】
2. Add smoke tests covering the CLI commands against a temporary registry to
   guarantee packaging changes do not regress.
3. Audit telemetry payloads to ensure explanation outputs keep emitting
   interval/percentile metadata; extend tests so percentile arguments surface
   low/high fields with CE-formatted payloads.
4. Close the loop by updating ADR-013/ADR-015 status blocks and noting
   implementation checkpoints for future contributors.
5. Implement the `CE_DENY_PLUGIN` registry toggle described in ADR-006 and wire
   it into CLI trust workflows so operators can explicitly block unsafe
   extensions prior to v1.0.0.【F:improvement_docs/adrs/ADR-006-plugin-registry-trust-model.md†L40-L76】
6. Create an `external_plugins/` namespace containing README and installation
   metadata, define the `external-plugins` extras group in packaging configs, and
   add documentation placeholders that list bundled plugins while reiterating the
   calibration contract.

### 4. Stretch follow-ups (post v0.7 hardening)

- Provide example third-party plugins (interval + explanation) in docs/tests to
  exercise trust/onboarding guidance end-to-end.
- Expose registry bootstrap hooks for enterprise extensions so they
  can auto-register trusted plugins when packages are installed.
- Consider shipping a light plugin-scaffold cookiecutter or CLI command once the
  APIs stabilise.
