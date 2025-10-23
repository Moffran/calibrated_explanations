# OSS Calibrated Explanations: Scope and Gaps

## What It Is

- Purpose: Local explanations with calibrated predictions and uncertainty. Minimizes aleatoric uncertainty via calibration, and surfaces epistemic uncertainty via per-rule intervals, for classification, regression, and thresholded regression (probability of an event like y ≤ t). See `README.md` Introduction; examples in quickstart and telemetry.
- Core APIs:
  - `CalibratedExplainer`: `explain_factual`, `explore_alternatives`, `explain_fast`; prediction APIs with optional intervals; thresholded regression probabilities. See `src/calibrated_explanations/core/calibrated_explainer.py:1683`.
  - `WrapCalibratedExplainer`: scikit‑learn‑style fit/calibrate/explain wrapper with permissive validation and preprocessing snapshot support (ADR‑009). See `src/calibrated_explanations/core/wrap_explainer.py:24`.
  - `quick_explain`: fit+calibrate+factual in one call. See `src/calibrated_explanations/api/quick.py:17`.

## Architecture & Plugins

- Plugin-first runtime: Explanation plugins (ADR‑015), Interval calibrator plugins (ADR‑013), Plot styles/builders/renderers (ADR‑014). Registry with trust model (ADR‑006). See `src/calibrated_explanations/plugins/registry.py:433`.
- Explanation plugins get frozen context + a `PredictBridge` they must use; runtime validates plugin meta (schema_version, modes/tasks/capabilities), asserts bridge usage, and validates returned batches. See `src/calibrated_explanations/core/calibrated_explainer.py:571`.
- Interval plugins wrap Venn‑Abers (classification) and IntervalRegressor (regression) semantics with fast‑mode reuse. See `src/calibrated_explanations/plugins/intervals.py:7`.
- Built‑ins mirror legacy behavior and are trusted: `core.explanation.{factual,alternative,fast}`, `core.interval.{legacy,fast}`, plot styles `{legacy, plot_spec.default}`. See `src/calibrated_explanations/plugins/builtins.py:744`.

## Data & Serialization

- Explanation Schema v1: stable JSON envelope for portability; helpers to `to_json`/`from_json` with optional `jsonschema` validation. See `src/calibrated_explanations/serialization.py:30`, `docs/schema_v1.md:1`.
- Domain models + adapters exist for round‑trips and legacy↔domain conversion. See `src/calibrated_explanations/explanations/models.py:33`.

## Visualization

- PlotSpec default builders + adapter to Matplotlib, with legacy fallback path preserved. See `src/calibrated_explanations/viz/builders.py:1`.
- Plot routing records style + fallbacks in telemetry; style can be forced via env/pyproject. See `src/calibrated_explanations/core/calibrated_explainer.py:1043`.

## Telemetry & Config

- Telemetry attached to each batch and mirrored on runtime: `mode`/`task`, `interval_source`/`proba_source`, `plot_source`/`plot_fallbacks`, `uncertainty` payload, preprocessor snapshot (ADR‑009). See `docs/concepts/telemetry.md:15`.
- Config layering: kwargs > env (`CE_EXPLANATION_PLUGIN_*`, `CE_INTERVAL_PLUGIN*`, `CE_PLOT_STYLE*`) > pyproject tables. See `src/calibrated_explanations/core/calibrated_explainer.py:425`.

## Interval Semantics

- Canonical semantics (ADR‑021): Venn‑Abers for classification; CPS percentile intervals for regression; thresholded regression uses Venn‑Abers‑calibrated probabilities over threshold events. See `improvement_docs/adrs/ADR-021-calibrated-interval-semantics.md:1`.

## Performance Scaffolding

- Perf module introduces LRU cache + parallel backend abstractions (opt‑in; off by default); plans include vectorization and batching in Python path to meet SLAs without C extensions. See `README.md` Performance scaffolding; `src/calibrated_explanations/perf/__init__.py:1`.

## Tooling & CLI

- Plugin CLI lists/inspects, trusts/untrusts explanation plugins; shows schema/trust/capabilities and fallbacks. See `src/calibrated_explanations/plugins/cli.py:172`.
- Docs IA, governance, coverage guardrails (ADR‑019), and docstring standards (ADR‑018) are in flight per release plan. See `improvement_docs/RELEASE_PLAN_v1.md:1`.

## What’s Obviously Missing (not covered by current plans)

- Trust toggles for interval/plot plugins in CLI
  - CLI supports trust/untrust only for explanation plugins; there is no symmetric command for intervals or plot components. See `src/calibrated_explanations/plugins/cli.py:160`, `src/calibrated_explanations/plugins/cli.py:16`.
  - Suggest: add `mark_interval_trusted`/`untrusted` and plot equivalents in the registry + CLI parity.
- Entry‑point discovery for interval/plot plugins
  - Entry‑point loader only discovers “explainer plugins”; there’s no entry‑point path for identifier‑keyed interval or plot plugins. See `src/calibrated_explanations/plugins/registry.py:883`.
  - Suggest: add separate groups or a unified descriptor loader for intervals/builders/renderers/styles.
- Plugin denylist
  - ADR‑006 calls out a denylist env var as “probably yes” for defense‑in‑depth, but it’s not implemented. See `improvement_docs/adrs/ADR-006-plugin-registry-trust-model.md:53`.
  - Suggest: implement `CE_DENY_PLUGIN` (or similar) in the registry filters.
- Streaming/generator batches for large explanations
  - ADR‑015 notes the batch shape is intentionally compatible with a future streaming provider, but there’s no plan to ship streaming for memory‑sensitive runs. See `improvement_docs/adrs/ADR-015-explanation-plugin.md:137`.
  - Suggest: optional generator‑based instances or chunked materialization.
- First‑class convenience export on explanation objects
  - Today, schema v1 export requires adapters + serialization helpers; there’s no simple `exp.to_json()`/`collection.to_json()` convenience on public objects.
  - Suggest: add a thin convenience that wraps existing adapters/serialization for usability.
