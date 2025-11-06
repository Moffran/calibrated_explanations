> **Status note (2025-10-24):** Last edited 2025-10-24 · Archive after: Retain indefinitely as architectural record · Implementation window: Per ADR status (see Decision).

# ADR-013: Interval Calibrator Plugin Strategy

Status: Accepted
Date: 2025-09-16 (revised 2025-10-02)
Deciders: Core maintainers
Reviewers: TBD
Supersedes: ADR-013-interval-and-plot-plugin-strategy
Superseded-by: None
Related: ADR-006-plugin-registry-trust-model, ADR-015-explanation-plugin

## Context

Calibrated Explanations exposes two calibration backbones today: VennAbers for probabilistic classification (src/calibrated_explanations/core/venn_abers.py) and IntervalRegressor for regression-style interval predictors (src/calibrated_explanations/core/interval_regressor.py). Both classes are orchestrated from CalibratedExplainer and its helpers, and they encapsulate critical behaviours such as Mondrian bin handling, residual-based conformal updates, and difficulty-aware probability adjustment. Previous discussions bundled interval and plotting extensions into a single plugin plan, but practise shows that any interval plugin must compose these backbones directly. Drifting from the proven implementations would change the mathematics and invalidate the guarantees that define the package. ADR-021 documents the canonical semantics these backbones must expose (classification probabilities, percentile intervals, and thresholded regression probabilities); this ADR defines the plugin architecture that preserves those semantics.

IntervalRegressor already leans on VennAbers: the compute_proba_cal specialisations rebuild a binary VennAbers calibrator and predict_probability delegates to its predict_proba output to produce calibrated interval probabilities. Any extension mechanism therefore needs a shared contract that captures the classification semantics first and lets regression build on top of it.

We therefore need a plugin model that lets external contributors add new interval behaviours while still driving the existing VennAbers and IntervalRegressor logic, sharing their calibration guarantees, and fitting inside the trust model defined in ADR-006. The legacy classes must keep working verbatim — they are the mathematical reference implementation and the default code path until a future ADR explicitly changes the default.

## Decision

1. **Introduce layered calibrator protocols.**
   - Define an `IntervalCalibratorContext` dataclass with read-only fields: `learner`, `calibration_splits`, `bins`, `residuals`, `difficulty`, `metadata`, and `fast_flags`. The context is prepared by `CalibratedExplainer` and handed to plugins; implementations MUST NOT mutate the contained structures.
   - Define an `IntervalCalibratorPlugin` protocol with `create(self, context: IntervalCalibratorContext, *, fast: bool = False) -> ClassificationIntervalCalibrator`. The optional `fast` hint allows the runtime to request the reduced-computation path used by FAST while keeping the same protocol surface.
   - `ClassificationIntervalCalibrator` must expose the exact callable surface of VennAbers:
     ```python
     def predict_proba(
         self,
         x,
         *,
         output_interval: bool = False,
         classes=None,
         bins=None,
     ) -> numpy.ndarray
     ```
     Returning `(n_samples, n_classes)` arrays when `output_interval=False` and `(n_samples, n_classes, 3)` (`predict`, `low`, `high`) otherwise. Implementations also provide `is_multiclass() -> bool` and `is_mondrian() -> bool` accessors with the same semantics as VennAbers.
   - `RegressionIntervalCalibrator` extends the classification protocol with the IntervalRegressor surface:
     ```python
     def predict_probability(self, x) -> numpy.ndarray  # shape (n_samples, 2) ordered (low, high)
     def predict_uncertainty(self, x) -> numpy.ndarray  # shape (n_samples, 2) ordered (width, confidence)
     def pre_fit_for_probabilistic(self, x, y) -> None
     def compute_proba_cal(self, x, y, *, weights=None) -> numpy.ndarray
     def insert_calibration(self, x, y, *, warm_start: bool = False) -> None
     ```
     Implementations may wrap or subclass `IntervalRegressor`, but probability/interval calculations must delegate to the reference logic so conformal guarantees remain intact.
   - When a plugin advertises regression support its returned calibrator object must satisfy the regression protocol. This ensures regression plugins necessarily expose the classification machinery they already depend on, preventing behavioural drift between modes.
   - Plugins can augment these calibrators (e.g., swapping the conformal engine used by IntervalRegressor), but they are required to invoke the base class logic when producing probabilities or intervals to preserve the calibrated guarantees. Any plugin that replaces those internals must demonstrate compatibility with the semantics in ADR-021 and document how classification, percentile regression, and thresholded regression expectations are upheld.
   - **Legacy default:** ship `DefaultIntervalCalibratorPlugin` inside the package. `create(context)` simply returns the frozen in-tree IntervalRegressor / VennAbers instances that are already constructed by the explainer. This plugin is registered under the identifier `core.interval.legacy` and marked as trusted. It is the mandatory fallback whenever resolution fails or no explicit plugin is configured.
   - **FAST plugin (second wave):** ship `FastIntervalCalibratorPlugin` that layers the FAST heuristics on top of the shared protocols. It reuses the existing IntervalRegressor/VennAbers instances but applies the reduced-computation path used by the FAST framework. The plugin registers as `core.interval.fast`, is flagged `fast_compatible=True`, and is never part of the fallback chain for the primary interval mode so FAST remains an opt-in experience separate from the legacy flow.

2. **Registry integration and metadata.**
   - Extend calibrated_explanations.plugins.registry with register_interval_plugin, find_interval_plugin, and find_interval_plugin_trusted helpers mirroring ADR-006 semantics.
   - Plugin descriptors must publish metadata fields: modes (classification, regression, or both), fast_compatible, requires_bins, confidence_source, dependencies, optional `interval_dependency` (identifier string), and shared keys from ADR-006 (name, provider, schema_version, capability tags such as interval:classification or interval:regression).
   - Trusted defaults register at import time. Third party plugins must be explicitly trusted before selection.
   - `DefaultIntervalCalibratorPlugin` self-registers during package import so that configuration toggles can reference it by identifier. `FastIntervalCalibratorPlugin` also self-registers under `core.interval.fast` with `fast_compatible=True` metadata, but it is not added to the default fallback chain so FAST activation is always explicit. These registrations keep the runtime behaviour identical to the legacy code path when no overrides are supplied.

3. **Configuration surfaces.**
   - Respect configuration entry points scoped to intervals only:
     - Environment: CE_INTERVAL_PLUGIN for the default plugin, CE_INTERVAL_PLUGIN_FAST for the fast path adaptor, and CE_INTERVAL_PLUGIN_FALLBACKS (comma separated) for ordered fallback resolution.
     - CalibratedExplainer keyword arguments: interval_plugin and fast_interval_plugin accept identifiers or callables returning a plugin instance.
     - pyproject.toml ([tool.calibrated_explanations.plugins] interval, fast_interval, and optional interval_fallbacks). CLI helpers hydrate these settings at start-up.
   - When no explicit fast plugin is configured, the resolver selects `core.interval.fast` exclusively for the FAST execution path; the primary interval plugin continues to default to `core.interval.legacy`.
   - Explanation plugins may declare an `interval_dependency` metadata field (ADR-015). When present, the interval resolver prepends that identifier to the fallback chain for the matching explanation mode. `core.explanation.fast` advertises `interval_dependency="core.interval.fast"`, ensuring FAST explanations automatically pair with the FAST calibrator unless the user overrides either selection.
   - If resolution fails, raise a configuration error after attempting the fallback chain and, finally, the legacy plugin. This protects the longstanding behaviour while still surfacing misconfiguration details to the user.

4. **Lifecycle hooks and validation.**
   - During calibration setup the registry resolves the desired plugin, instantiates it with the context, and asserts that the returned object implements the protocol required for the explainer mode. Missing capabilities raise a configuration error before calibration begins.
   - All calibrator methods must proxy to the underlying VennAbers and IntervalRegressor contracts. Runtime validation checks shapes, dtypes, and monotonicity of probability outputs against expectations from the in-tree classes. Fast-mode calibrators reuse the same validation but may skip expensive recomputation steps when metadata marks them fast_compatible.
   - Incremental calibration hooks (insert_calibration, CPS updates) remain the responsibility of IntervalRegressor; plugin authors extending those flows must invoke the superclass logic and only layer extra behaviour afterwards.
   - Provide a `LegacyIntervalContext` adaptor that wraps the existing `CalibratedExplainer` state without mutating it. Plugins receive a frozen view (read-only mappings, tuples) so they cannot accidentally leak mutations back into the explainer. The default plugin simply stores the objects it receives and returns them during `create()`.

5. **Documentation and tooling.**
   - Developer docs gain a dedicated section that explains the layered protocol, the context object, and examples that subclass and compose IntervalRegressor / VennAbers.
   - CLI commands (ce.plugins list --intervals, ce.plugins explain-interval --plugin <id>) expose plugin availability and validation results to help detect misconfigured packages.
   - Author migration guidance comparing “pre-plugin” helper functions to the new adaptor to make it clear that simply importing the default plugin yields the legacy behaviour.

## Security & Guardrails

- Trust remains opt-in per ADR-006: only plugins explicitly marked as trusted are eligible for automatic selection.
- Metadata validation occurs during registration. Missing required fields keeps the plugin out of the registry, preventing accidental downgrades to unsafe implementations.
- Runtime guards assert that calibrators return probabilities and intervals identical in structure to the built-in classes. Failures bubble up with actionable diagnostics so the system never silently downgrades calibration quality.

## Interval Propagation to Explanation Rules

### Factual Explanations

Interval calibrators produce calibrated probabilities and intervals for
the original instance prediction. These same intervals propagate to
feature-level weights in factual rules:

- When a factual feature rule is computed via perturbation, the weight
  (feature attribution) is accompanied by an uncertainty interval derived
  from the calibrated interval learner.
- The feature weight interval semantics match the underlying calibrator
  (Venn-Abers for classification, CPS for regression).

### Alternative Explanations

Each alternative scenario is evaluated with the same interval calibrator:

- When an alternative condition is tested, the calibrated prediction for
  that scenario includes both the point estimate and uncertainty interval.
- The uncertainty interval for each alternative prediction is produced by
  the interval calibrator, ensuring consistency with the original instance
  prediction interval.
- Feature-weight deltas may be computed but are auxiliary; the primary
  payload is the calibrated prediction interval for the alternative scenario.

## Implementation status (2025-10-10)

- `CalibratedExplainer` now resolves both legacy and FAST interval calibrators
  through the registry, honouring environment/pyproject overrides and metadata
  fallbacks.【F:src/calibrated_explanations/core/calibrated_explainer.py†L412-L513】【F:src/calibrated_explanations/core/calibration_helpers.py†L18-L88】
- Interval plugin identifiers are tracked for telemetry (`interval_source` /
  `proba_source`) and exposed alongside explanation results.【F:src/calibrated_explanations/core/calibrated_explainer.py†L1006-L1099】
- Registry protocols and trusted defaults remain the same; no sandboxing or
  distribution changes are required.【F:src/calibrated_explanations/plugins/intervals.py†L1-L80】【F:src/calibrated_explanations/plugins/builtins.py†L120-L183】

## Consequences

Positive:
- Clear contracts let contributors experiment with conformal variants, Bayesian posteriors, or domain specific calibrations without rewriting the integration path.
- Users can swap interval behaviours while keeping CalibratedExplainer APIs and guarantees stable.
- Centralised metadata and validation improve tooling visibility and protect against misconfigured plugins.

Negative / Risks:
- The adapter layer adds indirection during calibration, increasing the number of moving parts to debug.
- Enforcing conformity with IntervalRegressor/VennAbers may limit radical experimentation; a future ADR can define an escape hatch if needed.
- Additional registry metadata and validation code must be maintained alongside core calibration APIs.

## Non-Goals

- Defining new explanation types outside probabilistic intervals and regression intervals. Broader explainer plugins will be covered separately if needed.
- Shipping sandboxing, isolation, or remote plugin distribution mechanisms.
- Replacing the built-in calibrators; they remain the reference implementation and default plugin.

## Future Work

- Extend integration tests to cover third-party interval plugins with custom trust
  policies and capability matrices.
- Evaluate the need for caching hooks or memoisation in the context object once additional plugins exist.
