# ADR-013: Interval Calibrator Plugin Strategy

Status: Proposed
Date: 2025-09-16
Deciders: Core maintainers
Reviewers: TBD
Supersedes: ADR-013-interval-and-plot-plugin-strategy
Superseded-by: None
Related: ADR-006-plugin-registry-trust-model

## Context

Calibrated Explanations exposes two calibration backbones today: VennAbers for probabilistic classification (src/calibrated_explanations/_VennAbers.py) and IntervalRegressor for regression-style interval predictors (src/calibrated_explanations/_interval_regressor.py). Both classes are orchestrated from CalibratedExplainer and its helpers, and they encapsulate critical behaviours such as Mondrian bin handling, residual-based conformal updates, and difficulty-aware probability adjustment. Previous discussions bundled interval and plotting extensions into a single plugin plan, but practise shows that any interval plugin must compose these backbones directly. Drifting from the proven implementations would change the mathematics and invalidate the guarantees that define the package.

IntervalRegressor already leans on VennAbers: the compute_proba_cal specialisations rebuild a binary VennAbers calibrator and predict_probability delegates to its predict_proba output to produce calibrated interval probabilities. Any extension mechanism therefore needs a shared contract that captures the classification semantics first and lets regression build on top of it.

We therefore need a plugin model that lets external contributors add new interval behaviours while still driving the existing VennAbers and IntervalRegressor logic, sharing their calibration guarantees, and fitting inside the trust model defined in ADR-006.

## Decision

1. **Introduce layered calibrator protocols.**
   - Define an IntervalCalibratorPlugin protocol whose create(context) method returns an object implementing ClassificationIntervalCalibrator. The context exposes read-only access to the learner, calibration splits, bins, residuals, and difficulty estimators so the plugin does not reimplement orchestration.
   - ClassificationIntervalCalibrator captures the callable surface of VennAbers: predict_proba(X, *, output_interval=False, classes=None, bins=None), and helper introspection (is_multiclass(), is_mondrian()). Implementations must return numpy arrays with the same shapes and interval semantics as the in-tree VennAbers.
   - RegressionIntervalCalibrator extends ClassificationIntervalCalibrator and adds the extra methods required by IntervalRegressor: predict_probability, predict_uncertainty, pre_fit_for_probabilistic, compute_proba_cal, and insert_calibration. Implementations compose or subclass IntervalRegressor so Mondrian bins, CPS updates, probability post-processing, and incremental calibration continue to flow through the existing logic.
   - When a plugin advertises regression support its returned calibrator object must satisfy the regression protocol. This ensures regression plugins necessarily expose the classification machinery they already depend on, preventing behavioural drift between modes.
   - Plugins can augment these calibrators (e.g., swapping the conformal engine used by IntervalRegressor), but they are required to invoke the base class logic when producing probabilities or intervals to preserve the calibrated guarantees.

2. **Registry integration and metadata.**
   - Extend calibrated_explanations.plugins.registry with register_interval_plugin, find_interval_plugin, and find_interval_plugin_trusted helpers mirroring ADR-006 semantics.
   - Plugin descriptors must publish metadata fields: modes (classification, regression, or both), fast_compatible, requires_bins, confidence_source, dependencies, and shared keys from ADR-006 (name, provider, schema_version, capability tags such as interval:classification or interval:regression).
   - Trusted defaults register at import time. Third party plugins must be explicitly trusted before selection.

3. **Configuration surfaces.**
   - Respect configuration entry points scoped to intervals only:
     - Environment: CE_INTERVAL_PLUGIN for the default plugin, CE_INTERVAL_PLUGIN_FAST for the fast path adaptor, and CE_INTERVAL_PLUGIN_FALLBACKS (comma separated) for ordered fallback resolution.
     - CalibratedExplainer keyword arguments: interval_plugin and fast_interval_plugin accept identifiers or callables returning a plugin instance.
     - pyproject.toml ([tool.calibrated_explanations.plugins] interval, fast_interval, and optional interval_fallbacks). CLI helpers hydrate these settings at start-up.

4. **Lifecycle hooks and validation.**
   - During calibration setup the registry resolves the desired plugin, instantiates it with the context, and asserts that the returned object implements the protocol required for the explainer mode. Missing capabilities raise a configuration error before calibration begins.
   - All calibrator methods must proxy to the underlying VennAbers and IntervalRegressor contracts. Runtime validation checks shapes, dtypes, and monotonicity of probability outputs against expectations from the in-tree classes. Fast-mode calibrators reuse the same validation but may skip expensive recomputation steps when metadata marks them fast_compatible.
   - Incremental calibration hooks (insert_calibration, CPS updates) remain the responsibility of IntervalRegressor; plugin authors extending those flows must invoke the superclass logic and only layer extra behaviour afterwards.

5. **Documentation and tooling.**
   - Developer docs gain a dedicated section that explains the layered protocol, the context object, and examples that subclass and compose IntervalRegressor / VennAbers.
   - CLI commands (ce.plugins list --intervals, ce.plugins explain-interval --plugin <id>) expose plugin availability and validation results to help detect misconfigured packages.

## Security & Guardrails

- Trust remains opt-in per ADR-006: only plugins explicitly marked as trusted are eligible for automatic selection.
- Metadata validation occurs during registration. Missing required fields keeps the plugin out of the registry, preventing accidental downgrades to unsafe implementations.
- Runtime guards assert that calibrators return probabilities and intervals identical in structure to the built-in classes. Failures bubble up with actionable diagnostics so the system never silently downgrades calibration quality.
- No network or sandbox changes are introduced. Plugin authors manage their own dependencies and distribution while inheriting the host Python process permissions.

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

- Implement the default plugin as a formal adapter over the existing IntervalRegressor and VennAbers classes to validate the contract.
- Extend integration tests to exercise registry selection, protocol validation, and failure paths when plugins omit required capabilities.
- Evaluate the need for caching hooks or memoisation in the context object once additional plugins exist.
