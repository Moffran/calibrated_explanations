# ADR-015 — Explanation Plugin Interface (Code-Grounded)

Status: Proposed (requires coordinated rollout)

Date: 2025-09-18 (revised 2025-10-02)

Author: Internal

## Context

We need a stable plugin surface for third‑party “explanation providers” that plugs into calibrated_explanations without breaking the existing internal classes:

- Per‑instance explanation classes live in `src/calibrated_explanations/explanations/explanation.py` (abstract `CalibratedExplanation` and concrete `FactualExplanation` / `AlternativeExplanation` / `FastExplanation`).
- The collection/iterator is `CalibratedExplanations` in `src/calibrated_explanations/explanations/explanations.py` which finalizes instances and exposes plotting/interop.
- A minimal plugin registry scaffold exists at `src/calibrated_explanations/plugins/base.py` via `ExplainerPlugin(Protocol)` and `validate_plugin_meta`.
- A v1 JSON schema and adapters already exist at:
  - Schema: `src/calibrated_explanations/schemas/explanation_schema_v1.json`
  - Domain models + (de)serialization: `src/calibrated_explanations/explanations/models.py`, `src/calibrated_explanations/serialization.py`

This ADR aligns the “Explanation Plugin” contract to those real interfaces. The existing in-tree explanations remain the default behaviour: the plugin system must treat them as the reference implementation and expose them as the first registered provider so that current users see no change unless they explicitly opt into a third-party plugin.

## Decision

Adopt a dual contract for explanation plugins that reflects how the in‑tree classes construct explanations today and adds an optional JSON‑first path:

- Batch initializer path (preferred for core integration): Plugins return batch‑level arrays/maps that feed directly into `CalibratedExplanations.finalize(...)` or `finalize_fast(...)`.
- JSON‑first path: Plugins return per‑instance Explanation payloads conforming to schema v1, or domain model objects (`Explanation` / `FeatureRule`) that serialize via `serialization.to_json`.

We reuse the existing plugin base (`ExplainerPlugin`) and extend it with an optional initialization hook and explicit mode capabilities. Plugins must declare which modes they support: `explanation:factual`, `explanation:alternative`, `explanation:fast`. We ship two built-in providers: `core.explanation.legacy` covers the rule-based (factual/alternative) modes and remains the default fallback, while `core.explanation.fast` implements the FAST importance-only path and is treated as the second wave plugin. Both providers delegate to the existing `CalibratedExplanations.finalize` or `finalize_fast` flows, but they register separately so FAST is not grouped under the legacy identifier. Because FAST explanations reuse the factual probabilistic plots, whichever plot plugin is active (including the legacy default) can render their outputs without a dedicated FAST renderer.

- **Registry metadata requirements.**
  - Extend `calibrated_explanations.plugins.registry` with `register_explanation_plugin`, `find_explanation_plugin`, and `find_explanation_plugin_trusted`, mirroring ADR-006 semantics.
  - Metadata must include: `modes` (set drawn from `explanation:factual`, `explanation:alternative`, `explanation:fast`), `dependencies`, optional `interval_dependency` (identifier of the calibrator plugin requested for this explanation mode), optional `plot_dependency` (style identifier for plots), shared ADR-006 fields (`trust`, `version`, `description`), and capability tags (`explanation:factual-conditional`, etc.).
  - `core.explanation.legacy` registers with `interval_dependency="core.interval.legacy"` and marks itself trusted. `core.explanation.fast` registers with `interval_dependency="core.interval.fast"` and capability `explanation:factual-importance-only`.

- **Configuration and resolution order.**
  - Resolution order for explanation plugins is: explicit kwargs on `CalibratedExplainer` (`explanation_plugin`, `fast_explanation_plugin`) > environment variables (`CE_EXPLANATION_PLUGIN`, `CE_EXPLANATION_PLUGIN_FAST`) > project configuration (`pyproject.toml` under `[tool.calibrated_explanations.explanations]`) > package default.
  - Fallbacks are expressed via `CE_EXPLANATION_PLUGIN_FALLBACKS` (comma-separated) and the corresponding `explanation_fallbacks` table in project configuration. The package seeds that chain with `core.explanation.legacy` so behaviour matches the current implementation if no overrides are supplied.
  - CLI helpers (`ce.plugins list --explanations`, `ce.plugins validate-explanation --plugin <id>`, `ce.plugins set-default --explanation <id>`) mirror the plot and interval commands.

- **Inter-plugin coordination.**
  - During resolution the explainer inspects the selected explanation plugin metadata. If `interval_dependency` is present, that identifier is prepended to the interval plugin fallback chain (ADR-013). If `plot_dependency` is present, the plot resolver performs the same adjustment (ADR-014).
  - FAST explanations therefore select `core.interval.fast` unless the caller passes an explicit interval plugin. Legacy explanations continue to use `core.interval.legacy` and request the `legacy` plot style by default.
  - Plugins that require bespoke plotting styles must ship a compatible plot builder/renderer and advertise the corresponding `plot_dependency` identifier; otherwise, they inherit whichever plot plugin the caller configured.

## JSON Schema (v1)

Canonical path: `src/calibrated_explanations/schemas/explanation_schema_v1.json`.

Key properties (per instance):
- `task: string` — e.g., "classification" | "regression"
- `index: integer` — instance index in X
- `prediction: object` — typically contains `predict`, `low`, `high`
- `rules: array[object]` — each rule requires:
  - `feature: integer | array` (array for conjunctive rules)
  - `rule: string` (human‑readable condition)
  - `weight: object` (e.g., `predict`/`low`/`high` for contribution)
  - `prediction: object` (rule‑level predicted outcome; same keys as above)
  - optional: `instance_prediction`, `feature_value`, `is_conjunctive`, `value_str`, `bin_index`

Versioning: payloads may include `schema_version: "1.0.0"` (recommended). Validation via `serialization.validate_payload` is optional if `jsonschema` is installed.

## Python Protocol

Use the existing `ExplainerPlugin` Protocol (`src/calibrated_explanations/plugins/base.py`) and add a code‑grounded contract for initialization and mode‑specific explanation outputs. Plugins MUST NOT call the learner/model directly; all predictions and uncertainty come via a provided predict bridge that proxies to the interval calibrators (VennAbers/IntervalRegressor) exactly like `_predict` in core. The legacy plugin wraps that bridge with the same caching and batching semantics used by the legacy code so behaviour remains identical, while the FAST plugin reuses the bridge with the pared-down importance-only orchestration defined by the FAST framework.

Type aliases (internal, for clarity):

```python
from typing import Any, Callable, Dict, List, Mapping, Protocol, TypedDict, runtime_checkable, Union
from calibrated_explanations.explanations.models import Explanation

# JSON-first outputs
JSONPayload = Dict[str, Any]  # must conform to v1 schema
ExplanationOutput = Union[Explanation, JSONPayload, List[Explanation], List[JSONPayload]]

# Batch initializer path (shapes mirror finalize/finalize_fast)
class BatchFactualLike(TypedDict):  # used by factual and alternative
    binned: Mapping[str, Any]  # e.g., {"predict": arr[f,i,b], "low": ..., "high": ..., "rule_values": ...}
    feature_weights: Mapping[str, Any]  # {"predict": arr[f,i], "low": arr[f,i], "high": arr[f,i]}
    feature_predict: Mapping[str, Any]  # {"predict": arr[f,i], "low": arr[f,i], "high": arr[f,i]}
    prediction: Mapping[str, Any]  # {"predict": arr[i], "low": arr[i], "high": arr[i], optional extras}
    instance_time: Any | None
    total_time: Any | None

class BatchFast(TypedDict):  # used by fast explanations
    feature_weights: Mapping[str, Any]
    feature_predict: Mapping[str, Any]
    prediction: Mapping[str, Any]
    instance_time: Any | None
    total_time: Any | None

BatchOutput = Union[BatchFactualLike, BatchFast]

# Predict bridge signature (bound to the current explainer)
# Mirrors core._predict(X, threshold, low_high_percentiles, classes, bins, feature)
PredictBridge = Callable[
    [
        Any,  # X
        Any | None,  # threshold
        tuple[float, float] | None,  # low_high_percentiles
        Any | None,  # classes (classification targets, when applicable)
        Any | None,  # bins (Mondrian categories)
        int | None,  # feature (for fast/per-feature isolation)
    ],
    tuple[Any, Any, Any, Any],  # (predict, low, high, classes)
]

@runtime_checkable
class ExplanationProvider(Protocol):
    plugin_meta: Dict[str, Any]
    # Should include e.g. {
    #   "schema_version": 1,
    #   "capabilities": ["explain", "explanation:factual", ...],
    #   "name": str
    # }

    def supports(self, mode: str) -> bool: ...

    def initialize(
        self,
        context: Dict[str, Any],
        predict_bridge: PredictBridge,
        *,
        legacy_handles: Dict[str, Any] | None = None,
    ) -> None: ...  # optional but recommended

    # mode: "factual" | "alternative" | "fast"; no learner/model is passed
    # prefer returning BatchOutput; JSON-first also allowed
    def explain(
        self,
        X: Any,
        *,
        mode: str,
        legacy_handles: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Union[BatchOutput, ExplanationOutput]: ...
```

Required metadata and validation remain via `validate_plugin_meta`.

Expected kwargs (mirrors core explainer and `_predict`):
- `low_high_percentiles: tuple[float, float] | None` (regression intervals; supports one‑sided ±∞)
- `threshold: float | tuple[float, float] | None` (probabilistic regression)
- `bins`, `features_to_ignore`, `rnk_metric`, `rnk_weight`, `filter_top`, `allow_conjunctions` (where applicable)

Prohibitions
- Plugins must not call the learner/model or interval calibrators directly. All predictions and uncertainty must go through `predict_bridge`.

## Initialization and Lifecycle

Plugins may need contextual information to construct explanations efficiently and consistently with the core classes. We define an optional initialization stage and a mode‑aware execution stage, and we provide a predict bridge that mirrors the core `_predict` method and delegates to the interval calibrators. We also pass `legacy_handles` so the built-in plugin can continue using the optimized numpy buffers created by `_build_explanations` without reallocation. Third-party plugins can safely ignore that parameter:

1) Initialization (optional but recommended)

- `initialize(context: Mapping[str, Any], predict_bridge: PredictBridge, *, legacy_handles: Mapping[str, Any] | None = None) -> None`
- The core passes a read‑only context derived from `FrozenCalibratedExplainer` and current run‑time settings, and a `predict_bridge` callable bound to the current explainer. Suggested fields in `context`:
  - `task`, `mode`, `class_labels`, `feature_names`, `categorical_features`, `categorical_labels`, `feature_values`
  - `discretizer`, `rule_boundaries`, `bins` (Mondrian bins per instance if available)
  - `X_cal`, `y_cal`, `sample_percentiles`, `assign_threshold`, `low_high_percentiles`, `y_threshold`
  - `learner`, `difficulty_estimator`
  - `features_to_ignore`, `allow_conjunctions`

2) Explain (per batch)

- `explain(X, *, mode=..., legacy_handles: Mapping[str, Any] | None = None) -> BatchOutput | ExplanationOutput`
- Mode determines the minimal required outputs (see next section). The plugin uses `predict_bridge` internally to obtain calibrated predictions/intervals. The core bridges outputs to either `finalize(...)` / `finalize_fast(...)` (batch path) or builds `CalibratedExplanations` from JSON/domain payloads (JSON‑first path).

Notes
- Alternative explanations require richer inputs (binned arrays and rule boundaries) than fast explanations. See “Output Contracts by Mode”.
- For classification, plugins may attach full per‑batch probability matrices under the magic key `__full_probabilities__` in `prediction` for golden baselines; unknown keys are ignored by schema v1.
- The core passes `legacy_handles` that include the existing `RuleBuilder`, precomputed discretizer outputs, and numpy work arrays. External plugins MUST NOT mutate these handles; they are provided solely for the built-in adaptor. They may be `None` for third-party providers.

## Output Contracts by Mode

These contracts are shaped to match `CalibratedExplanations.finalize(...)` and `finalize_fast(...)`.

- Factual (mode = "factual")
  - Must return `BatchFactualLike`:
    - `binned`: mapping with keys at least `predict`, `low`, `high`, and `rule_values`.
      - Shapes: `predict[f][i][b]`, etc., where `f`=feature, `i`=instance, `b`=bin index (inner shape may be ragged for categorical).
    - `feature_weights`: per‑feature contributions with keys `predict`, `low`, `high` shaped `[f][i]`.
    - `feature_predict`: per‑feature predictions with keys `predict`, `low`, `high` shaped `[f][i]`.
    - `prediction`: per‑instance mapping with keys `predict`, `low`, `high` shaped `[i]`; may include `classes` (class id) and `__full_probabilities__` for classification.

- Alternative (mode = "alternative")
  - Same as Factual, but `binned` must include per‑feature, per‑bin arrays for the alternative values used by `AlternativeExplanation`:
    - `binned["predict"][f][i][value_bin]`, and similarly for `low`/`high`.
    - `rule_values` must carry the original/current bin value used to render rule text.
  - The `discretizer`, `rule_boundaries`, and categorical metadata in `context` are required to construct the alternatives consistently.

- Fast (mode = "fast")
  - Must return `BatchFast`:
    - No `binned` required.
    - `feature_weights`, `feature_predict`, and `prediction` with the same key conventions and shapes as above.
  - Core will route to `CalibratedExplanations.finalize_fast(...)`.

Validation and Invariants
- Keys `predict`, `low`, `high` must be present where intervals are applicable to the task.
- One‑sided intervals are permitted for regression; the core will treat `±inf` percentiles as one‑sided and disable uncertainty bands in plots.
- Indexing: arrays must be aligned such that the second dimension corresponds to instance index `i` used by `CalibratedExplanations`.
- Feature‑level UQ: For per‑feature weights, plugins should compute
  - `weight_predict[f] = assign_weight(instance_predict_predict[f], prediction_predict)` and
  - bounds using low/high deltas with `weight_low[f] = min(assign_weight(instance_predict_low[f]), assign_weight(instance_predict_high[f]))` and `weight_high[f] = max(...)`,
  matching core `_assign_weight` semantics to ensure `weight_low <= weight_high`.

Conditional‑rule Consistency
- For conditional rule engines (factual/alternative), plugins must ensure the same rule condition is used for both the lower and upper surfaces when generating rule text and per‑feature values (e.g., a single discretizer condition drives both low and high). This guarantees consistent interpretation across bounds.

## Code‑Grounded Interface Notes

- Per‑instance explanation objects provide `prediction` with keys `predict`, `low`, `high` and rule‑level arrays with parallel keys. See `FactualExplanation._get_rules()` and `AlternativeExplanation._get_rules()` in `src/calibrated_explanations/explanations/explanation.py`.
- Collections are materialized via `CalibratedExplanations.finalize(...)`, which constructs per‑instance explanations and can be iterated or plotted.
- Classification can also expose per‑instance probability vectors; internal enrichment may attach a `__full_probabilities__` key on the instance `prediction` mapping for golden baselines. Plugins should ignore unknown keys per schema’s `additionalProperties: true`.

## Predict Bridge and Calibrators

- The predict bridge mirrors `CalibratedExplainer._predict` and is the only route for plugins to obtain calibrated outputs. It dispatches to interval calibrators:
  - Classification: `interval_learner.predict_proba(X, output_interval=True, bins=...)` (or `interval_learner[feature]` for fast/per-feature)
  - Regression: `interval_learner.predict_uncertainty(X, low_high_percentiles, bins=...)` or `interval_learner.predict_probability(X, threshold, bins=...)`
- Fast mode: per‑feature isolation is handled by `feature` argument to the bridge (internally selects `interval_learner[feature]`).
- Plugins must treat the bridge as a pure function of inputs and must not attempt to access the learner or calibrators directly.

## Finalization Bridge

- The explanations collection remains the authority for constructing `FactualExplanation`, `AlternativeExplanation`, and `FastExplanation` instances.
- Batch outputs from plugins are passed to:
  - `CalibratedExplanations.finalize(...)` for conditional (factual/alternative) rules.
  - `CalibratedExplanations.finalize_fast(...)` for importance‑only (fast) rules.
- Recommendation: introduce user‑facing aliases to clarify intent without breaking compatibility:
  - `finalize_conditional_rules(...)` (alias of `finalize`)
  - `finalize_importance_rules(...)` (alias of `finalize_fast`)

## Capability Metadata

Extend plugin `plugin_meta["capabilities"]` with explicit tags so the registry/core can pick compatible providers:
- `explanation:factual` | `explanation:alternative` | `explanation:fast`
- `schema:explanation/v1` if using the JSON‑first path
- Optional diagnostics tags, e.g., `supports:multiclass`, `supports:one_sided`
- A hard requirement tag signaling bridge use: `predict:bridge`

Example

```json
{
  "schema_version": 1,
  "name": "acme.adapter",
  "capabilities": [
    "explain",
    "explanation:fast",
    "predict:bridge",
    "supports:multiclass"
  ]
}
```

## Conformance Checklist (Plugin Authors)

- Metadata
  - [ ] Provide `plugin_meta` with `schema_version: int`, `capabilities: list[str]` (must include `"explain"` and should include `"schema:explanation/v1"`), and `name`.
  - [ ] Pass `plugin_meta` through `validate_plugin_meta` during registration.

- Predictions (UQ)
  - [ ] For every instance, include calibrated UQ in `prediction` — keys `predict`, `low`, `high` as applicable to task.
  - [ ] For rule items, include rule‑level `prediction` with the same keys.

- Importance (UQ)
  - [ ] For every rule item, include `weight` with keys `predict`, `low`, `high` (or a task‑appropriate equivalent). For fast/no‑rule modes, emit per‑feature weights with intervals.

 - Batch initializer path
   - [ ] Factual/Alternative: provide `binned`, `feature_weights`, `feature_predict`, `prediction` with shapes matching `CalibratedExplanations.finalize(...)`.
   - [ ] Fast: provide `feature_weights`, `feature_predict`, `prediction` (no `binned`) for `CalibratedExplanations.finalize_fast(...)`.
   - [ ] Optional: `instance_time` per instance and `total_time` for diagnostics.

 - JSON‑first path
   - [ ] Emit schema v1 payloads or domain model objects; `serialization.validate_payload` can check shape if `jsonschema` is installed.
   - [ ] Ensure `task`, `index`, `prediction`, and `rules[...]` populate required fields; unknown extra fields are allowed.

- Semantics & Controls
  - [ ] Honor `low_high_percentiles` for regression (accept one‑sided intervals).
  - [ ] Treat `threshold` as probabilistic regression only (scalar or 2‑tuple). Reject misuse for plain regression.
  - [ ] Support multiclass classification (rule `classes` may be present in internal pipelines); plugins should provide consistent rule semantics.
  - [ ] If conjunctions are produced, set `is_conjunctive` and use `feature` as a sequence.

- Output shape
  - [ ] Return either a single Explanation payload, a list of payloads, or domain models convertible by `serialization.to_json`.
  - [ ] When emitting JSON, optionally set `schema_version: "1.0.0"` and consider validating with `serialization.validate_payload`.

## Consequences

- Plugins interoperate with existing plotting and downstream adapters by emitting schema v1 per‑instance payloads or by returning batch outputs for finalization.
- The core remains owner of calibration. Third‑party plugins that do their own calibration MUST still return UQ fields in the v1 shape.
- Future schema versions can be introduced alongside v1 with capability tags like `schema:explanation/v2`.

## Error Handling and Validation

- Plugins should raise clear `ValueError`s when required context is missing (e.g., `rule_boundaries` for alternative mode) or shapes are inconsistent.
- Plugins must not access the underlying learner or calibrators directly; violations will be rejected in code review and may be guarded at runtime.
- The core can perform early validation of shapes/keys before invoking `finalize(...)`/`finalize_fast(...)` to surface configuration errors upfront.
- JSON‑first payloads may be validated against schema v1 when `jsonschema` is present; otherwise, rely on downstream adapters.

Special Requirements for Conditional‑Rule Plugins
- When emitting rule text, use the same discretizer boundaries/conditions for both lower and upper bound interpretations of a feature; do not create divergent conditions per bound.
- At minimum, provide `binned["rule_values"]` so `FactualExplanation` can render feature value strings consistently. Other `binned` keys may be omitted for factual mode if not used by the plugin; alternative mode requires full per‑bin arrays.

## Backwards Compatibility

- Dual output paths allow gradual adoption without breaking existing code. Batch outputs drop into existing finalize methods; JSON‑first integrates via adapters.
- Capability tags prevent accidental selection of incompatible plugins.

## Rationale

- Fast explanations do not require binned per‑feature arrays; factual and alternative do. Making initialization explicit avoids under‑specification and mis‑shapen outputs.
- An initialization hook avoids recomputing static artefacts (e.g., rule boundaries) on every call and aligns with how `FrozenCalibratedExplainer` exposes state today.

## Terminology and Renaming Plan

- Current terms in code and docs:
  - `factual` (conditional rules)
  - `alternative` (conditional rules)
  - `fast` (importance‑only)

- Proposed canonical terminology going forward:
  - `factual-conditional` (alias of `factual`)
  - `alternative-conditional` (alias of `alternative`)
  - `factual-importance-only` (alias of `fast`)
  - `alternative-importance-only` (future; not yet implemented)

- Capability tags and backward compatibility:
  - Recognize both the legacy tags (`explanation:factual`, `explanation:alternative`, `explanation:fast`) and the new tags (`explanation:factual-conditional`, `explanation:alternative-conditional`, `explanation:factual-importance-only`).
  - The new `explanation:alternative-importance-only` may be introduced later; until then, plugins should not advertise it.

- Finalization aliases (non‑breaking, recommended):
  - `finalize_conditional_rules(...)` → alias of `finalize(...)` (for conditional engines)
  - `finalize_importance_rules(...)` → alias of `finalize_fast(...)` (for importance‑only engines)

- Documentation and migration:
  - Prefer the new, explicit names in future docs and examples to improve clarity; keep legacy names as aliases to preserve API stability.
