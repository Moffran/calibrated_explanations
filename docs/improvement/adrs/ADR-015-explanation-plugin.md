> **Status note (2025-10-24):** Last edited 2025-10-24 · Archive after: Retain indefinitely as architectural record · Implementation window: Per ADR status (see Decision).

# ADR-015 — Explanation Plugin Architecture

Status: Accepted

Date: 2025-10-05

Authors: Core maintainers

Supersedes: N/A

Related: ADR-006-plugin-registry-trust-model, ADR-013-interval-calibrator-plugin-strategy, ADR-014-plot-plugin-strategy, ADR-026-explanation-plugin-semantics

## Context

`CalibratedExplainer` exposes three public entry points that all rely on the
same calibration, perturbation, and plotting machinery: factual explanations,
alternative (counterfactual-style) explanations, and the FAST importance-only
variant. In the current implementation each path hard-codes how perturbations
are generated, how results are materialised via
`CalibratedExplanations.finalize/finalize_fast`, and which concrete
`AbstractCalibratedExplanation` subclasses get instantiated. The factual and
alternative helpers merely flip discretiser flags before delegating to the
monolithic `explain` method, and the FAST pathway duplicates large sections of
that method while routing through a separate finaliser.

This tight coupling prevents explanation strategies from being authored as
plugins that integrate with the registry trust model (ADR-006) or that can
signal dependencies to the interval (ADR-013) and plot (ADR-014) plugin
resolvers. It also makes it difficult to refactor the 400+ line
`CalibratedExplainer.explain` pipeline or to introduce new explanation
collections without rewriting the container finalisation logic that sits deep
inside `CalibratedExplanations` today.【F:src/calibrated_explanations/core/calibrated_explainer.py†L622-L900】【F:src/calibrated_explanations/explanations/explanations.py†L318-L433】

We need a single orchestration surface that delegates explanation assembly to
plugins, keeps legacy behaviour byte-for-byte intact, and allows new
explanation strategies to participate in the same dependency coordination as
interval and plot plugins.

## Decision

### 1. Single orchestrator with mode-aware plugin resolution

`CalibratedExplainer.explain` (renamed to `_explain` in v0.10.0 per ADR-026) becomes the canonical orchestration entry. It
prepares the perturbation context, resolves an explanation plugin for the
requested mode, and delegates batch construction. Public helpers remain thin
wrappers:

- `explain_factual(...)` forwards to `_explain(..., mode="factual")` and seeds
the resolution chain with `core.explanation.factual`.
- `explore_alternatives(...)` calls `_explain(..., mode="alternative")` with a
fallback to `core.explanation.alternative`.
- `explain_fast(...)` invokes `_explain(..., mode="fast")` and prefers
`core.explanation.fast`.

**Note on Visibility:** While this ADR defines the plugin architecture and orchestration mechanics, **ADR-026** defines the visibility policy. `CalibratedExplainer.explain` is strictly an internal primitive (`_explain`) and is not part of the public API.

Resolution precedence mirrors ADR-006: explicit keyword arguments on the call
(e.g. `explanation_plugin`, `fast_explanation_plugin`) > environment variables
(`CE_EXPLANATION_PLUGIN`, `CE_EXPLANATION_PLUGIN_FAST`) > project configuration
(`[tool.calibrated_explanations.explanations]`) > package defaults. Each mode
maintains its own fallback chain, and plugin metadata may extend those chains
with mode-specific fallbacks. The resolver filters candidates by the explainer
`task` (classification/regression) and requires that selected plugins be
trusted unless the user explicitly opts into an untrusted identifier.

### 2. Explanation plugin protocol and shared data structures

Explanation plugins conform to a code-first protocol that exchanges frozen
context and batch payloads. The protocol lives in
`src/calibrated_explanations/plugins/explanations.py` and extends the common
`ExplainerPlugin` base from ADR-006.

```python
from dataclasses import dataclass
from typing import Mapping, MutableMapping, Optional, Protocol, Sequence, Tuple, Type

from calibrated_explanations.explanations.base import AbstractCalibratedExplanation
from calibrated_explanations.explanations.explanations import CalibratedExplanations
from calibrated_explanations.plugins.types import PluginMeta
from calibrated_explanations.plugins.predict import PredictBridge

@dataclass(frozen=True)
class ExplanationContext:
    task: str
    mode: str  # "factual" | "alternative" | "fast" | vendor extensions
    feature_names: Sequence[str]
    categorical_features: Sequence[int]
    categorical_labels: Mapping[int, Mapping[int, str]]
    discretizer: object
    helper_handles: Mapping[str, object]  # read-only perturbation buffers
    predict_bridge: PredictBridge  # delegated calibrated predictor
    interval_settings: Mapping[str, object]
    plot_settings: Mapping[str, object]

@dataclass(frozen=True)
class ExplanationRequest:
    threshold: Optional[object]
    low_high_percentiles: Optional[Tuple[float, float]]
    bins: Optional[object]
    features_to_ignore: Sequence[int]
    extras: Mapping[str, object]  # additional kwargs forwarded from callers

@dataclass
class ExplanationBatch:
    container_cls: Type[CalibratedExplanations]
    explanation_cls: Type[AbstractCalibratedExplanation]
    instances: Sequence[Mapping[str, object]]  # payload per explanation
    collection_metadata: MutableMapping[str, object]

class ExplanationPlugin(Protocol):
    plugin_meta: PluginMeta

    def supports_mode(self, mode: str, *, task: str) -> bool: ...

    def initialize(self, context: ExplanationContext) -> None: ...

    def explain_batch(
        self,
        x,
        request: ExplanationRequest,
    ) -> ExplanationBatch: ...
```

Key characteristics:

- `plugin_meta` reuses ADR-006 metadata and adds explanation-specific fields
(`modes`, `tasks`, `interval_dependency`, `plot_dependency`, optional
capability tags like `"explanation:factual"`).
- `initialize(...)` is invoked once per `CalibratedExplainer` instance to give
the plugin access to frozen context and the calibrated prediction bridge from
ADR-013. Plugins must treat the context as read-only.
- `explain_batch(...)` executes per request, returning a batch payload that
  identifies the container class and explanation class, supplies per-instance
  payloads, and can emit collection-level metadata (timings, provenance, raw
  perturbed predictions, plugin-defined artefacts).
- Plugins never call the learner directly; all predictions flow through the
  `PredictBridge`, ensuring interval guarantees are respected (ADR-013). The
  bridge enforces the invariant that every prediction triple obeys
  `low <= predict <= high`. Any plugin, bridge implementation, or downstream
  consumer that observes a `predict` value outside the inclusive
  `[low, high]` interval **must** raise a failure; silently accepting such a
  payload violates the fundamental semantics of calibrated explanations. This
  rule applies uniformly anywhere the runtime exposes paired `[low, high]`
  bounds (for example, feature weight intervals surfaced by explanation
  plugins).
- The batch contract materialises instance payloads today, but the signature
  is intentionally compatible with a future streaming or generator-based
  `instances` provider so that extremely large datasets can opt into lazy
  production without redesigning the protocol.

#### 2a. Factual Explanation Requirements

Explanation plugins generating factual batches must return an `ExplanationBatch` with
instances conforming to the structure:

```python
{
    "explanation": {
        "task": "classification" | "regression",
        "prediction": {
            "predict": float,
            "low": float,
            "high": float
        },
        "rules": [
            {
                "feature_id": int,
                "feature_name": str,
                "condition": str,  # e.g. "age <= 65"
                "weight": float,   # feature attribution
                "weight_low": float,  # calibrated interval lower
                "weight_high": float, # calibrated interval upper
                "support": float | None,
                "confidence": float | None,
                ...
            },
            ...
        ]
    }
}
```

The batch metadata must include the calibrated prediction interval. Each rule's condition must be tied to the observed feature value, and the feature weight must be accompanied by calibrated uncertainty intervals derived from the interval calibrator.

#### 2b. Alternative Explanation Requirements

Explanation plugins generating alternative batches must return an `ExplanationBatch` with
instances conforming to the structure:

```python
{
    "explanation": {
        "task": "classification" | "regression",
        "reference_prediction": {  # REQUIRED: original instance prediction
            "predict": float,
            "low": float,
            "high": float
        },
        "rules": [
            {
                "feature_id": int,
                "feature_name": str,
                "alternative_condition": str,  # e.g. "age = 40"
                "predicted_value": float,       # prediction for this scenario
                "prediction_low": float,        # calibrated interval lower
                "prediction_high": float,       # calibrated interval upper
                "support": float | None,
                "weight_delta": float | None,   # auxiliary metadata (not primary)
                ...
            },
            ...
        ]
    }
}
```

The batch metadata must include the reference prediction interval. Alternative rules
pair each scenario with the calibrated prediction for that scenario. Feature-weight
deltas may be included as auxiliary metadata but do NOT replace the scenario-level
prediction interval in the primary payload.

### 3. Collection construction and metadata handling

`CalibratedExplanations` becomes a lightweight collection façade with a single
`from_batch(...)` constructor used by the runtime and plugins alike. The class
stores the originating `CalibratedExplainer`, request metadata, the ordered
list of instantiated explanation objects, and any collection metadata supplied
by the plugin. Convenience methods such as `.append(...)`, `.extend(...)`, and
`.set_metadata(...)` remain available for in-tree plugins. The legacy
`finalize` and `finalize_fast` helpers wrap `from_batch(...)` so external code
that still calls them behaves identically but now routes through the plugin
pipeline.

### 4. Built-in plugins and compatibility guarantees

Three in-tree plugins preserve today’s behaviour while exercising the new
contract:

1. **`LegacyFactualExplanationPlugin` (`core.explanation.factual`)**
   - Reuses the existing perturbation logic (`_explain_predict_step`,
     `_assign_weight`, discretiser helpers) to produce factual
     `ExplanationBatch` payloads.
   - Emits `FactualExplanation` instances inside a `CalibratedExplanations`
     container and declares dependencies `interval_dependency="core.interval.legacy"`
     and `plot_dependency="legacy"` so the interval and plot resolvers receive
     the same hints as before.

2. **`LegacyAlternativeExplanationPlugin` (`core.explanation.alternative`)**
   - Shares the factual helper stack but instantiates `AlternativeExplanation`
     objects inside an `AlternativeExplanations` collection.
   - Exposes the same dependency metadata as the factual plugin.

3. **`FastExplanationPlugin` (`core.explanation.fast`)**
   - Ports the FAST orchestration, relying on the calibrated prediction bridge
     for per-feature perturbations without invoking the legacy rule grid.
   - Declares `interval_dependency="core.interval.fast"` and inherits the
     factual plot dependency so FAST plots continue to render through the
     default renderer.

All three register as trusted plugins. When no overrides are supplied, the
resolver selects these built-ins, producing byte-for-byte identical outputs to
the pre-plugin implementation.

### 4a. Execution Strategy Wrapper Plugins (v0.10.0+)

In addition to the three legacy plugins, a set of six execution strategy wrapper plugins enable users to select parallelism strategies for factual and alternative explanations via the plugin configuration system:

**Factual mode execution strategies:**

- `core.explanation.factual.sequential` - Single-threaded sequential processing
- `core.explanation.factual.feature_parallel` - Parallel processing across features
- `core.explanation.factual.instance_parallel` - Parallel processing across instances

**Alternative mode execution strategies:**

- `core.explanation.alternative.sequential` - Single-threaded sequential processing
- `core.explanation.alternative.feature_parallel` - Parallel processing across features
- `core.explanation.alternative.instance_parallel` - Parallel processing across instances

Each wrapper plugin delegates to the corresponding execution plugin from
`src/calibrated_explanations/core/explain/` and declares a fallback chain for graceful degradation:

```text
instance_parallel → feature_parallel → sequential → legacy
```

When a user selects an execution strategy plugin, the system attempts to use that strategy. If the underlying executor is unavailable or execution fails, it falls back to the next strategy in the chain. This ensures backward compatibility and safety while giving users fine-grained control over performance characteristics.

Example configuration:

```python
# Select feature-parallel strategy (falls back to sequential if executor unavailable)
explainer.explain_factual(x, explanation_plugin="core.explanation.factual.feature_parallel")
```

Or via environment:

```bash
export CE_EXPLANATION_PLUGIN_FACTUAL="core.explanation.factual.feature_parallel"
```

### 5. Registry metadata, configuration, and dependency coordination

- Plugin metadata extends ADR-006 with `modes` (subset of `{factual,
  alternative, fast}`), `tasks` (`{"classification", "regression", "both"}`),
  `interval_dependency`, `plot_dependency`, optional `fallbacks`, and an
  explicit `capabilities` list. Capabilities capture task-specific support and
  advanced behaviours (for example `"task:classification"`,
  `"task:regression"`, `"rules:conjunctions"`) so downstream tooling can reason
  about plugin features without inspecting code. Metadata validation is
  implemented alongside existing registry checks.
  `CalibratedExplainer` accepts keyword overrides per mode
  (`factual_plugin`, `alternative_plugin`, `fast_plugin`) and honours
  environment variables (`CE_EXPLANATION_PLUGIN_FACTUAL`,
  `CE_EXPLANATION_PLUGIN_ALTERNATIVE`, `CE_EXPLANATION_PLUGIN_FAST`). Prefer
  enabling FAST-mode by passing `fast=True` to `CalibratedExplainer`; use
  `fast_plugin` only when you need to target a specific FAST implementation.
  Project configuration mirrors these keys under
  `[tool.calibrated_explanations.explanations]` and may supply mode-specific
  fallback arrays. CLI helpers mirror the interval/plot commands for listing,
  validating, and setting explanation plugins.

## Implementation status (2025-10-10)

- Runtime orchestration, plugin validation, telemetry, and built-in adapters are
  live in `CalibratedExplainer` and `plugins.builtins`, matching the protocol
  described above.【F:src/calibrated_explanations/core/calibrated_explainer.py†L324-L606】【F:src/calibrated_explanations/plugins/builtins.py†L120-L318】
- Interval and plot dependency chains now honour metadata hints, environment
  overrides, and project configuration, keeping ADR-013/ADR-014 selections in
  sync with explanation plugins.【F:src/calibrated_explanations/core/calibrated_explainer.py†L652-L737】
- The `ce.plugins` console script surfaces registry state for explanations,
  intervals, and plots, enabling operators to audit trust and dependencies from
  packaging environments.【F:pyproject.toml†L33-L60】【F:src/calibrated_explanations/plugins/cli.py†L1-L145】

### 6. JSON and external payloads

The primary integration path is code-first batch construction. `CalibratedExplanations`
continues to expose `.to_json()` utilities that serialise known explanation
subclasses using the existing models and schema helpers. Third-party plugins
may return custom subclasses that implement the serialisation hooks, or they
may emit JSON artefacts inside `collection_metadata` for downstream consumers.
No additional guardrails are imposed for plugins that choose to emit entirely
custom containers; it is the plugin author’s responsibility to ensure
downstream tooling can consume those results.
No mandatory JSON contract is imposed by the plugin interface.

## Consequences

- **Backward compatibility:** Public APIs and explanation outputs remain
  unchanged for legacy modes because the built-in plugins reuse the existing
  helper stack and containers. `CalibratedExplanations.finalize` wrappers keep
  downstream integrations working during the migration window.
- **Extensibility:** New explanation strategies can provide their own
  `AbstractCalibratedExplanation` subclasses, containers, and metadata without
  modifying `CalibratedExplainer`. They can request specific interval or plot
  dependencies via metadata, giving them first-class participation in the
  broader plugin ecosystem defined by ADR-006/013/014.
- **Separation of concerns:** `CalibratedExplainer` focuses on perturbation
  orchestration and plugin resolution, while plugins manage batch assembly and
  container instantiation. The collection class centralises metadata handling
  and validation, reducing duplication across legacy and future plugins.
- **Testing impact:** Existing factual, alternative, and FAST unit tests run
  unchanged against the built-in plugins. New protocol tests cover plugin
  resolution, context immutability, dependency forwarding, and validation of
  `ExplanationBatch` payloads.

## Migration Plan

1. Introduce the plugin protocol, dataclasses, and registry validation helpers
   in `plugins/explanations.py`.
2. Refactor `CalibratedExplainer.explain` to resolve plugins and delegate batch
   execution, removing direct calls to `CalibratedExplanations.finalize` /
   `finalize_fast`.
3. Extract the legacy perturbation helpers into reusable functions that can be
   invoked by the built-in plugins when constructing `ExplanationBatch`
   payloads.
4. Implement `CalibratedExplanations.from_batch(...)`, update existing
   finalisers to delegate to it, and ensure collection metadata flows through.
5. Ship the three built-in plugins described above, registering them as trusted
   defaults.
6. Update configuration surfaces, CLI commands, and developer documentation to
   describe explanation plugin resolution and dependency coordination.
7. Add regression tests asserting that legacy and plugin-mediated outputs are
   identical across factual, alternative, and FAST modes, and add new tests for
   plugin metadata validation and fallback resolution.

## Resolved Questions

- **Streaming batches?** Not initially. The current design materialises
  instance payloads, but both the protocol and the batch dataclass deliberately
  leave room to swap `instances` for a streaming-friendly provider so the
  architecture is ready when we prioritise that enhancement.
- **Capability tags for specialised support?** Yes. Plugins must declare the
  capabilities they support (tasks, rule features, etc.) through the mandatory
  `capabilities` list so the runtime and downstream tooling can reason about
  compatibility without executing plugin code.
- **Protocol versioning?** We reuse the `schema_version` field already present
  in `plugin_meta`. Protocol changes increment the schema version, and the
  loader validates compatibility during plugin activation.
- **Guardrails for custom containers?** No additional guardrails are added.
  Plugins that emit bespoke containers are responsible for ensuring any
  downstream tooling they care about can consume the result.
