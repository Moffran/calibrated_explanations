# Optional telemetry scaffolding

```{admonition} Optional telemetry extra
:class: tip

Telemetry and performance snapshots are **opt-in** governance tooling.
The core quickstarts, notebooks, and practitioner flows keep telemetry
**disabled by default** so calibrated explanations stay lightweight until
compliance teams explicitly request provenance.
```

Telemetry records execution metadata for every calibrated explanation
batch. The payload mirrors the fields surfaced in
:doc:`../how-to/configure_telemetry` and provides
opt-in audit trails when regulators require reproducible provenance.

## Payload structure

Each explanation batch exposes a ``telemetry`` attribute and the wrapped
calibrator surfaces the most recent payload on
``explainer.runtime_telemetry``. Review the mapping before
persisting or routing the metadata to other systems:

| Key | Description |
| --- | ----------- |
| ``mode`` / ``task`` | Explanation mode (``factual``, ``alternative``, ``fast``) and learner task (``classification``/``regression``). |
| ``interval_source`` | Identifier of the interval calibrator used to generate bounds. |
| ``proba_source`` | Source of calibrated probabilities (often matches ``interval_source``). |
| ``plot_source`` | Plot style that rendered the figure (for example ``plot_spec.default.factual``). |
| ``plot_fallbacks`` | Ordered tuple of fallbacks when the primary plot style failed. |
| ``uncertainty`` | Calibrated prediction details (representation, bounds, percentiles, optional threshold metadata). |
| ``rules`` | Rule payload for each explanation, including per-feature uncertainty. |
| ``preprocessor`` | Snapshot of preprocessing metadata (identifier, pipeline steps, auto-encoding flags). |
| ``interval_dependencies`` | Interval plugin hints passed through explanation metadata. |

Additional fields may appear as the schema evolves; always inspect
``telemetry.keys()`` before assuming availability.

## Preprocessor snapshots

When the wrapper is built via configuration, telemetry includes an
ADR-009 preprocessor snapshot:

```python
payload = explainer.runtime_telemetry
pre = payload.get("preprocessor", {})
print(pre.get("identifier"))  # e.g. sklearn.compose:ColumnTransformer
print(pre.get("auto_encode"))
```

Snapshots are limited to safe metadata—raw training data is never
embedded.

## Interval provenance

Runtime records the active interval source for both default and FAST
modes:

```python
explainer.explain_fast(X_test[:5])
fast_meta = explainer.runtime_telemetry
print(fast_meta.get("interval_source"))
```

The payload also captures the most recent probabilistic threshold
metadata so thresholded regression runs remain auditable.

## Plot routing

PlotSpec adapters annotate telemetry with the selected builder and
renderer. If a fallback occurs (for example, the legacy Matplotlib
renderer), the fallback list records the full chain so dashboards can
differentiate first-choice and recovery paths.

## Persisting telemetry

Telemetry payloads are JSON serialisable and can be stored alongside
exported explanations to provide an audit trail for plugin choices,
preprocessing policy, and interval provenance:

```python
import json

with open("batch.telemetry.json", "w", encoding="utf-8") as fh:
    json.dump(explainer.runtime_telemetry, fh, indent=2)
```

## Instrumentation examples

Telemetry payloads are plain dictionaries, so existing observability
infrastructure can ingest them without custom adapters:

```python
import logging
from prometheus_client import Gauge

logger = logging.getLogger("calibrated_explanations.telemetry")
interval_source = Gauge("ce_interval_source", "Active interval calibrator", ["identifier"])

batch = explainer.explain_factual(X_test[:10])
payload = getattr(batch, "telemetry", {})

logger.info("explain_factual", extra={"telemetry": payload})
interval_source.labels(payload.get("interval_source", "unknown")).set(1)
```

For configuration-first deployments, declare defaults in
``pyproject.toml`` and mirror them with environment variables for
overrides so the payload documents the effective identifiers:

```toml
[tool.calibrated_explanations.intervals]
default = "core.interval.fast"

[tool.calibrated_explanations.explanations]
factual = "core.explanation.factual"
```

```bash
export CE_INTERVAL_PLUGIN=core.interval.fast
export CE_EXPLANATION_PLUGIN=core.explanation.factual
```

Use ``python -m calibrated_explanations.plugins.cli show <identifier>``
to audit the resolved configuration described in the telemetry payload.
Governance and contributor workflows centralise these checks under
:doc:`index` so operators can surface optional instrumentation only when
required.
