# Telemetry semantics

Telemetry records the execution path for every prediction and explanation. It is
enabled by default and aligns with ADR-013 (interval), ADR-014 (plotting), and
ADR-015 (plugin orchestrator).

## Payload structure

Each explanation batch receives a `telemetry` attribute; the wrapper mirrors the
same payload on `CalibratedExplainer.runtime_telemetry`.

| Key | Description |
| --- | ----------- |
| `mode` / `task` | Explanation mode (`factual`, `alternative`, `fast`) and learner task (`classification`/`regression`). |
| `interval_source` | Identifier of the interval calibrator used to generate bounds. |
| `proba_source` | Source of calibrated probabilities (often matches `interval_source`). |
| `plot_source` | Plot style that rendered the figure (e.g., `plot_spec.default.factual`). |
| `plot_fallbacks` | Ordered tuple of fallbacks used if the primary plot style failed. |
| `uncertainty` | Calibrated prediction details (representation, bounds, percentiles, optional threshold metadata). |
| `rules` | Rule payload for each explanation, including per-feature uncertainty. |
| `preprocessor` | Snapshot of preprocessing metadata (identifier, pipeline steps, auto-encoding flags). |
| `interval_dependencies` | Interval plugin hints passed through explanation metadata. |

Additional fields may appear as the schema evolves; always inspect
`telemetry.keys()` before assuming availability.

## Preprocessor snapshots

When the wrapper is built via configuration, telemetry includes an ADR-009
preprocessor snapshot:

```python
payload = explainer.runtime_telemetry
pre = payload.get("preprocessor", {})
print(pre.get("identifier"))  # e.g. sklearn.compose:ColumnTransformer
print(pre.get("auto_encode"))
```

Snapshots are limited to safe metadata—raw training data is never embedded.

## Interval provenance

Runtime records the active interval source for both default and FAST modes:

```python
explainer.explain_fast(X_test[:5])
fast_meta = explainer.runtime_telemetry
print(fast_meta.get("interval_source"))
```

The payload also captures the most recent probabilistic threshold metadata so
thresholded regression runs remain auditable.

## Plot routing

PlotSpec adapters annotate telemetry with the selected builder and renderer. If
a fallback occurs (e.g., legacy Matplotlib renderer), the fallback list records
the full chain so dashboards can differentiate first-choice and recovery paths.

## Persisting telemetry

Telemetry payloads are JSON serialisable:

```python
import json

with open("batch.telemetry.json", "w", encoding="utf-8") as fh:
    json.dump(explainer.runtime_telemetry, fh, indent=2)
```

Persist them alongside exported explanations to provide an audit trail for
plugin choices, preprocessing policy, and interval provenance.

## Instrumentation examples

Telemetry payloads are plain dictionaries, so existing observability tooling can
pick them up without custom adapters:

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

For configuration-first deployments, declare defaults in `pyproject.toml` and
mirror them with environment variables for overrides:

```toml
[tool.calibrated_explanations.intervals]
default = "core.interval.fast"

[tool.calibrated_explanations.explanations]
factual = "core.explanation.factual"
```

```bash
export CE_INTERVAL_PLUGIN=core.interval.fast
export CE_EXPLANATION_PLUGIN_FACTUAL=core.explanation.factual
```

Use ``python -m calibrated_explanations.plugins.cli show <identifier>`` to audit
the resolved configuration described in the telemetry payload. The navigation
for these workflows now lives under :doc:`../how-to/index` and
:doc:`../extending/index` so operators, integrators, and plugin authors can find
the appropriate task-focused guidance quickly.
