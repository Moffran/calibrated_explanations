# Configure telemetry

```{admonition} Optional telemetry extra
:class: tip

Telemetry is opt-in for v0.9.0. The core quickstarts and notebooks do not enable it—switch it on only when governance teams need provenance for calibrated runs.
```

Calibrated Explanations ships an optional structured telemetry payload that documents plugin routing, preprocessing, and uncertainty sources. Enable it when you need auditable provenance without changing the calibrated predictions themselves.

## Inspect telemetry in Python

Once telemetry is enabled, use the helpers below to review payloads in code.

```python
factual = explainer.explain_factual(x_test[:1])
telemetry = getattr(factual, "telemetry", {})
print(telemetry["interval_source"], telemetry["plot_source"])
print(telemetry["uncertainty"]["calibrated_value"])
print(telemetry.get("preprocessor", {}))
```

When you need the most recent payload outside the batch, reach into the wrapped
calibrator: `explainer.runtime_telemetry`. The wrapper keeps the last
telemetry dictionary there so background workers can log provenance without
storing the full batch object.

## Configure via environment variables

Opt-in teams can set telemetry-related plugins at process startup so the payload documents the
selected identifiers:

```bash
export CE_INTERVAL_PLUGIN=core.interval.fast
export CE_PLOT_STYLE=plot_spec.default.factual
python run_batch.py
```

The telemetry dictionary will report `interval_source="core.interval.fast"` and
list the configured plot style with fallbacks.

```{note}
Install the external FAST bundle before referencing `core.interval.fast`.
Run ``pip install "calibrated-explanations[external-plugins]"`` and call
``external_plugins.fast_explanations.register()`` to populate the registry.
```

## CLI inspection

Use the bundled CLI when governance teams need to review registry state, trusted plugins, and default
routing:

```bash
python -m calibrated_explanations.plugins.cli list all
python -m calibrated_explanations.plugins.cli show core.interval.fast --kind intervals
```

The CLI echoes schema versions, trust flags, and dependency hints so you can
confirm what telemetry should report before invoking the explainer.

## Export telemetry snapshots

When telemetry is enabled, persist the payload alongside predictions to enable audit
trails:

```python
import json

payload = explainer.runtime_telemetry
with open("telemetry.json", "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2)
```

Store the payload with batch identifiers so you can debug plugin fallbacks or
preprocessor mismatches in production.

## Feature-filtering telemetry example (ADR-027)

When FAST-based feature filtering is enabled, explanation collections can
include `filter_telemetry` details while runtime telemetry keeps interval and
resolver provenance.

```python
batch = explainer.explain_factual(x_test[:2])

runtime = explainer.runtime_telemetry
print(runtime.get("interval_dependencies", ()))
print(runtime.get("interval_source"))

# Collection-level feature-filter telemetry (if present)
print(getattr(batch, "filter_telemetry", {}))
```

Expected keys, aligned to emitted runtime payloads:

- `runtime_telemetry["interval_dependencies"]`: tuple of interval dependency hints
- `runtime_telemetry["interval_source"]`: resolved interval plugin source
- `batch.filter_telemetry["filter_enabled"]`: filter path was enabled
- `batch.filter_telemetry["filter_skipped"]`: fallback reason when filtering is skipped
- `batch.filter_telemetry["filter_error"]`: strict-observability error context when configuration fails

## STD-005 logger domains (minimum enforcement)

Telemetry and governance routing should use project logger domains rooted at
`calibrated_explanations.*` (ADR-028 / STD-005). The repository enforces this
minimum contract with:

```bash
python scripts/quality/check_logging_domains.py \
  --root src/calibrated_explanations \
  --report reports/quality/logging_domain_report.json
```

The check accepts `logging.getLogger(__name__)` and explicit
`calibrated_explanations.<domain>...` literals, and fails if non-project logger
literals are introduced in library code.
