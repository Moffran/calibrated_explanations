# Configure telemetry

Calibrated Explanations emits structured telemetry for every prediction and
explanation. Use it to audit plugin routing, preprocessing, and uncertainty
sources.

## Inspect telemetry in Python

```python
factual = explainer.explain_factual(X_test[:1])
telemetry = getattr(factual, "telemetry", {})
print(telemetry["interval_source"], telemetry["plot_source"])
print(telemetry["uncertainty"]["calibrated_value"])
print(telemetry.get("preprocessor", {}))
```

`explainer.runtime_telemetry` mirrors the latest batch payload and is safe to log
from background workers.

## Configure via environment variables

Set telemetry-related plugins at process startup so the payload documents the
selected identifiers:

```bash
export CE_INTERVAL_PLUGIN=core.interval.fast
export CE_PLOT_STYLE=plot_spec.default.factual
python run_batch.py
```

The telemetry dictionary will report `interval_source="core.interval.fast"` and
list the configured plot style with fallbacks.

## CLI inspection

Use the bundled CLI to review registry state, trusted plugins, and default
routing:

```bash
python -m calibrated_explanations.plugins.cli list all
python -m calibrated_explanations.plugins.cli show core.interval.fast --kind intervals
```

The CLI echoes schema versions, trust flags, and dependency hints so you can
confirm what telemetry should report before invoking the explainer.

## Export telemetry snapshots

Telemetry is serialisable. Persist it alongside predictions to enable audit
trails:

```python
import json

payload = explainer.runtime_telemetry
with open("telemetry.json", "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2)
```

Store the payload with batch identifiers so you can debug plugin fallbacks or
preprocessor mismatches in production.
