# Export explanations

Calibrated explanations can be exported for offline analysis, dashboards, or QA
reviews.

## Serialize with schema v1

Use the collection helper to generate a schema-aligned payload that bundles the
explanations and calibration metadata.

```python
import json

batch = explainer.explain_factual(X_test[:10])
payload = batch.to_json()

with open("factual.json", "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2, default=str)
```

The payload includes:

- `collection`: size, mode, feature names, thresholds, and runtime telemetry
  snapshot so integrations know how the explanations were generated.
- `explanations`: each instance serialised via schema v1 to preserve rule
  weights, probabilistic outputs, and provenance.

## Rehydrate exported explanations

`CalibratedExplanations.from_json()` returns domain-model objects that can be
inspected or re-exported without rebuilding the explainer.

```python
from calibrated_explanations.explanations import CalibratedExplanations

exported = CalibratedExplanations.from_json(payload)
for explanation in exported.explanations:
    print(explanation.task, explanation.prediction)
```

Pair the parsed explanations with the collection metadata to keep the
calibration context alongside downstream analytics or dashboards.

## Persist rule tables

```python
import pandas as pd

rows = []
for idx, exp in enumerate(batch):
    rows.append(
        {
            "instance": idx,
            "prediction": exp.predict,
            "low": exp.prediction_interval[0],
            "high": exp.prediction_interval[1],
        }
    )

df = pd.DataFrame(rows)
df.to_parquet("factual_rules.parquet")
```

Store the rule payload separately (from the JSON export above) if you need
feature-level detail inside analytics tools.

## Streaming large exports

Streaming delivery remains deferred for v0.9.0 while we monitor batch sizes.
The OSS scope inventory captures the decision and memory profiling notes so the
runtime team can revisit chunked exports post-release.

- Status: deferred (tracked in `improvement_docs/OSS_CE_scope_and_gaps.md`).
- Interim guidance: use `to_json()` in manageable batches and persist the
  resulting files or database rows according to your governance requirements.

### Optional extras

- **Telemetry attachment (optional).** Persist telemetry snapshots only when
  compliance workflows require them:

  ```python
  telemetry = explainer.runtime_telemetry
  with open("factual.telemetry.json", "w", encoding="utf-8") as fh:
      json.dump(telemetry, fh, indent=2)
  ```

Include batch identifiers and timestamps in filenames so you can reconcile the
payload with model monitoring pipelines.
