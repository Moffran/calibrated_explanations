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

For large explanation collections, use the experimental streaming export to avoid loading all explanations into memory at once.

```python
# Stream explanations as JSON Lines (default)
for chunk in explanations.to_json_stream(chunk_size=256, format="jsonl"):
    # Each chunk is a JSON string; write to file or process incrementally
    print(chunk)  # or fh.write(chunk + "\n")

# Or use chunked JSON arrays
for chunk in explanations.to_json_stream(chunk_size=256, format="chunked"):
    # Each chunk is a JSON array string like "[{...},{...},...]"
    print(chunk)
```

- **Status:** Experimental in v0.10.1+.
- **Memory profile:** Tested for < 200 MB peak usage with 10k explanations at `chunk_size=256`.
- **Telemetry:** Export metrics (rows, elapsed time, peak memory) are captured in collection metadata and explainer telemetry.

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
