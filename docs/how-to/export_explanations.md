# Export explanations

Calibrated explanations can be exported for offline analysis, dashboards, or QA
reviews.

## Serialize to JSON

```python
import json

batch = explainer.explain_factual(X_test[:10])
payload = []
for idx, exp in enumerate(batch):
    entry = exp.to_telemetry()
    entry["prediction"] = exp.predict
    entry["prediction_interval"] = exp.prediction_interval
    entry["instance_index"] = idx
    payload.append(entry)

with open("factual.json", "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2)
```

Each explanation exposes `to_telemetry()` which converts numpy types and rule
metadata into JSON-friendly structures.

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

## Attach telemetry

Store the runtime telemetry payload alongside exported rows to trace plugin
execution:

```python
telemetry = explainer.runtime_telemetry
with open("factual.telemetry.json", "w", encoding="utf-8") as fh:
    json.dump(telemetry, fh, indent=2)
```

Include batch identifiers and timestamps in filenames so you can reconcile the
payload with model monitoring pipelines.
