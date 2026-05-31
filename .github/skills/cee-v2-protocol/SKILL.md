---
name: cee-v2-protocol
description: >
  Implement and test KServe V2 protocol adapters for CEE inference endpoints, including request parsing, response building, and metadata injection.
---

## Inputs

- **`content`** (text, required): The input relevant to this skill. See instructions for details.

## Output Format

Format: `markdown`

Required sections:
- output

# Cee V2 Protocol — Core Instructions

# CEE KServe V2 Protocol

## Use this skill when
- Implementing or modifying V2 request/response adapters in `common.adapters`
- Adding new V2 inference endpoints to `main.py`
- Debugging V2 protocol parsing errors
- Writing contract tests for the V2 inference API
- Integrating CEE with KServe or Triton serving platforms

## Inputs
- `packages/common/src/calibrated_explanations_enterprise/common/adapters/`
- `src/calibrated_explanations_enterprise/main.py` — V2 endpoint definitions
- `tests/contract/` — V2 contract tests
- `AGENTS.md` §"Key API Endpoints" and §"OSS Terminology Alignment > Output Attribute Names"

## V2 Protocol Specification

### Inference Request (POST `/v2/models/{model_name}/infer`)

```json
{
  "inputs": [
    {
      "name": "features",
      "shape": [n_samples, n_features],
      "datatype": "FP64",
      "data": [[...], [...]]
    }
  ],
  "parameters": {
    "semi_online": { "checkpoint": false },
    "return_explanations": true
  }
}
```

### Inference Response

```json
{
  "model_name": "model_name",
  "outputs": [
    { "name": "predictions", "datatype": "FP64", "shape": [n], "data": [...] },
    { "name": "probabilities", "datatype": "FP64", "shape": [n, n_classes], "data": [...] },
    { "name": "uncertainty_low", "datatype": "FP64", "shape": [n], "data": [...] },
    { "name": "uncertainty_high", "datatype": "FP64", "shape": [n], "data": [...] },
    { "name": "feature_importance", "datatype": "FP64", "shape": [n, n_features], "data": [...] }
  ],
  "parameters": {
    "ce_strategy": "semi_online",
    "drift_detected": false
  }
}
```

### Feedback Request (POST `/v2/models/{model_name}/feedback`)

```json
{
  "inputs": [
    { "name": "features", "shape": [n, f], "datatype": "FP64", "data": [...] },
    { "name": "target", "shape": [n], "datatype": "INT64", "data": [...] }
  ],
  "parameters": {
    "semi_online": { "checkpoint": true }
  }
}
```

## V2 Tensor Naming Rules (CRITICAL)

| Internal (Python) | V2 Protocol Name | Notes |
|---|---|---|
| `X` | `"features"` | Input features tensor |
| `y` | `"target"` | Labels/feedback tensor |
| `prediction` | `"predictions"` | Model output |
| `proba` | `"probabilities"` | Class probabilities |
| `low` | `"uncertainty_low"` | Lower bound |
| `high` | `"uncertainty_high"` | Upper bound |
| `feature_importance` | `"feature_importance"` | Feature contributions |

**Do NOT** use `"input"`, `"output"`, `"label"`, `"score"`, or any other names.

## Workflow

### Implementing a new V2 adapter

1. Read existing adapter code to understand the pattern
2. Create adapter in `packages/common/src/.../common/adapters/`
3. Implement request parser:
   ```python
   class V2RequestParser:
       @staticmethod
       def parse_features(inputs: list[dict]) -> np.ndarray:
           """Extract features tensor from V2 inputs list."""
           features_input = next(i for i in inputs if i["name"] == "features")
           return np.array(features_input["data"])
   ```
4. Implement response builder using canonical tensor names
5. Add contract tests under `tests/contract/`

### Writing V2 contract tests

```python
@pytest.mark.integration
def test_v2_infer_returns_predictions_and_uncertainty(client):
    request_body = {
        "inputs": [
            {"name": "features", "shape": [2, 4], "datatype": "FP64",
             "data": [[5.1, 3.5, 1.4, 0.2], [6.7, 3.0, 5.2, 2.3]]}
        ]
    }
    response = client.post("/v2/models/test_model/infer", json=request_body)

    assert response.status_code == 200
    outputs = {o["name"]: o for o in response.json()["outputs"]}
    assert "predictions" in outputs
    assert "uncertainty_low" in outputs
    assert "uncertainty_high" in outputs
    assert len(outputs["predictions"]["data"]) == 2
```

## Health Endpoints

| Path | Expected Response |
|---|---|
| `GET /health` | `{"status": "healthy"}` |
| `GET /v2/health/live` | KServe liveness — always 200 if server running |
| `GET /v2/health/ready` | KServe readiness — 200 only if CE library available |

The readiness endpoint must return 503 if `calibrated-explanations` is not importable.

## Verification
```bash
pytest tests/contract/ -q
pytest tests/integration/ -m integration -q
ruff check packages/common/ src/
```

## Output contract
For implementation tasks, return:
1. Adapter code in `packages/common/src/.../common/adapters/`
2. Endpoint handler additions/changes in `main.py`
3. Contract tests under `tests/contract/`
4. Confirmation all V2 tensor names are canonical

## Constraints
- Never use "input/inputs" for feature tensors (V2 `inputs` array is structural, not a tensor name)
- V2 responses must NEVER omit `model_name` field
- Entry points (`main.py`) must delegate to managers — no business logic in route handlers
- `common` package must not import from `adaptive` or `governance`
