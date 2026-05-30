---
name: cee-semi-online
description: >
  Implement and configure semi-online adaptive calibration pipelines using SlidingWindowBuffer, SemiOnlineManager, and CalibratedAdaptiveExplainer.
---

## Inputs

- **`content`** (text, required): The input relevant to this skill. See instructions for details.

## Output Format

Format: `markdown`

Required sections:
- output

# Cee Semi Online — Core Instructions

# CEE Semi-Online Calibration

## Use this skill when
- Implementing or modifying the semi-online calibration pipeline
- Configuring `SemiOnlineConfig`, `DriftPolicy`, or `DriftMonitoringStrategy`
- Adding new window processing logic or extending `SemiOnlineManager`
- Debugging `recalibrate()` behaviour on the `CalibratedAdaptiveExplainer`
- Writing integration tests for the semi-online pipeline

## Inputs
- `packages/adaptive/src/calibrated_explanations_enterprise/adaptive/` — all adaptive package source
- `development/current-work/IMPLEMENTATION_GUIDE.md` §Stage 1 — implementation reference
- `development/strategic-pillars/02_online_calibration.md` — design rationale

## Architecture

```
CalibratedAdaptiveExplainer          (public API — only user-facing interface)
    └── _StrategyHandlers
           └── SemiOnlineManager     (orchestrates window + drift)
                  ├── SlidingWindowBuffer   (ring buffer, PyArrow backend)
                  └── CompositeDriftDetector (KS, MMD, Martingale)
```

## Key Types

| Class | File | Purpose |
|---|---|---|
| `CalibratedAdaptiveExplainer` | `calibrated_adaptive_explainer.py` | Public CE-First wrapper |
| `SemiOnlineManager` | `semi_online/manager.py` | Batch processing + drift evaluation |
| `SlidingWindowBuffer` | `semi_online/window.py` | Ring buffer with checkpoint support |
| `CompositeDriftDetector` | `semi_online/composite_drift.py` | Multi-method drift evaluation |
| `SemiOnlineConfig` | `semi_online/config.py` | Configuration (frozen Pydantic) |
| `DriftPolicy` | `semi_online/config.py` | Drift thresholds and remediation |
| `DriftMonitoringStrategy` | `semi_online/config.py` | `FEATURE`, `PREDICTION`, `CALIBRATION`, `FULL`, `AUTO`, etc. |

## Workflow

### Implementing a semi-online feature

1. **Read** the existing implementation files to understand current patterns
2. **Check** which `DriftMonitoringStrategy` values are affected
3. **Implement** following the CE-First wrapper pattern — never expose internals
4. **Add input adaptation**: convert `(X, y)` arrays to record format:
   ```python
   records = [{"features": x_i, "target": y_i} for x_i, y_i in zip(X, y)]
   ```
5. **Handle drift remediation** — either `FALLBACK_LAST_CHECKPOINT` or `ALERT_ONLY`
6. **Write tests** under `packages/adaptive/tests/semi_online/` and `tests/integration/`

### Configuring the pipeline

```python
from calibrated_explanations_enterprise.adaptive import CalibratedAdaptiveExplainer
from calibrated_explanations_enterprise.adaptive.semi_online.config import (
    SemiOnlineConfig, DriftPolicy, DriftMonitoringStrategy
)
from calibrated_explanations_enterprise.common.config import EnterpriseConfig

config = EnterpriseConfig(
    semi_online=SemiOnlineConfig(
        window={"size": 1000},
        drift_policy=DriftPolicy(
            strategy=DriftMonitoringStrategy.AUTO,
            ks_pvalue_floor=0.1,
            martingale_threshold=100.0,
            remediation="FALLBACK_LAST_CHECKPOINT",
        ),
    )
)

explainer = CalibratedAdaptiveExplainer(model, config=config)
explainer.fit(X_proper, y_proper)
explainer.calibrate(X_cal, y_cal)

# After new data arrives:
drift_result = explainer.recalibrate(X_new, y_new, checkpoint=True)
```

### `recalibrate()` input contract

- `X`: array-like (n_samples, n_features) **or** list of dicts with `"features"` key
- `y`: array-like (n_samples,) **or** `None` (if X is already record format)
- `checkpoint`: bool — save window state to disk before recalibrating
- Returns: `DriftResult` or `None`

## Performance Budgets
- Semi-online recalibration: <200ms P95 latency
- Tests: unit tests <100ms, integration tests <2s

## Verification
```bash
pytest packages/adaptive/tests/semi_online/ -q
pytest tests/integration/ -m integration -q
pytest tests/parity/ -q   # must pass after any recalibration change
ruff check packages/adaptive/
```

## Output contract
For implementation tasks, return:
1. Code changes with file paths
2. Test additions under `packages/adaptive/tests/semi_online/`
3. Parity test confirmation (no regressions)

## Constraints
- Never expose `SemiOnlineManager`, `SlidingWindowBuffer`, or `CompositeDriftDetector` as public API
- Only `CalibratedAdaptiveExplainer` is user-facing
- Config objects must be frozen Pydantic models
- Package isolation: `adaptive` must not import from `governance`
- If drift detection logic changes, always run `pytest tests/parity/ -q`