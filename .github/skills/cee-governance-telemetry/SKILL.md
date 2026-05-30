---
name: cee-governance-telemetry
description: >
  Implement CEE governance telemetry including audit trails, evidence packs, Prometheus metrics, and structured enterprise logging per strategic pillars 4 and 6.
---

## Inputs

- **`content`** (text, required): The input relevant to this skill. See instructions for details.

## Output Format

Format: `markdown`

Required sections:
- output

# Cee Governance Telemetry — Core Instructions

# CEE Governance Telemetry

## Use this skill when
- Implementing or extending audit logging in the governance package
- Adding Prometheus metrics for a new enterprise event
- Building evidence pack generation for regulatory compliance
- Configuring `FileSink`, `PrometheusSink`, or the enterprise logger
- Debugging missing audit events or telemetry gaps
- Working on EU AI Act / regulatory compliance telemetry requirements

## Inputs
- `packages/governance/src/calibrated_explanations_enterprise/governance/telemetry/`
- `packages/common/src/.../common/telemetry.py` — `TelemetryProtocol`
- `development/strategic-pillars/04_governance_visibility_and_audit.md`
- `development/strategic-pillars/06_observability_and_telemetry.md`

## Architecture

```
TelemetryProtocol (common)        ← abstract interface
    ├── NoOpTelemetryEmitter       ← default, used in tests
    ├── FileSink                   ← JSON Lines file logging
    └── PrometheusSink             ← Prometheus metrics backend

CalibratedGovernanceExplainer (governance)
    └── emits events to TelemetryProtocol on:
           - predict / predict_proba calls
           - explain_factual / explore_alternatives calls
           - calibration events
           - drift events (forwarded from adaptive)
```

## Telemetry Event Categories

### Governance Events (immutable audit trail)
- `prediction_made` — every inference with model name, timestamp, n_samples
- `explanation_generated` — explanation type, n_instances
- `calibration_updated` — recalibration with drift_detected flag
- `model_loaded` / `model_unloaded` — lifecycle events

### Operational Events (Prometheus metrics)
- `ce_requests_total` — Counter: total V2 inference requests, labels: `model_name`, `status`
- `ce_calibrations_total` — Counter: recalibrations, labels: `strategy`, `drift_detected`
- `ce_window_size` — Gauge: current buffer window size, labels: `strategy`
- `ce_latency_seconds` — Histogram: end-to-end inference latency, buckets: standard HTTP buckets
- `ce_drift_detections_total` — Counter: drift triggers, labels: `detector`, `strategy`

## Implementation Patterns

### Using the enterprise logger

```python
from calibrated_explanations_enterprise.governance.telemetry.ce_logging import (
    get_enterprise_logger
)

logger = get_enterprise_logger(__name__)
logger.info("Calibration updated", extra={
    "event_type": "calibration_updated",
    "model_name": self.model_name,
    "n_samples": len(X),
    "drift_detected": drift_result.triggered if drift_result else False,
})
```

### Adding a Prometheus metric

```python
# In governance/telemetry/metrics.py
from prometheus_client import Counter, Histogram, Gauge

ce_requests_total = Counter(
    "ce_requests_total",
    "Total V2 inference requests",
    ["model_name", "status"],
)

def record_request(model_name: str, status: str) -> None:
    ce_requests_total.labels(model_name=model_name, status=status).inc()
```

### Emitting to FileSink

```python
# FileSink writes JSON Lines format
sink = FileSink(log_path=Path("audit.jsonl"))
sink.emit_event("prediction_made", {
    "model_name": "credit_risk_v2",
    "n_samples": 10,
    "timestamp": "2026-02-28T12:00:00Z",
})
```

### Evidence pack generation (audit trail for regulators)

An evidence pack is a collection of:
1. Model audit log (all governance events for a time range)
2. Calibration history (checkpoint metadata + MLflow run IDs)
3. Drift detection log (all drift events with statistics)
4. Parity test results (timestamp of last passing run)

## Package Isolation Reminder
- Governance telemetry is in the `governance` package
- It must NOT import from `adaptive`
- `TelemetryProtocol` (the interface) lives in `common`
- `CalibratedAdaptiveExplainer` emits to telemetry via dependency injection, NOT via direct governance import

## Verification
```bash
pytest packages/governance/tests/ -q
pytest tests/integration/ -m integration -q
# Check Prometheus metrics endpoint works:
# curl http://localhost:8080/metrics
ruff check packages/governance/
```

## Output contract
For implementation tasks, return:
1. New event types added to the telemetry layer
2. Prometheus metric definitions with correct labels
3. Enterprise logger usage in the affected governance code
4. Tests that assert events are emitted (use `FileSink` in tests, not NoOp)

## Constraints
- Governance code must NEVER import from `adaptive`
- All telemetry events must have `event_type`, `timestamp`, and `model_name` fields minimum
- Prometheus metric names must start with `ce_` prefix
- Never use bare `logging.getLogger()` — always `get_enterprise_logger()`
- Evidence packs must be immutable once generated (no in-place modification)