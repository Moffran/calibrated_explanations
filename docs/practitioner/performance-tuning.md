# Performance Tuning Guide

This guide provides tips and configuration options for optimizing the performance of calibrated explanations, especially for large datasets or high-dimensional feature spaces.

## FAST-Based Feature Filtering (Experimental)

Calibrated explanations support an experimental feature to reduce computation time for factual and alternative explanations by filtering out unimportant features based on a preliminary FAST (Feature Attribution via Shapley Additive explanations) pass.

### Enabling Feature Filtering

Set the `CE_FEATURE_FILTER` environment variable to enable filtering:

```bash
export CE_FEATURE_FILTER="enable,top_k=8"
```

Configuration options:
- `enable` or `true`: Enable filtering
- `top_k=N`: Keep the top N most important features per instance (default: 8)
- `off` or `false`: Disable filtering

### How It Works

1. A FAST explanation is run on the batch to obtain per-instance feature weights.
2. For each instance, only the top-k features with the highest absolute weights are retained.
3. A global ignore set is computed for features that are unimportant across all instances.
4. The expensive factual/alternative explanation runs with the reduced feature set.

### Telemetry Events

When telemetry is enabled, the following events are emitted:

- `filter_enabled`: Filtering was attempted
- `filter_skipped`: Filtering was skipped due to configuration or errors
- `filter_error`: An error occurred during filtering

### Logging and Observability

Operational and governance logs follow :doc:`../standards/STD-005-logging-and-observability-standard`, keeping traces under domain-specific logger names and structured `extra` fields.

Sample operational log emitted by `calibrated_explanations.core.explain.feature_filter` when filtering falls back:

```
2026-01-15T12:00:00.123Z WARNING calibrated_explanations.core.explain.feature_filter.compute_filtered_features_to_ignore
  extra={
    "mode": "factual",
    "num_instances": 5,
    "per_instance_top_k": 8,
    "candidates": 32,
  }
```

When `CE_STRICT_OBSERVABILITY=1` the same site escalates to `WARNING` and a governance log is emitted under `calibrated_explanations.governance.feature_filter` with Standard-005 fields such as `decision`, `reason`, and `strict_observability`:

```
2026-01-15T12:00:00.124Z WARNING calibrated_explanations.governance.feature_filter
  extra={
    "decision": "filter_skipped",
    "mode": "factual",
    "reason": "missing 'predict' weights",
    "strict_observability": True,
  }
```

The governance logger can be routed to a dedicated sink while keeping operational logs on the main pipeline. A minimal ``dictConfig`` example:

```python
import logging.config

logging.config.dictConfig({
    "version": 1,
    "formatters": {
        "json": {
            "format": "%(message)s",
        },
    },
    "handlers": {
        "governance": {
            "class": "logging.StreamHandler",
            "formatter": "json",
            "stream": "sys.stderr",
        },
    },
    "loggers": {
        "calibrated_explanations.governance.feature_filter": {
            "handlers": ["governance"],
            "level": "INFO",
            "propagate": False,
        },
    },
})
```

Governance logs follow Standard-005’s domain differentiation, so they can be filtered or routed without impacting the operational log stream. The same configuration also demotes extra fields like ``explainer_id`` or ``plugin_identifier`` onto each record, keeping trace IDs in sync.

### Debugging Filtered Explanations

Filtered explanations include best-effort metadata for transparency:

```python
explanations = explainer.explain(X_test)
if hasattr(explanations, 'feature_filter_per_instance_ignore'):
    for i, ignored in enumerate(explanations.feature_filter_per_instance_ignore):
        print(f"Instance {i} ignored features: {ignored}")
```

This metadata shows which features were ignored for each instance due to filtering.

### Limitations

- Filtering is experimental and may change semantics in future versions.
- Only applies to factual and alternative modes.
- Falls back gracefully if FAST fails or configuration is invalid.
