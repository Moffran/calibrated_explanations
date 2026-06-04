# Configure runtime behaviour (ConfigManager)

`ConfigManager` (ADR-034) is the single authoritative source for all runtime
configuration in `calibrated_explanations`. It captures the process environment
and any `pyproject.toml` settings once at construction time so every subsequent
lookup resolves against a deterministic, immutable view.

## Precedence order

Configuration keys resolve in this order (highest to lowest):

1. Call-site override (highest)
2. Environment variable
3. `pyproject.toml` (`[tool.calibrated_explanations.*]`)
4. Versioned default profile (lowest)

`ConfigManager` is snapshot-based: it captures env and pyproject content at
construction time and does not observe later environment changes. Code that
depends on mutated env values must reconstruct a new `ConfigManager`.

## Process-level singleton

Most runtime code accesses configuration through the process-level singleton:

```python
from calibrated_explanations.core.config_manager import get_process_config_manager

cfg = get_process_config_manager()
print(cfg.env("CE_PLOT_STYLE"))          # resolved env value or None
print(cfg.pyproject_section("plots"))    # resolved pyproject section dict
```

The singleton is constructed lazily on first call and reused for the lifetime
of the process. To install a custom manager before any runtime component
initializes (e.g. in a server startup routine), call
`init_process_config_manager()` once at the process boundary:

```python
from calibrated_explanations.core.config_manager import (
    ConfigManager,
    init_process_config_manager,
)

manager = ConfigManager.from_sources(strict=False)  # permissive validation
init_process_config_manager(manager)
```

A second call raises `CalibratedError` because double initialization would
make snapshot ownership ambiguous.

## Supported environment variables

The table below lists every recognized `CE_*` key and the equivalent
`pyproject.toml` path where one exists. Keys marked *env-only in v0.11.x*
have no pyproject.toml wiring yet; they are intentionally env- or
programmatic-only for this release cycle.

| Environment variable | pyproject.toml key | Description |
|---|---|---|
| `CE_EXPLANATION_PLUGIN` / `CE_EXPLANATION_PLUGIN_FACTUAL` | `explanations.factual` | Active factual explanation plugin ID |
| `CE_EXPLANATION_PLUGIN_ALTERNATIVE` | `explanations.alternative` | Active alternative explanation plugin ID |
| `CE_EXPLANATION_PLUGIN_FAST` | `explanations.fast` | Active fast explanation plugin ID |
| `CE_EXPLANATION_PLUGIN_FACTUAL_FALLBACKS` | `explanations.factual_fallbacks` | Ordered fallback chain for factual plugin |
| `CE_EXPLANATION_PLUGIN_ALTERNATIVE_FALLBACKS` | `explanations.alternative_fallbacks` | Ordered fallback chain for alternative plugin |
| `CE_EXPLANATION_PLUGIN_FAST_FALLBACKS` | `explanations.fast_fallbacks` | Ordered fallback chain for fast plugin |
| `CE_INTERVAL_PLUGIN` | `intervals.default` | Active interval calibrator plugin ID |
| `CE_INTERVAL_PLUGIN_FAST` | `intervals.fast` | Fast interval calibrator plugin ID |
| `CE_INTERVAL_PLUGIN_FALLBACKS` | `intervals.default_fallbacks` | Ordered fallback chain for default interval |
| `CE_INTERVAL_PLUGIN_FAST_FALLBACKS` | `intervals.fast_fallbacks` | Ordered fallback chain for fast interval |
| `CE_PLOT_STYLE` | `plots.style` | Active plot style plugin ID |
| `CE_PLOT_STYLE_FALLBACKS` | `plots.style_fallbacks` | Ordered fallback chain for plot style |
| `CE_PLOT_RENDERER` | `plots.renderer` | Active plot renderer plugin ID |
| `CE_TRUST_PLUGIN` | `plugins.trusted` | Plugin IDs to trust (comma-separated list) |
| `CE_DENY_PLUGIN` | *(env-only)* | Plugin IDs to deny explicitly |
| `CE_PLUGIN_CONFIG_JSON` | *(env-only)* | JSON blob of per-plugin config overrides |
| `CE_TELEMETRY_DIAGNOSTIC_MODE` | `telemetry.diagnostic_mode` | Enable telemetry diagnostic mode (`true`/`false`) |
| `CE_CACHE` | *(env-only in v0.11.x)* | Cache toggle and settings (e.g. `enable,max_items=512,ttl=600`) |
| `CE_PARALLEL` | *(env-only in v0.11.x)* | Parallel backend toggle and settings |
| `CE_PARALLEL_MIN_BATCH_SIZE` | *(env-only in v0.11.x)* | Minimum batch size threshold for parallel execution |
| `CE_FEATURE_FILTER` | *(env-only in v0.11.x)* | FAST feature filter toggle and `top_k` setting |
| `CE_STRICT_OBSERVABILITY` | *(env-only in v0.11.x)* | Strict observability mode for config/filter failures |

Unknown `CE_*` variables emit a `UserWarning` rather than aborting, because
environment namespaces can contain stale or mistyped settings.

## pyproject.toml sections

Persistent project-level configuration lives under
`[tool.calibrated_explanations.*]`. Settings here take effect for all
processes that share the same working directory, making them suitable for
project-wide defaults:

```toml
[tool.calibrated_explanations.explanations]
factual = "my_org.factual_v2"
factual_fallbacks = ["core.factual.default", "legacy"]

[tool.calibrated_explanations.intervals]
default = "core.interval.fast"
default_fallbacks = ["core.interval.legacy"]

[tool.calibrated_explanations.plots]
style = "plot_spec.default"
style_fallbacks = ["legacy"]
renderer = "matplotlib"

[tool.calibrated_explanations.plugins]
trusted = ["my_org.factual_v2"]

[tool.calibrated_explanations.telemetry]
diagnostic_mode = true
```

These values are captured by `ConfigManager.from_sources()` at startup and
resolved below environment-variable values.

## Plugin-specific configuration

Per-plugin parameters can be supplied through `pyproject.toml` or via
`CE_PLUGIN_CONFIG_JSON`. Both sources are merged with environment values
taking precedence:

```toml
# pyproject.toml
[tool.calibrated_explanations.plugin_configs."my_org.factual_v2"]
threshold = 0.05
max_features = 10
```

```bash
# Equivalent as an environment variable (JSON)
CE_PLUGIN_CONFIG_JSON='{"my_org.factual_v2": {"threshold": 0.05, "max_features": 10}}'
```

Retrieve resolved plugin config in Python:

```python
from calibrated_explanations.core.config_manager import get_process_config_manager

cfg = get_process_config_manager()

# Frozen, immutable mapping of resolved key/value pairs
plugin_cfg = cfg.plugin_config("my_org.factual_v2")
print(plugin_cfg)       # MappingProxy({"threshold": 0.05, "max_features": 10})

# Per-key source attribution ("pyproject", "env", or "override")
sources = cfg.plugin_config_sources("my_org.factual_v2")
print(sources)          # {"threshold": "pyproject", "max_features": "pyproject"}

# Which plugin IDs have any raw config present
print(cfg.configured_plugin_ids())   # ("my_org.factual_v2",)
```

```{note}
Plugin config validation — schema binding, defaults, semantic checks — happens
*after* plugin selection and trust resolution through the registry, not at
ConfigManager construction time. The raw values returned by `plugin_config()`
are provisional until trusted plugin code binds them to the plugin's own schema.
```

## Export effective configuration for debugging

`export_effective()` returns a frozen snapshot of all resolved values with
per-key source attribution, safe to include in support bundles:

```python
from calibrated_explanations.core.config_manager import get_process_config_manager

cfg = get_process_config_manager()
snapshot = cfg.export_effective()

print(f"Profile: {snapshot.profile_id}, schema: {snapshot.schema_version}")
for key, value in snapshot.values.items():
    source = snapshot.sources[key]
    print(f"  {key}: {value!r}  [{source}]")
```

Secret-like keys (passwords, tokens, API keys, credentials) are automatically
redacted to `"<redacted>"`. Schema-marked sensitive plugin config values are
also redacted when a plugin config schema is supplied.

The CLI equivalents are:

```bash
# Print effective config for the current process invocation
python -m calibrated_explanations.cli config show

# Write effective config to a file
python -m calibrated_explanations.cli config export --output config_snapshot.json
```

## Strict vs. permissive validation

By default, `ConfigManager` uses strict validation (`strict=True`): unknown
pyproject keys and type/value mismatches raise `ConfigurationError` at
construction. Use `strict=False` to collect all issues in a validation report
without aborting startup:

```python
from calibrated_explanations.core.config_manager import ConfigManager

manager = ConfigManager.from_sources(strict=False)
report = manager.validation_report()
if report.has_errors:
    for issue in report.issues:
        print(f"{issue.location}: {issue.message}")
```

This is useful for migration scenarios or when deploying a new pyproject.toml
configuration into a staging environment to surface all problems at once.

## Telemetry diagnostic mode

Enable telemetry diagnostics either via the environment variable or via
`pyproject.toml`:

```bash
CE_TELEMETRY_DIAGNOSTIC_MODE=true python run_batch.py
```

```toml
[tool.calibrated_explanations.telemetry]
diagnostic_mode = true
```

Query the resolved value in Python:

```python
cfg = get_process_config_manager()
print(cfg.telemetry_diagnostic_mode())   # True or False
```

## See also

- [Tune runtime performance](tune_runtime_performance.md) — cache, parallel, and
  feature-filter env-var settings
- [Configure telemetry](configure_telemetry.md) — telemetry diagnostics and
  governance logging
- ADR-034: `docs/improvement/adrs/ADR-034-centralized-configuration-management.md`
