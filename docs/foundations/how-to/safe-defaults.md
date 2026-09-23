# Safe defaults for production deployments

This guide documents the recommended configuration for production deployments
of `calibrated_explanations`. It complements the full env var reference in
[configure_runtime.md](configure_runtime.md) and the performance tuning guide
in [tune_runtime_performance.md](tune_runtime_performance.md).

---

## 1. Recommended env var settings for production

The library's defaults are conservative. Most env vars should be left unset in
production.

| Key | Production recommendation | Notes |
|-----|--------------------------|-------|
| `CE_CACHE` | **Unset** (disabled by default) | Enable only after benchmarking; see [tune_runtime_performance.md](tune_runtime_performance.md) for sizing |
| `CE_PARALLEL` | **Unset** (disabled by default) | Enable only with an explicit strategy after understanding fork/spawn hygiene in your deployment environment |
| `CE_STRICT_OBSERVABILITY` | Unset or `false` | Set `true` in CI to surface soft observability errors |
| `CE_TELEMETRY_DIAGNOSTIC_MODE` | Unset or `false` | Set `true` only for diagnostic sessions |
| `CE_DEBUG_TRUST_INVARIANTS` | **Never set in production** | Bypasses trust invariant checks; is a sanctioned direct `os.getenv` read not governed by ConfigManager — setting it disables plugin trust enforcement |
| `CE_TRUST_PLUGIN` | Set via pyproject.toml instead | Env var overrides pyproject.toml; prefer declarative config for production |
| `CE_DENY_PLUGIN` | Unset (empty denylist) | Add identifiers only when you have a specific plugin to block |
| `CE_DEPRECATIONS` | Unset (warnings only) | Set `error` in CI for migration verification — not in production |

For any plugin selection keys (`CE_EXPLANATION_PLUGIN_*`, `CE_INTERVAL_PLUGIN_*`,
`CE_PLOT_STYLE`, etc.) — if not set, the library uses the pyproject.toml value
then the versioned default. Override only when you have a specific non-default
plugin to use.

### Validate cache and parallel telemetry in staging

Cache and parallel telemetry are optional diagnostic instrumentation. Enabling
`CE_CACHE` or `CE_PARALLEL` does not install a collector; register a callback
with `ExplainerBuilder.perf_telemetry(...)`, `CacheConfig.telemetry`, or
`ParallelConfig.telemetry` when staging evidence is needed.

Parallel execution in v1 requires an explicit strategy. `CE_PARALLEL=1` enables
parallelism but leaves the removed `auto` strategy selected, so configuration
fails fast with `ConfigurationError`: `build_config()` and
`WrapCalibratedExplainer.from_config()` raise on the builder path, and
`calibrate()` raises on the plain `WrapCalibratedExplainer` path. The same
applies to `perf_parallel(True)` without a `backend=`. Use an explicit value
such as:

```bash
CE_PARALLEL="enable,threads,workers=2,min_batch=1,min_instances=1,tiny=1" python serve.py
```

The optional callback receives the following evidence fields. These names
describe the current diagnostic surface; ADR-003 and ADR-004 do not make
telemetry collection mandatory for production deployments.

| Subsystem | Event | Evidence fields |
|-----------|-------|-----------------|
| Cache | `cache_hit`, `cache_miss` | `namespace` and `key`; `CacheMetrics.snapshot()` exposes aggregate `hits` and `misses` counters |
| Parallel | `parallel_execution` | `strategy`, `items`, `duration`, `workers`, `worker_utilisation_pct`, `work_items`, and threshold/granularity context |

`worker_utilisation_pct` is a saturation proxy calculated from submitted items
and configured workers; it is not an operating-system CPU utilisation measure.

---

## 2. Recommended pyproject.toml configuration

Declare plugin trust declaratively rather than via environment variable:

```toml
[tool.calibrated_explanations.plugins]
# Allowlist: only plugins listed here are trusted. Empty list = no plugins trusted.
trusted = ["your-plugin-id"]

# No need to set explanations/intervals/plots overrides unless you are
# explicitly customising plugin selection from the versioned defaults.
```

ConfigManager reads pyproject.toml at process start. Snapshot semantics mean
changes require a process restart to take effect.

---

## 3. Diagnostics with `export_effective()`

Use the process-level config singleton for in-process diagnostics:

```python
from calibrated_explanations.core.config_manager import get_process_config_manager

snapshot = get_process_config_manager().export_effective()
# Keys are namespaced: "effective.<KEY>", "env.<KEY>", "pyproject.<section>",
# and "diagnostic.*". Resolved values live under the "effective." prefix.
for key, value in snapshot.values.items():
    print(f"{key}: {value!r}  [{snapshot.sources[key]}]")
```

The source shows where each value came from: `env`, `pyproject`, or `default`.

CLI equivalent (standalone one-shot snapshot; does **not** inspect a running
process):

```bash
python -m calibrated_explanations.cli config show
```

**ConfigManager is snapshot-based.** Env var changes made after process start
are not visible to runtime components without reconstructing the manager or
restarting the process. Do not use `ConfigManager.from_sources().export_effective()`
as a diagnostic pattern — that creates an isolated snapshot unrelated to what
runtime components use.

---

## 4. Plugin security posture

- **Allowlist** (`CE_TRUST_PLUGIN` / `pyproject.toml plugins.trusted`): only
  plugins in this list are treated as trusted.
- **Denylist** (`CE_DENY_PLUGIN`): plugins in this list are always rejected,
  even if also in the allowlist.

Production recommendation: explicit allowlist, empty denylist. Do not rely on
default-open behaviour.

`CE_DEBUG_TRUST_INVARIANTS` bypasses trust invariant enforcement entirely. It
is intended only for internal debugging and must never be set in production. It
is read via a sanctioned direct `os.getenv` call and is not governed by
`ConfigManager`, so it is not visible in `export_effective()` output.

---

## 5. `CE_DEPRECATIONS` for CI migration verification

`CE_DEPRECATIONS` is not routed through ConfigManager but is essential for RC
adopters. Set it in CI to catch any remaining use of deprecated APIs before
promoting to v1.0.0:

```bash
CE_DEPRECATIONS=error pytest your_tests/
```

This converts `DeprecationWarning` emissions from the `deprecate()` helper to
hard errors, surfacing deprecated API usage that would silently succeed in
production. Do not set `CE_DEPRECATIONS=error` in production — it will break
any code that still uses deprecated symbols.

See [deprecations.md](../../migration/deprecations.md) for the full list of
currently deprecated symbols and their replacements.
