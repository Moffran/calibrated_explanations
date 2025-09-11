# Plugin registry and trust model

This page shows the minimal, opt-in plugin registry shipped in ADR-006 and how
to use it today, plus a short note about the planned full workflow and the
limits of what plugins can (and cannot) do.

## Quick start (today)

The repository includes a tiny example plugin used by tests at
`tests/plugins/example_plugin.py`.

To register and trust a plugin at runtime you can do the following:

```py
from calibrated_explanations.plugins import registry
from tests.plugins.example_plugin import PLUGIN

# Register the plugin (no side-effects beyond adding to the in-process registry)
registry.register(PLUGIN)

# Optionally mark the plugin as trusted (controls discoverability via 'trusted' helpers)
registry.trust_plugin(PLUGIN)

# List registered plugins
print(registry.list_plugins())

# Find plugins that claim to support a model
registry.find_for("supported-model")
# Find trusted plugins only
registry.find_for_trusted("supported-model")

# Remove or untrust when needed
registry.untrust_plugin(PLUGIN)
registry.unregister(PLUGIN)
```

Notes:

- `register` validates minimal metadata (see `plugins.base.validate_plugin_meta`) and will raise `ValueError` for malformed `plugin_meta`.
- `trust_plugin` is an explicit opt-in step; the registry does not implicitly trust third-party code.
- The registry is in-process and performs no automatic sandboxing or network calls.

Important: current status

- The registry and helper APIs are available, but plugins are not yet integrated into the library's main explain flows by default. Registering a plugin adds it to the in-process registry and allows discovery via `find_for`/`find_for_trusted`, but it will not automatically be invoked by core expose functions unless you explicitly call the plugin's API. This minimizes risk while the trust model and discovery semantics are stabilized.

Example: invoking a registered plugin explicitly

```py
# after register/ trust as above
from calibrated_explanations.plugins import registry

plugin = registry.find_for("supported-model")[0]
# Call the plugin's explain method directly (explicit invocation) — this executes plugin code in-process
result = plugin.explain("supported-model", X=[1, 2, 3])
print(result)
```

## Planned full-support workflow (what will be available when ADR-006 is fully realized)

When the plugin model is fully supported the plan is to provide:

- Entry-point discovery: plugins can be discovered via the `calibrated_explanations.plugins`
  setuptools entry point group or explicit programmatic registration.
- Environment & CLI opt-in: a user can pre-authorize plugins using environment variables (e.g., `CE_TRUST_PLUGIN=<name>`) or a one-time programmatic trust API to allow safe, repeatable runs.
- Richer metadata: plugins will expose structured metadata (name, version, provider, capabilities, optional checksum) to help discovery and basic compatibility checks.
- Trust controls: `list_plugins(include_untrusted=True)` for diagnostics, and `find_for_trusted(...)` to only consider trusted plugins during automated discovery.
- Documentation and examples: published examples showing how to write a plugin that implements explanation strategies and visualization adapters.

These features are deliberately conservative: the trust model is opt-in and the registry will emit actionable warnings when untrusted plugins are present.

## What plugins are intended to do (capabilities)

Plugins are intended to be small, focused extension points such as:

- Custom explanation strategies (implement `explain(model, X, **kwargs)` and return either a domain `Explanation` or a legacy explanation dict).
- Lightweight visualization adapters or renderers (return `PlotSpec` or render directly when requested by caller).
- Small helper adapters that adapt model types not supported by the core library.

Each plugin should declare its capabilities via `plugin_meta["capabilities"]` so callers can filter available plugins by the features they need.

## Limits and security considerations (important)

- In-process execution only: plugins run in the same Python process as the host library. There is no sandboxing. A plugin can execute arbitrary Python code and therefore may access or modify process state.
- Coarse trust flag: the `trusted` marker is an explicit opt-in but it is not a security boundary. Treat it as a usability guard, not a sandbox.
- Metadata validation only: the registry validates only minimal metadata (presence and basic types). It does not verify code provenance or cryptographic signatures. Optional checksum fields may be supported later but will be best-effort.
- API surface constraints: plugins are limited to the public plugin Protocol (see `calibrated_explanations.plugins.base.ExplainerPlugin`) — they cannot alter private internals unless those internals are explicitly exposed by the library's plugin hooks or public APIs.
- Schema compatibility: explanation outputs should conform to the documented Explanation schema (schema v1) or to the legacy shapes accepted by adapters; otherwise consumers may fail downstream serializers or visualizers.

## Recommended practices for plugin authors

- Keep plugins focused and small (one responsibility: explainers vs visualizers).
- Declare capability metadata and a stable `name` and `version` in `plugin_meta`.
- Avoid side-effects at import time; prefer lazy initialization and explicit `register` calls.
- Provide a small test-suite or smoke test so consumers can validate behavior.

---
For more on the ADR and the trust model, see `improvement_docs/adrs/ADR-006-plugin-registry-trust-model.md`.
