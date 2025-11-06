# Plugins

Calibrated Explanations supports an optional, extensible plugin system. By default,
you don’t need plugins to run calibrated factual and alternative explanations.
When you want speed-ups (e.g., FAST) or custom visualizations, install a curated
external bundle and wire it in. If you’re extending the framework, follow the
plugin contract to preserve calibration semantics.

Choose your path:

- For practitioners: Use external plugins to enable optional speed-ups and plots → {doc}`practitioner/advanced/use_plugins`
- For contributors: Develop plugins that honor the CE contract → {doc}`contributor/plugin-contract`

Community listings and the curated install extra live here: {doc}`appendices/external_plugins`.

Notes

- Plugins are optional and externally distributed. Core workflows work without them (ADR-027).
- Wiring methods (priority order):
  1) Explainer parameters; 2) Environment variables; 3) pyproject.toml; 4) Plugin-declared dependencies.
- Trust/deny controls and discovery are available via the registry and CLI; see contributor docs for details.
- For detailed wiring (env vars, pyproject, dependency seeding), see {doc}`contributor/extending/plugin-advanced-contract`.
