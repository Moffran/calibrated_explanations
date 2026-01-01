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

- Plugins are optional and externally distributed. Core workflows work without them (STD-027).
- Wiring methods (priority order):
  1) Explainer parameters; 2) Environment variables; 3) pyproject.toml; 4) Plugin-declared dependencies.
- Trust/deny controls and discovery are available via the registry and CLI; see contributor docs for details.
- For detailed wiring (env vars, pyproject, dependency seeding), see {doc}`contributor/extending/plugin-advanced-contract`.

Writing plot plugins
--------------------

This short guide helps implement a plot plugin compatible with the ADR-014 plugin
contracts.

- **Base classes**: Subclass `BasePlotBuilder` and `BasePlotRenderer` from
  `src/calibrated_explanations/viz/plugins.py`. Use `initialize()` to capture a
  `PlotRenderContext` and implement `build()` / `render()`.
- **Validation**: If your builder emits a PlotSpec-shaped mapping, call
  `validate_plotspec()` (available from `src/calibrated_explanations/viz/serializers.py`) to
  surface schema problems early. The base renderer will also attempt best-effort
  validation for PlotSpec payloads.
- **Metadata**: Provide `plugin_meta` on both builder and renderer. Builders may
  include an optional `default_renderer` key recommending a renderer id.
- **Registration**: Register your builder/renderer via the registry helpers:
  `register_plot_builder(<id>, builder)` and `register_plot_renderer(<id>, renderer)`;
  then map a style with `register_plot_style(<style_id>, metadata={...})`.
- **CLI helpers**: Use `ce.plugins list --plots` to inspect style mappings,
  `ce.plugins validate-plot --builder <id>` to run a dry build, and
  `ce.plugins set-default --plot-style <id>` to set the default style.

See `src/calibrated_explanations/viz/plugins.py` for the minimal base classes and
`src/calibrated_explanations/plugins/cli.py` for the CLI tooling examples.
