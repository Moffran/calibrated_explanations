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
- Plugins execute in-process without sandboxing or isolation. Only install and trust
  plugins from sources you control or have reviewed.
- Trust/deny controls and discovery are available via the registry and CLI; see contributor docs for details.
- For detailed wiring (env vars, pyproject, dependency seeding), see {doc}`contributor/extending/plugin-advanced-contract`.

Trust model (who, why, when)
----------------------------

**Why it exists:** Plugins run in-process with the same permissions as your application.
That means third-party code can access data, file systems, and network resources.
The trust model forces explicit operator approval to reduce supply-chain risk and
avoid accidental execution of unreviewed code.

**How it works (summary):**

- **Built-ins are trusted by default.** Only in-tree plugins ship as auto-trusted.
- **Third-party plugins require explicit trust.** Set `CE_TRUST_PLUGIN` or add an
  allowlist to `pyproject.toml` under `[tool.calibrated_explanations.plugins] trusted = ["id"]`.
- **Denied identifiers are blocked.** `CE_DENY_PLUGIN` skips loading and registration
  even if a plugin is otherwise discoverable.
- **Diagnostics are available.** Use the CLI to see trusted vs. untrusted plugins,
  and include skipped entry points when needed.

**Who should care:**

- **Operators / platform teams:** Govern what plugin code is allowed in production.
- **Security reviewers:** Audit trust/deny lists and verify plugin provenance.
- **Plugin authors:** Understand that `trusted` metadata is informational only until
  an operator explicitly trusts the identifier.

**When to use it:**

- **Development:** Use `CE_TRUST_PLUGIN` for quick, local opt-in.
- **CI/CD or production:** Prefer the `pyproject.toml` allowlist for auditable,
  versioned trust decisions.
- **Incident response or testing:** Use `CE_DENY_PLUGIN` to block a plugin without
  changing code.

CLI examples:

```bash
python -m calibrated_explanations.plugins.cli list all --trusted-only
python -m calibrated_explanations.plugins.cli list all --include-skipped
```

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
