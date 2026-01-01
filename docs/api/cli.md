# CLI reference

The package ships a CLI entry point, `ce.plugins`, for inspecting and managing
registered plugins.

## Entry point

Install the project and run:

```bash
ce.plugins --help
```

The CLI is backed by `calibrated_explanations.plugins.cli:main`.

## Common commands

- `ce.plugins list all` – list registered explanation, interval, and plot plugins.
- `ce.plugins list --plots` – list only plot plugins with their IDs and trust
  status.
- `ce.plugins validate-plot --builder <id>` – validate a plot builder's metadata
  and PlotSpec compatibility.
- `ce.plugins set-default --plot-style <id>` – persist the default plot style
  (for example, via `pyproject.toml` config).

## Related documentation

- Plugin governance and trust model: `docs/improvement/adrs/ADR-006-plugin-registry-trust-model.md`.
- Plot plugin lifecycle and metadata: `docs/improvement/adrs/ADR-014-plot-plugin-strategy.md`.
