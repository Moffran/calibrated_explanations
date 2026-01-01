# Plugin registry reference

Calibrated Explanations uses a centralized plugin registry to discover
explanation, interval, and plot builders/renderers.

## Registry scope

- **Registry module**: `src/calibrated_explanations/plugins/registry.py`
- **Built-in plugins**: `src/calibrated_explanations/plugins/builtins.py`
- **CLI tooling**: `calibrated_explanations.plugins.cli`

## Key concepts

- **Plugin IDs** are dot-delimited strings (`core.explanation.factual`).
- **Trusted vs. untrusted plugins** control which plugins are allowed to load by
  default.
- **Metadata validation** enforces required fields such as versioning and (for
  plots) the default renderer.

## Trust model

The trust model is documented in
`docs/improvement/adrs/ADR-006-plugin-registry-trust-model.md`. External plugins
should register through the approved entry points and include the required
metadata for auditability.

## Related ADRs

- ADR-006 (trust model)
- ADR-014 (plot plugin strategy)
- ADR-010 (core vs. evaluation split)
