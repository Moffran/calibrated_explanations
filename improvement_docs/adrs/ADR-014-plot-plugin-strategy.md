# ADR-014: Plot Plugin Strategy

Status: Proposed
Date: 2025-09-16
Deciders: Core maintainers
Reviewers: TBD
Supersedes: ADR-013-interval-and-plot-plugin-strategy
Superseded-by: None
Related: ADR-006-plugin-registry-trust-model, ADR-007-visualization-abstraction-layer, ADR-013-interval-calibrator-plugin-strategy

## Context

Explanation objects expose .plot() methods that currently delegate to helpers in src/calibrated_explanations/_plots.py. Those helpers build matplotlib figures for probabilistic and regression explanations and lean on the PlotSpec abstraction introduced in ADR-007 for serialization and backend independence. Demand for alternative renderers (e.g., Plotly, Altair, static SVG exporters) is growing, but any plugin model must keep parity with the data contracts emitted by the explanation classes. Plot plugins therefore need dedicated guidance separate from interval calibrators while continuing to integrate with the shared plugin registry.

## Decision

1. **Define plot builder and renderer protocols.**
   - Introduce a PlotBuilderPlugin protocol exposed through the registry. Its build(context) method receives a context containing the explanation payload (ExplanationPayload), default style, DPI/theme hints, and helper utilities that mirror the existing _plot_probabilistic, _plot_regression, _plot_alternative, and _plot_triangular functions.
   - Builders must output a validated PlotSpec. They are required to preserve semantic keys already used by built-in plots (axes labels, confidence intervals, class labels) so downstream consumers remain compatible.
   - Optional PlotRendererPlugin protocol converts a PlotSpec into a concrete artifact (matplotlib figure, Plotly figure, SVG bytes, etc.). Builders may bundle a renderer when tightly coupled; otherwise the registry pairs builders and renderers based on declared capabilities.

2. **Registry namespaces and metadata.**
   - Extend the plugin registry with register_plot_builder, register_plot_renderer, find_plot_builder, and find_plot_renderer helpers. Built-in matplotlib builder and renderer register themselves on import.
   - Metadata requirements: styles (supported style identifiers such as probabilistic or regression), output_format, optional renderer, dependencies, themes, along with shared ADR-006 metadata fields. Capability tags include plot:build:\<style> and plot:render:\<format> to support diagnostics and compatibility checks.

3. **Configuration entry points.**
   - Environment variables: CE_PLOT_STYLE selects the default builder, CE_PLOT_RENDERER forces a renderer backend, and CE_PLOT_THEME picks a registered theme (falling back to builder defaults).
   - CalibratedExplainer kwargs and explanation .plot() parameters accept a plot_style identifier and optional plot_renderer. Passing callables is allowed for advanced use cases, but the registry remains the canonical lookup path.
   - pyproject.toml entries under [tool.calibrated_explanations.plugins] (keys plot_style, plot_renderer, plot_theme) provide project defaults hydrated via CLI helpers (ce.plugins apply-config).

4. **Lifecycle and validation.**
   - .plot() resolves the builder style via kwargs, project config, or environment variables. The registry returns a trusted builder whose build output undergoes validate_plotspec before rendering.
   - When a renderer is requested the registry resolves it using output_format compatibility checks. If the builder bundles a renderer identifier it wins unless the caller overrides it.
   - Validation failures surface clear diagnostics (missing fields, unsupported styles, dependency import errors) and never fall back silently to partial plots.

5. **Documentation and tooling.**
   - Developer docs add a section explaining the builder and renderer protocols, the expected PlotSpec structure for each built-in style, and migration guides for existing bespoke plotting code.
   - CLI tooling exposes ce.plugins list --plots, ce.plugins validate-plot --builder <id> commands to introspect capabilities and run schema validation on plugin outputs.

## Security & Guardrails

- Trust semantics mirror ADR-006: only trusted builders or renderers are used automatically. Untrusted plugins can be listed but must be explicitly selected by the user.
- Metadata validation occurs at registration. Missing or malformed metadata prevents plugin activation and surfaces actionable errors.
- Builders and renderers execute in-process and therefore share the host process permissions; documentation reiterates that enabling untrusted plugins executes arbitrary code.
- Schema validation ensures external builders cannot emit malformed specs that would break downstream renderers or serialization.

## Consequences

Positive:
- Clear separation of builder and renderer concerns enables new visualization surfaces without touching core explanation code.
- Registry metadata gives users insight into available styles, themes, and dependencies, aiding diagnostics and support tooling.
- Built-in plots become formal plugins, easing maintenance and incremental refactors.

Negative / Risks:
- Additional indirection complicates debugging when plot generation fails; strong validation and CLI tooling mitigate this risk.
- Supporting multiple renderers introduces more dependency combinations to test and document.
- Plugin authors must understand the PlotSpec schema in depth, which raises the entry barrier compared with ad-hoc matplotlib hooks.

## Non-Goals

- Defining new explanation payload schemas; plot plugins operate on the existing ExplanationPayload structures.
- Introducing remote hosting, sandboxing, or auto-installation of plotting plugins. Dependency management remains the responsibility of plugin authors.
- Mandating a specific rendering backend; the matplotlib based plugin remains the default implementation.

## Future Work

- Wrap the current matplotlib helpers as builder and renderer plugins to serve as reference implementations and examples.
- Expand integration tests to cover registry lookups, style overrides, renderer fallbacks, and schema validation errors.
- Evaluate whether a theme registry or parameterisation helper is needed once third-party builders appear.
