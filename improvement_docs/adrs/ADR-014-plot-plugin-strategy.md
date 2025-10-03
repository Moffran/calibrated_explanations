# ADR-014: Plot Plugin Strategy

Status: Accepted  
Date: 2025-09-16 (updated 2025-10-02)  
Deciders: Core maintainers  
Reviewers: TBD  
Supersedes: ADR-013-interval-and-plot-plugin-strategy  
Superseded-by: None  
Related: ADR-006-plugin-registry-trust-model, ADR-007-visualization-abstraction-layer, ADR-016-plot-spec-separation

## Context
Explanation objects expose `.plot()` methods that currently delegate to helpers in `src/calibrated_explanations/_plots.py`. The helpers can either call the legacy matplotlib routines (`_plots_legacy.py`) or build a PlotSpec and render it with the `viz` adapter stack. Demand for alternative renderers (Plotly, Altair, static SVG) keeps growing, but any plugin model must retain parity with the data contracts emitted by explanation classes and maintain backwards compatibility with the legacy visuals that many users depend on.

Earlier revisions of this ADR described high-level goals but did not provide actionable guidance for implementers or plugin authors. As a result, no registry integration, interfaces, or configuration mechanisms were built. This update makes the strategy concrete so that both the legacy pathway and future plot plugins can coexist.

## Decision
1. **Define public builder and renderer protocols.**
   - Introduce `PlotRenderContext` (dataclass) that encapsulates the explanation payload, instance metadata, plotting intent (style identifier, uncertainty flags, etc.), and caller-supplied options such as `show`, `path`, `save_ext`.
   - Define `class PlotBuilder(Protocol)` with a `build(self, context: PlotRenderContext) -> PlotArtifact` method. `PlotArtifact` is a `typing.Protocol` union allowing `PlotSpec`, `dict` payloads, or backend-specific handles.
   - Define `class PlotRenderer(Protocol)` with `render(self, artifact: PlotArtifact, *, context: PlotRenderContext) -> PlotRenderResult`. `PlotRenderResult` exposes optional handles (figure object, primitive trace, saved paths).
   - Provide helper base classes in `src/calibrated_explanations/viz/plugins.py` to ease implementation (default no-op lifecycle hooks, validation helpers).

2. **Extend the plugin registry.**
   - Add registry namespaces for plots: `register_plot_builder(id, builder, metadata)`, `register_plot_renderer(id, renderer, metadata)`, `find_plot_builder(id)`, `find_plot_renderer(id)`.
   - Metadata must include: `style` (plot kind identifier), `output_formats`, `default_renderer`, `dependencies`, `legacy_compatible` flag, and shared ADR-006 fields (`trust`, `version`, `description`).
   - Builders and renderers register themselves on import. Entry points under `calibrated_explanations.plugins.plot_builders` and `...plot_renderers` remain supported for third-party packages.

3. **Configuration and resolution order.**
   - Resolution order for `.plot()` is: explicit kwargs (`style`, `renderer`) > environment variables (`CE_PLOT_STYLE`, `CE_PLOT_RENDERER`) > project configuration (`pyproject.toml` under `[tool.calibrated_explanations.plots]`) > package default.
   - The package default is `style="legacy"` with renderer `"legacy.matplotlib"`, preserving existing visuals.
   - Provide CLI helpers: `ce.plugins list --plots`, `ce.plugins validate-plot --builder <id>`, `ce.plugins set-default --plot-style <id>`.

4. **Legacy pathway as first-class plugin.**
   - Ship two built-in builder/renderer pairs:
     * `legacy` builder + renderer wrap `_plots_legacy`. They accept the PlotRenderContext but bypass PlotSpec entirely. This pairing remains the default to avoid regressions.
     * `bars/matplotlib` builder uses `viz.builders` to emit PlotSpecs and delegates rendering to the matplotlib adapter. This pairing demonstrates the new architecture and is used when `use_legacy=False` is requested.
   - The legacy plugin may emit a warning if invoked in environments without matplotlib to match existing behaviour.

5. **Validation and error handling.**
   - Provide `validate_plotspec(PlotSpec) -> None` in `viz/plotspec.py` and call it inside the default renderer path before drawing.
   - Registry resolution failures (missing builder/renderer) raise `PlotPluginError` with actionable messages. Legacy fallback is not implicit; users must opt in via configuration.
   - All plugins must declare dependency import guards and surface `MissingDependencyError` when optional libraries are unavailable.

6. **Documentation and tooling.**
   - Update developer docs to include a “Writing plot plugins” guide outlining the context object, validation helpers, registry registration, and testing strategies (including primitive exports).
   - Provide code examples for wrapping the legacy renderer, creating a new PlotSpec builder, and packaging a third-party renderer.

## Consequences
### Positive
- Clear separation of builder and renderer responsibilities enables new visualizations without modifying explanation classes.
- Registry metadata and CLI tooling give users visibility into available styles and diagnostics when resolution fails.
- Treating the legacy renderer as a plugin reduces duplication and clarifies the migration path.

### Negative / Risks
- Additional indirection complicates debugging; thorough logging and validation mitigate this.
- Plugin authors must understand the PlotRenderContext and PlotSpec schemas, raising the entry barrier compared with modifying `_plots.py` directly.
- Maintaining both the legacy plugin and the PlotSpec-based plugin increases test surface until the new path becomes the default.

## Status & Next Steps
- Protocols and registry extensions must be implemented in `viz/plugins.py` and `plugins/registry.py`.
- `.plot()` should resolve builders/renderers using the order defined above, defaulting to the legacy plugin to preserve behaviour.
- CLI commands and documentation updates are required before this ADR is considered fully realized.
- Future work: add automated tests covering registry resolution, plugin validation, and primitive export comparisons between legacy and PlotSpec builders.
