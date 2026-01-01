> **Status note (2025-10-24):** Last edited 2025-10-24 · Archive after: Retain indefinitely as architectural record · Implementation window: Per ADR status (see Decision).

# ADR-016: PlotSpec separation and legacy parity

Status: Accepted (scoped)

## Context
The 0.5.x plotting stack mixed multiple responsibilities inside a handful of matplotlib-heavy functions. As part of the visualization refactor we introduced `PlotSpec` builders and a renderer adapter so that plots can be expressed in a backend-agnostic format. Subsequent revisions of this ADR attempted to document every primitive, colour, and axis detail. That level of prescription has proven counter-productive: the implementation has drifted, tests cannot meet the exhaustive requirements, and the document no longer reflects the hybrid reality where the legacy renderer is still the default code path.

This update narrows the decision to the essential invariants needed for interoperability and clarifies how legacy rendering coexists with the PlotSpec pathway.

## Decision
1. **Semantic contract only**
   - This ADR defines the minimum semantic requirements for PlotSpec payloads so adapters can interpret them consistently.
   - Exact reproduction of historical colours, layout quirks, and pixel-level rendering is **out of scope** for PlotSpec and remains the responsibility of the legacy renderer.

2. **Plot kinds and metadata (minimal)**
   - `PlotSpec.kind` identifies the semantic plot type (e.g., probabilistic, regression, triangular, global). Additional kinds may be introduced as needed.
   - Each spec MUST include `mode` (`"classification" | "regression"`), and SHOULD include `title` and `feature_order` when features are displayed.
   - Header/body layouts are optional and should use the dataclasses defined in `viz/plotspec.py` when present.

3. **Validation and testing hooks**
   - `viz/plotspec.validate_plotspec` performs structural checks (presence of kind/mode, valid dataclass instances, numeric intervals). Builders should call this helper before handing control to adapters.
   - Tests should assert semantic correctness (ordering, interval coverage, required fields) rather than pixel-perfect colour matching.

4. **Hybrid execution**
   - The package supports two execution paths: the legacy renderer (default) and the PlotSpec adapter path (opt-in).
   - When the PlotSpec path becomes the default, this ADR should be revisited to tighten requirements and, if necessary, deprecate the legacy plugin.

## Consequences
- **Pros**: The decision captures the minimum contract required for interoperability, reducing churn between code and documentation. Plugin authors have a clear understanding of which fields are mandatory and where they have flexibility.
- **Cons**: Less prescriptive guidance means adapters may diverge visually. We rely on validation helpers and documentation to maintain coherence across implementations.

## Implementation notes
- Update `viz/plotspec.py` with validation utilities and ensure builders invoke them.
- Document the acceptable dictionary structure for non-panel plots in developer docs.
- Review tests to ensure they assert semantic correctness (ordering, interval coverage) rather than pixel-perfect colour matching.

## Future work
- Formalise schemas for triangular and global plots once at least two adapters require them.
- Introduce PlotSpec versioning if breaking changes become necessary.
- Evaluate when the PlotSpec pathway is stable enough to become the default renderer and retire the legacy plugin.
