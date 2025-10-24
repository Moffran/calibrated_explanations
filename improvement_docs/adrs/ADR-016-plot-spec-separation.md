> **Status note (2025-10-24):** Last edited 2025-10-24 · Archive after: Retain indefinitely as architectural record · Implementation window: Per ADR status (see Decision).

# ADR-016: PlotSpec separation and legacy parity

Status: Accepted (updated 2025-10-02)

## Context
The 0.5.x plotting stack mixed multiple responsibilities inside a handful of matplotlib-heavy functions. As part of the visualization refactor we introduced `PlotSpec` builders and a renderer adapter so that plots can be expressed in a backend-agnostic format. Subsequent revisions of this ADR attempted to document every primitive, colour, and axis detail. That level of prescription has proven counter-productive: the implementation has drifted, tests cannot meet the exhaustive requirements, and the document no longer reflects the hybrid reality where the legacy renderer is still the default code path.

This update narrows the decision to the essential invariants needed for interoperability and clarifies how legacy rendering coexists with the PlotSpec pathway.

## Decision
1. **Plot kinds and metadata**
   - `PlotSpec.kind` identifies the semantic plot type: `factual_probabilistic`, `alternative_probabilistic`, `factual_regression`, `alternative_regression`, `triangular`, `global_probabilistic`, `global_regression`. Additional kinds may be introduced as needed.
   - Each spec MUST include `mode` (`"classification" | "regression"`), optional `title`, and `feature_order` (list of indices used for the plot) when features are displayed.
   - Header/body layouts are optional but when present MUST use the dataclasses defined in `viz/plotspec.py` (`IntervalHeaderSpec`, `BarHPanelSpec`, `BarItem`).
   - Saving hints are represented via `save_behavior` with keys `{path, title, default_exts}` when callers request filesystem output.

2. **Legacy parity through plugins**
   - Exact reproduction of historical colours, layout quirks, and interval heuristics is delegated to the `legacy` plot plugin (see ADR-014). PlotSpec builders should focus on semantic correctness, not byte-for-byte parity.
   - Adapters MAY expose a `legacy_mode` flag that toggles parity behaviour when required for tests, but ADR-016 does not mandate specific hex values or primitive identifiers.

3. **Non-panel plots**
   - Until a richer schema is defined, triangular and global plots MAY return plain dict payloads that follow the structure produced by `viz.builders.build_triangular_plotspec_dict` and `build_global_plotspec_dict`. Adapters are responsible for interpreting these dictionaries.
   - Future schema enhancements should be versioned and documented alongside builder updates; this ADR intentionally leaves room for evolution.

4. **Validation and testing hooks**
   - `viz/plotspec.validate_plotspec` performs structural checks (presence of kind/mode, valid dataclass instances, numeric intervals). Builders must call this helper before handing control to adapters.
   - The matplotlib adapter continues to support `export_drawn_primitives=True`, returning a simplified record of solids/overlays/header elements. Tests may rely on this trace but ADR-016 does not standardise the format beyond requiring that it be JSON-serialisable.

5. **Hybrid execution**
   - The package supports two execution paths: the legacy renderer (default) and the PlotSpec adapter path (opt-in). Both paths must remain functional. PlotSpec builders should not assume they are the only rendering mechanism.
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
