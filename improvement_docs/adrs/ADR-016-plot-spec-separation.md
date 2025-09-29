# ADR 016: Plot spec separation and uncertainty parity

Status: Proposed

Context
-------
The plotting code in v0.5.1 multiplexed multiple plotting modes (factual vs
alternative, probabilistic vs regression) into a small set of functions. This
led to complex branching, implicit color-role swaps, and brittle visual
semantics. Tests did not capture intended visual invariants (e.g. parity
between uncertainty on/off), allowing regressions.

Decision
--------
1. Introduce explicit plot kinds: `factual_probabilistic`,
   `alternative_probabilistic`, `factual_regression`, `alternative_regression`.
   Each plot kind has a focused builder and renderer.
   `triangular` is a special plot that draws a special kind of alternative plot for probabilistic explanations. It is invoked by the style parameter `style=triangular`, following the legacy logic from v0.5.1.
   Additional plots can be added later.
2. Make semantics explicit in `PlotSpec` metadata:
   - header: `dual` (default factual probabilistic), `single` (default factual regression), `none` (default alternatives, and triangular), `...` (placeholder for future headers, e.g. true multiclass)
   - body: `weight` (default factual and fast), `predict` (default alternative, and triangular), `...` (placeholder for future plots)
   - style: `regular` (default factual, alternative, and fast), `triangular` (triangular), `...` (placeholder for future styles).
   - uncertainty: `true`/`false` (whether uncertainty is drawn, applicable to factual plots, default=false).
   - rank_by: `ensured` (default alternative), `feature_weight` (default factual and fast), `uncertainty`, `...` (placeholder for future ranking metrics).
   - rank_weight : float [-1, 1] (Used with the `ensured` ranking metric, the weight of the uncertainty in the ranking, default=0.5).
3. Adopt default legacy behaviour: if uncertainty=False, solid bars are drawn from 0 to the feature weight value; if uncertainty=True, solid bars are drawn to the uncertainty-area, and translucent uncertainty overlays replace the solid bars and cover the uncertainty area. If the uncertainty interval crosses zero, the solid bar is suppressed to avoid visual confusion. This matches the existing behaviour in v0.5.1. Only the legacy visual mode is supported by `BarItem` and `BarHPanelSpec`.
4. Add an adapter test hook `export_drawn_primitives` that returns a
   backend-agnostic description of drawn primitives (solids/overlays/header) for robust testing.
5. Add adapter primitive tests to assert legacy visual mode and parity between uncertainty on/off for each plot kind.
6. Add additional adapter options, allowing dynamic configuration of color roles and uncertainty visual style.

Consequences
------------
- Pros:
  - Clear separation reduces bugs and simplifies adapter implementations.
  - Tests can assert drawing semantics reliably via primitives export.
  - Legacy behaviour can be reproduced when needed.
- Cons:
  - Small API surface growth (PlotSpec metadata and adapter options).
  - Migration needed for consumers who depended on implicit behaviour.

Notes
-----
This ADR is a practical decision to stabilize visuals and tests.
Renderers are split into small functions per plot kind and the
builders updated to return consistent `PlotSpec` metadata.

Amendment: Global plots and PlotSpec precisions
----------------------------------------------
This ADR is amended (2025-09-29) to explicitly include global plots and
clarify several PlotSpec and adapter primitives conventions chosen by the
team to ensure full support for the legacy v0.5.1 plotting behaviour.

1) Global plots
   - New canonical kinds `global_probabilistic` and `global_regression`
     are added. These map to the legacy function `_plot_global` and must
     preserve the legacy semantics: scatter of uncertainty vs probability
     (or prediction), color/marker mapping by `y_test` when provided,
     and the regularized branch that draws the proba-triangle.

2) PlotSpec additions and clarifications
   - PlotSpec MUST include `mode` ('classification'|'regression'),
     `threshold` (None|scalar|tuple), and `class_index` (None|int) where
     relevant to disambiguate multiclass and thresholded-regression
     semantics.
   - For alternative plots the header field is `none` (no small header
     subplots). For factual probabilistic header='dual' (two small
     probability panels) and for factual regression header='single'
     (one top panel). The triangular plot is a separate kind `triangular`.
   - A `feature_order` array must be present in PlotSpec to guarantee
     deterministic ordering; builders must populate it from the
     Explanation-level ordering.
   - Coordinates in exported primitives MUST be in data-space (not pixels)
     and the primitives payload MUST include human-readable axis labels.

3) Uncertainty visual parity (legacy behaviour)
   - If `uncertainty=False`: draw solid bars from zero to feature weight.
   - If `uncertainty=True`: draw translucent uncertainty overlays between
     low/high and draw a solid bar only when the interval does not cross
     zero (otherwise the solid bar is suppressed). This is authoritative
     and must be reproduced by adapters.

4) Adapter primitives contract
   - Adapters SHOULD implement the `export_drawn_primitives()` hook
     that returns a JSON-serializable dict with a `plot_spec` and a list
     of `primitives`. Axis ids (e.g., `header.pos`, `header.neg`, `main`)
     must be supported so adapters can map logical axes to backend axes.
   - Colors MAY be validated by role (e.g., 'positive_fill') in tests. The
     default color mixing behaviour used in v0.5.1 (`__get_fill_color`) is
     documented and adapters should either reproduce it or present role
     based colors in primitives.

5) Tests and parity
   - Adapter tests will use `export_drawn_primitives` to assert semantic
     equivalence to legacy plots (happy path + edge cases such as
     zero-crossing, infinite bounds, threshold variants, multiclass).

If conflicts exist between this ADR and other plotting ADRs, this ADR
takes precedence for plotting behaviour and primitive schema.
